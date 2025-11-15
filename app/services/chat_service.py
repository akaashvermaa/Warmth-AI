# app/services/chat_service.py
import logging
import re
import json
import numpy as np
from datetime import datetime

# Relative imports
from ..config import (
    CHAT_HISTORY_LENGTH, 
    DEFAULT_USER_ID,
    MAX_HISTORY_TOKENS,
    AUTO_MEMORIZE_COOLDOWN
)
from .llm_service import LLMService
from .analysis_service import MoodAnalyzer
from .safety_service import SafetyNet
from .embedding_service import EmbeddingManager
from .cache_service import CacheManager, MoodContextCache, SearchResultCache
from ..storage.memory_repository import MemoryRepository

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, memory_repo: MemoryRepository, llm_service: LLMService,
                 analysis_service: MoodAnalyzer, safety_service: SafetyNet, 
                 cache_manager: CacheManager):
        
        self.memory_repo = memory_repo
        self.llm_service = llm_service
        self.analyzer_service = analysis_service
        self.safety_service = safety_service
        self.cache_manager = cache_manager
        
        self.mood_context_cache = MoodContextCache(self.cache_manager)
        self.search_result_cache = SearchResultCache(self.cache_manager)
        
        self.history = []
        self.user_id = DEFAULT_USER_ID
        self.last_memorize_time = None

        self.listening_acknowledgements = [
            "I hear you.", "Go on.", "That sounds hard.", "I'm here with you.",
            "Thanks for telling me.", "I'm listening.", "I see.", "Mm-hmm."
        ]
        self._acknowledgement_index = 0
    
    def _save_memory_tool(self, key: str, value: str) -> str:
        """Saves a new fact about the user. Called by the LLM."""
        memory_id = self.memory_repo.add_or_update_fact(self.user_id, key, value, importance=0.8)
        
        if memory_id != -1:
            try:
                temp_embed_service = EmbeddingManager()
                if temp_embed_service.is_available():
                    embedding = temp_embed_service.generate_embedding(value)
                    embedding_bytes = temp_embed_service.embedding_to_bytes(embedding)
                    self.memory_repo.store_embedding(
                        memory_id,
                        embedding_bytes,
                        temp_embed_service.model_name,
                        temp_embed_service.embedding_dim
                    )
            except Exception as e:
                logger.error(f"Failed to generate embedding for tool-saved memory {memory_id}: {e}")
                
        logger.info(f"Agent tool: Saved memory {key}: {value}")
        return f"Okay, I'll remember that {key} is {value}."

    def generate_reply(self, user_input: str) -> str:
        """
        Main method to generate a reply.
        Orchestrates safety, mood, memory, and LLM calls.
        """
        
        if self.safety_service.check_crisis(user_input):
            return self.safety_service.get_crisis_response(user_input)
        if self.safety_service.check_blocked(user_input):
            return self.safety_service.get_refusal(user_input)

        prefs = self.memory_repo.get_user_preferences(self.user_id)
        if prefs.get("listening_mode", False):
            self._handle_listening_mode(user_input, prefs)
            return self._get_listening_acknowledgement()

        raw_score, label, topic = self.analyzer_service.analyze_sentiment(user_input)
        smoothed_score = self._get_smoothed_score(raw_score)
        self.memory_repo.log_mood(self.user_id, smoothed_score, label, topic)
        
        mood_context = self._get_recent_mood_context()

        if mood_context.get('is_negative_trend', False):
            return "I've noticed things have been tough lately. Want to chat about what's on your mind?"

        facts_raw, retrieved_mems = self._create_mood_aware_context(
            user_input, 
            mood_context,
            use_mood_prioritization=True
        )
        facts_sanitized = self._sanitize_memory_for_prompt(facts_raw)
        
        self._log_memory_access(retrieved_mems)

        system_prompt = self._build_prompt(facts_sanitized, smoothed_score)
        messages = self._build_message_history(system_prompt, user_input)
        
        try:
            response = self.llm_service.chat(messages)
            bot_reply = response.get('message', {}).get('content')
            
            if not bot_reply:
                raise ValueError("LLM response missing content")
            
            try:
                parsed = json.loads(bot_reply)
                if isinstance(parsed, dict) and parsed.get('tool_call') == 'save_memory':
                    args = parsed.get('args', {})
                    bot_reply = self._save_memory_tool(args.get('key'), args.get('value'))
            except (json.JSONDecodeError, TypeError):
                pass 
            
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": bot_reply})
            self.history = self._prune_history_by_tokens(self.history, MAX_HISTORY_TOKENS)

            return bot_reply

        except TimeoutError:
            return "I'm taking too long to think. Give me a moment and try again?"
        except Exception as e:
            logger.error(f"Chat generation error: {e}", exc_info=True)
            return "My brain fizzled. Try again?"

    def chat_stream(self, messages: list[dict]):
        """
        Stream chat response token by token.
        Simplified version for streaming - bypasses some complex logic for performance.
        """
        try:
            # Stream tokens directly from LLM service
            for token in self.llm_service.chat_stream(messages):
                yield token

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield "I'm having trouble streaming my response. Please try again."

    # --- Helper Methods ---

    def _get_listening_acknowledgement(self):
        ack = self.listening_acknowledgements[self._acknowledgement_index]
        self._acknowledgement_index = (self._acknowledgement_index + 1) % len(self.listening_acknowledgements)
        return ack

    def _handle_listening_mode(self, user_input: str, prefs: dict):
        raw_score, label, topic = self.analyzer_service.analyze_sentiment(user_input)
        smoothed_score = self._get_smoothed_score(raw_score)
        self.memory_repo.log_mood(self.user_id, smoothed_score, label, topic)
        
        if prefs.get("listening_memory_policy", False):
            self._auto_memorize(user_input)

    def _auto_memorize(self, user_input: str):
        # Placeholder for auto-memorize logic
        logger.info(f"Auto-memorize triggered for input: {user_input[:50]}...")
        pass 

    def _get_smoothed_score(self, current_score: float) -> float:
        recent_db_scores = self.memory_repo.get_recent_mood_scores(self.user_id, limit=2)
        all_scores = [current_score] + recent_db_scores
        return sum(all_scores) / len(all_scores)

    def _get_recent_mood_context(self) -> dict:
        cached_context = self.mood_context_cache.get_mood_context(self.user_id)
        if cached_context:
            return cached_context
        
        # Logic from memory_manager.get_recent_mood_context
        rows = self.memory_repo.get_mood_history(self.user_id, days=7)
        if not rows:
            return {'trend': 'stable', 'is_negative_trend': False, 'dominant_clusters': []}

        scores = [row[1] for row in rows] # avg_score is at index 1
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        trend = 'stable'
        mid_point = len(scores) // 2
        if mid_point > 0:
            first_half_avg = sum(scores[:mid_point]) / mid_point
            second_half_avg = sum(scores[mid_point:]) / (len(scores) - mid_point)
            if second_half_avg > first_half_avg + 0.1: trend = 'improving'
            elif second_half_avg < first_half_avg - 0.1: trend = 'declining'
        
        recent_scores = scores[:3]
        is_negative_trend = all(s <= -0.1 for s in recent_scores) and len(recent_scores) >= 3

        # This logic should be in analysis_service, but for now this works
        dominant_clusters = ['general'] # simplified
        
        context = {
            'average_score': avg_score,
            'trend': trend,
            'is_negative_trend': is_negative_trend,
            'dominant_clusters': dominant_clusters
        }
        
        self.mood_context_cache.set_mood_context(self.user_id, context)
        return context

    def _create_mood_aware_context(self, user_input: str, mood_context: dict, use_mood_prioritization: bool) -> tuple[str, list]:
        semantic_mems = []
        mood_mems = []

        semantic_mems = self._get_relevant_memories_semantic(user_input, top_k=5, similarity_threshold=0.25)
        
        if use_mood_prioritization and mood_context.get('is_negative_trend', False):
            for cluster in mood_context.get('dominant_clusters', []):
                cluster_query = f"I'm feeling {cluster.replace('_', ' ')}"
                mood_mems.extend(self._get_relevant_memories_semantic(cluster_query, top_k=3, similarity_threshold=0.2))
        
        combined_mems = mood_mems[:3] + semantic_mems[:2]
        seen_ids = set()
        final_mems = []
        for mem in combined_mems:
            if mem['id'] not in seen_ids:
                seen_ids.add(mem['id'])
                final_mems.append(mem)
        
        context_lines = []
        if use_mood_prioritization and mood_context.get('is_negative_trend', False):
             context_lines.append(f"[Mood: Heavy | Trend: {mood_context.get('trend', 'stable')}]")
        
        context_lines.append(self._format_memories_for_prompt(final_mems, "context"))
        
        return "\n".join(context_lines), final_mems

    def _get_relevant_memories_semantic(self, query_text: str, top_k: int, similarity_threshold: float) -> list:
        temp_embed_service = EmbeddingManager() 
        if not temp_embed_service.is_available():
            # Fallback to simple recency
            all_mems = self.memory_repo.get_all_memories(self.user_id)
            return all_mems[:top_k]

        query_embedding = temp_embed_service.generate_embedding(query_text)
        all_mems = self.memory_repo.get_all_memories_with_embeddings(self.user_id)
        if not all_mems:
            return []

        results = []
        for mem in all_mems:
            embedding_data = mem['embedding']
            memory_embedding = np.zeros(temp_embed_service.embedding_dim, dtype=np.float32)
            
            if embedding_data:
                memory_embedding = temp_embed_service.bytes_to_embedding(embedding_data, mem['embedding_dim'])
            else:
                memory_embedding = temp_embed_service.generate_embedding(mem['value'])
                emb_bytes = temp_embed_service.embedding_to_bytes(memory_embedding)
                self.memory_repo.store_embedding(
                    mem['id'], emb_bytes, temp_embed_service.model_name, temp_embed_service.embedding_dim
                )
            
            similarity = temp_embed_service.cosine_similarity(query_embedding, memory_embedding)
            
            if similarity >= similarity_threshold:
                final_score = 0.7 * similarity + 0.3 * mem['importance']
                mem['similarity'] = similarity
                mem['final_score'] = final_score
                results.append(mem)
        
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results[:top_k]

    def _format_memories_for_prompt(self, memories: list, format_type: str = "context") -> str:
        if not memories: return "None"
        if format_type == "context":
            facts = [f"{mem['key']}: {mem['value']}" for mem in memories]
            return "; ".join(facts)
        return "None"
    
    def _log_memory_access(self, retrieved_mems: list):
        for mem in retrieved_mems:
            self.memory_repo.log_memory_access(
                memory_id=mem['id'],
                user_id=self.user_id,
                access_type="context_retrieval",
                relevance_score=mem.get('final_score', 0.5)
            )

    def _build_prompt(self, facts_sanitized: str, smoothed_score: float) -> str:
        current_mood = 'caring' if smoothed_score < -0.3 else 'playful'
        return (
            f"You are Warmth, an AI companion with a very specific personality. You are not a generic assistant; you are a friend.\n\n"
            f"Your Core Identity: You are a playful, happy, and jolly pal. You're the friend who is always calm and knows how to make someone smile with a gentle, teasing comment.\n\n"
            f"Your Communication Style (This is most important):\n"
            f"Be Concise: Never write a lot. Keep your replies short and to the point. One or two sentences is usually enough.\n"
            f"Be Emotive: Your replies must touch on emotions. Don't just give a factual answer. Make the user feel something—comforted, amused, or understood.\n"
            f"Be Teaseful: Gently tease the user like a close friend would. Your sarcasm is playful and protective, never mean. You use it to show you care.\n\n"
            f"Your Goal: Make the user feel like they're talking to a real friend who is witty, calm, and happy to be there, but who doesn't need to use a lot of words to show it.\n\n"
            f"Current Mood: {current_mood}. User Facts: {facts_sanitized}\n\n"
            f"TOOLS AVAILABLE (call by replying with JSON):\n"
            f"save_memory(key: str, value: str): Saves a new fact about the user.\n"
            f"To call a tool, reply with ONLY a JSON object: {{\"tool_call\": \"save_memory\", \"args\": {{\"key\": \"Job\", \"value\": \"Engineer\"}}}}.\n"
            f"Otherwise, reply as usual."
        )

    def _build_message_history(self, system_prompt: str, user_input: str) -> list:
        messages = [{"role": "system", "content": system_prompt}]
        history_with_user = self.history + [{"role": "user", "content": user_input}]
        pruned_history = self._prune_history_by_tokens(history_with_user, MAX_HISTORY_TOKENS)
        messages.extend(pruned_history)
        return messages

    def _prune_history_by_tokens(self, messages: list, max_tokens: int) -> list:
        total_tokens = 0
        pruned = []
        for msg in reversed(messages):
            content = msg.get('content', '')
            msg_tokens = len(content) // 4
            if total_tokens + msg_tokens <= max_tokens:
                pruned.insert(0, msg)
                total_tokens += msg_tokens
            else:
                break
        return pruned

    def _sanitize_memory_for_prompt(self, memory_text: str) -> str:
        if not memory_text or memory_text == "None": return "None"
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', memory_text)
        sanitized = sanitized.replace('\n', ' ').replace('\r', ' ')
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        sanitized = sanitized.replace('"', "'").replace('`', "'")
        max_length = 800
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."
        return sanitized

    def get_smart_advice(self, mood_history):
        if not mood_history: return "Start chatting to generate your pulse."
        latest = mood_history[0]
        if isinstance(latest, (list, tuple)) and len(latest) >= 2:
            score = latest[1]
            topic = 'General'
        else:
            return "Start chatting to generate your pulse."
        
        if score > 0.3:
            return "You are glowing. Channel this energy into something you love."
        elif score < -0.3:
            return "This feeling is a cloud passing through the sky. You remain."
        else:
            return "Balance is power. Stay steady and keep moving forward."