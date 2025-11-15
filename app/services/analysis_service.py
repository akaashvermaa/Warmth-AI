import logging
import re
import string
from functools import lru_cache
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

# === NLTK Caching ===
# Cache tokenizer and lemmatizer results to avoid repeated heavy calls
_tokenize_cache = {}
_MAX_CACHE_SIZE = 1000  # Limit cache size to prevent memory issues

# Create a single lemmatizer instance to reuse (expensive to create)
_shared_lemmatizer = WordNetLemmatizer()

def _cached_tokenize(text):
    """Cached version of word_tokenize to avoid repeated heavy calls."""
    if text in _tokenize_cache:
        return _tokenize_cache[text]
    
    try:
        tokens = word_tokenize(text)
    except LookupError:
        # Fallback tokenizer when NLTK punkt data is missing.
        # This keeps the app working even if the user hasn't downloaded punkt.
        logger.warning("NLTK tokenizer resource missing (punkt). Using fallback tokenizer.")
        # A simple regex-based tokenizer: words, including contractions and hyphenated words
        tokens = re.findall(r"\w+(?:['-]\w+)?", text)
    
    # Simple LRU: remove oldest if cache is full
    if len(_tokenize_cache) >= _MAX_CACHE_SIZE:
        # Remove first (oldest) item
        _tokenize_cache.pop(next(iter(_tokenize_cache)))
    
    _tokenize_cache[text] = tokens
    return tokens

@lru_cache(maxsize=_MAX_CACHE_SIZE)
def _cached_lemmatize(word):
    """Cached version of lemmatizer to avoid repeated heavy calls."""
    return _shared_lemmatizer.lemmatize(word)

class MoodAnalyzer:
    def __init__(self, 
                 great_threshold=0.5, 
                 good_threshold=0.05, 
                 heavy_threshold=-0.5, 
                 low_threshold=-0.05):
        """
        Initializes the analyzer with tunable thresholds.
        """
        self.analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation_table = str.maketrans('', '', string.punctuation)

        # --- Improvement 4: Tunable Thresholds ---
        # The thresholds are now passed in during initialization
        self.great_t = great_threshold
        self.good_t = good_threshold
        self.heavy_t = heavy_threshold
        self.low_t = low_threshold

        # Raw topics as defined by you
        raw_topics = {
            'work': ['work', 'job', 'boss', 'office', 'deadline', 'meeting', 'career'],
            'love': ['love', 'date', 'breakup', 'partner', 'crush', 'relationship', 'ex'],
            'health': ['sick', 'tired', 'pain', 'sleep', 'gym', 'workout', 'health', 'headache'],
            'family': ['mom', 'dad', 'sister', 'brother', 'family', 'parents', 'home'],
            'money': ['money', 'expensive', 'broke', 'buy', 'cost', 'rent', 'bill'],
            'future': ['hope', 'dream', 'goal', 'plan', 'worried', 'scared', 'nervous']
        }
        
        # --- Improvement 2: Normalization & Lemmatization (for keywords) ---
        # We pre-process all keywords one time for faster matching later
        self.processed_topics = self._process_topics(raw_topics)

    def _process_topics(self, topics):
        """
        Internal helper to lemmatize all keywords at initialization.
        """
        processed = {}
        for category, keywords in topics.items():
            processed[category] = [
                self.lemmatizer.lemmatize(k.lower()) for k in keywords
            ]
        return processed

    # --- Improvement 2: Normalization & Lemmatization (for input text) ---
    def _normalize_and_lemmatize(self, text):
        """
        Cleans, normalizes, and lemmatizes the input text for topic matching.
        Uses cached tokenizer and lemmatizer for performance.
        """
        # 1. Normalize: Lowercase
        text = text.lower()
        # 2. Normalize: Remove punctuation
        text = text.translate(self.punctuation_table)
        # 3. Tokenize: Split into words (cached)
        tokens = _cached_tokenize(text)
        # 4. Lemmatize: Reduce words to their root form (cached)
        lemmatized_tokens = [_cached_lemmatize(token) for token in tokens]
        
        # Return as a single string for easy regex searching
        return " ".join(lemmatized_tokens)

    # --- Improvement 3: Expose get_score_0_100() helper ---
    def get_score_0_100(self, compound_score, v_min=-1.0, v_max=1.0):
        """
        Public helper to map a VADER score (-1 to 1) to a 0-100 scale.
        """
        # (value - min) / (max - min) gives a 0-1 scale
        # Then multiply by 100
        return ((compound_score - v_min) / (v_max - v_min)) * 100

    def analyze_sentiment(self, text):
        """
        Analyzes sentiment and detects topics using improved methods.
        """
        # 1. SENTIMENT ANALYSIS
        # VADER works best on raw text with punctuation
        score = self.analyzer.polarity_scores(text)['compound']
        
        # Use the tunable thresholds from __init__
        if score >= self.great_t: 
            label = "Great"
        elif score >= self.good_t: 
            label = "Good"
        elif score <= self.heavy_t: 
            label = "Heavy"
        elif score <= self.low_t: 
            label = "Low"
        else: 
            label = "Neutral"

        # 2. TOPIC DETECTION
        # Use the normalized/lemmatized text for matching
        processed_text = self._normalize_and_lemmatize(text)
        
        detected_topic = "General"
        
        for category, keywords in self.processed_topics.items():
            topic_found = False
            for keyword in keywords:
                # --- Improvement 1: Word-Boundary Checks ---
                # Use \b for word boundaries.
                # Use re.escape() to safely handle keywords with special characters
                pattern = r'\b' + re.escape(keyword) + r'\b'
                
                if re.search(pattern, processed_text):
                    detected_topic = category.capitalize()
                    topic_found = True
                    break  # Stop searching keywords in this category
            
            if topic_found:
                break  # Stop searching other categories
        
        return score, label, detected_topic

    def get_mood(self, text):
        """
        This method is unchanged as its logic is simple.
        """
        score = self.analyzer.polarity_scores(text)['compound']
        return 'caring' if score < -0.3 else 'playful'

# --- Example Usage ---
if __name__ == "__main__":
    # You can now tune the thresholds when creating the class
    analyzer = MoodAnalyzer(great_threshold=0.6, low_threshold=-0.2)

    # Test 1: "work" vs "network"
    text1 = "My new job is great, but the office network is slow."
    score1, label1, topic1 = analyzer.analyze_sentiment(text1)
    print(f"Text: '{text1}'")
    print(f"-> Score: {score1}, Label: {label1}, Topic: {topic1}")
    print("---")

    # Test 2: Lemmatization ("meetings" -> "meeting")
    text2 = "I have three meetings today about my career plans."
    score2, label2, topic2 = analyzer.analyze_sentiment(text2)
    print(f"Text: '{text2}'")
    print(f"-> Score: {score2}, Label: {label2}, Topic: {topic2}")
    print("---")

    # Test 3: Negative sentiment
    text3 = "I'm so worried about my deadline. My boss will be angry."
    score3, label3, topic3 = analyzer.analyze_sentiment(text3)
    print(f"Text: '{text3}'")
    print(f"-> Score: {score3}, Label: {label3}, Topic: {topic3}")
    print("---")
    
    # Test 4: Using the new 0-100 helper
    score_100 = analyzer.get_score_0_100(score3)
    print(f"Score {score3} on a 0-100 scale is: {score_100:.2f}")