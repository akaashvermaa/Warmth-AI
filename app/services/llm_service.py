# app/services/llm_service.py
import os
import logging
import time
import concurrent.futures
import ollama
from ..config import OLLAMA_MODEL, OLLAMA_TIMEOUT, SLOW_RESPONSE_THRESHOLD

logger = logging.getLogger(__name__)
llm_logger = logging.getLogger(f"{__name__}.llm")

try:
    ollama.list()
    _OLLAMA_AVAILABLE = True
    logger.info(f"Ollama connection successful. Default model: {OLLAMA_MODEL}")
except Exception as e:
    _OLLAMA_AVAILABLE = False
    logger.warning(f"Ollama not available. LLM calls will be stubbed. Error: {e}")


class LLMService:
    def __init__(self):
        self.model = OLLAMA_MODEL
        self.heavy_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.getenv('HEAVY_TASK_WORKERS', '4')),
            thread_name_prefix='HeavyTask'
        )

    def is_available(self):
        return _OLLAMA_AVAILABLE

    def chat(self, messages: list[dict]) -> dict:
        start_time = time.time()
        
        def _ollama_call():
            if not _OLLAMA_AVAILABLE:
                raise RuntimeError("Ollama not available")
            try:
                return ollama.chat(model=self.model, messages=messages)
            except Exception as e:
                logger.error(f"Ollama call exception: {e}", exc_info=True)
                raise

        try:
            future = self.heavy_executor.submit(_ollama_call)
            response = future.result(timeout=OLLAMA_TIMEOUT)
            elapsed = time.time() - start_time
            
            if elapsed > SLOW_RESPONSE_THRESHOLD:
                llm_logger.warning("Slow LLM response", extra={"response_time": round(elapsed, 2), "threshold": SLOW_RESPONSE_THRESHOLD, "model": self.model})
            else:
                llm_logger.info("LLM response", extra={"response_time": round(elapsed, 2), "model": self.model})
            
            return response
            
        except concurrent.futures.TimeoutError:
            elapsed = time.time() - start_time
            llm_logger.error("LLM timeout", extra={"timeout_seconds": OLLAMA_TIMEOUT, "elapsed_time": round(elapsed, 2), "model": self.model})
            raise TimeoutError(f"Ollama request timed out after {OLLAMA_TIMEOUT} seconds")
            
        except Exception as e:
            elapsed = time.time() - start_time
            llm_logger.error("LLM error", extra={"error_type": type(e).__name__, "error_message": str(e), "model": self.model}, exc_info=True)
            raise
        except RuntimeError:
            llm_logger.warning("Ollama not available; returning fallback response")
            return {"message": {"content": "(LLM not available locally)"}}