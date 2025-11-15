# app/__init__.py
import os
import logging
import threading
import time
from flask import Flask, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# --- Core App Imports ---
from . import config
from .security import get_or_create_password_hash

# --- Service Layer Imports ---
from .services.llm_service import LLMService
from .services.analysis_service import MoodAnalyzer
from .services.safety_service import SafetyNet
from .services.embedding_service import EmbeddingManager
from .services.cache_service import CacheManager
from .services.chat_service import ChatService

# --- Storage Layer Import ---
from .storage.memory_repository import MemoryRepository

# --- Web Blueprint Imports ---
from .web import main, auth, errors, memory, mood, preferences

# --- Initialize Extensions ---
cors = CORS()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per day", "30 per hour", "10 per minute"],
    storage_uri="memory://",
    headers_enabled=True
)

# --- Service Instantiation ---
logger = logging.getLogger(__name__)

try:
    # We must use the .config file that is inside the app/ folder
    memory_repo = MemoryRepository(db_name=config.DB_NAME) 
    llm_service = LLMService()
    analysis_service = MoodAnalyzer()
    safety_service = SafetyNet()
    cache_manager = CacheManager()
    
    chat_service = ChatService(
        memory_repo=memory_repo,
        llm_service=llm_service,
        analysis_service=analysis_service,
        safety_service=safety_service,
        cache_manager=cache_manager
    )
    
except Exception as e:
    logger.critical(f"CRITICAL: Failed to initialize core services: {e}", exc_info=True)
    raise

def create_app():
    """
    Application Factory: Creates and configures the Flask app.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # === THE FIX IS HERE ===
    # We remove all arguments. Flask's default is to look for 'static' and 'templates'
    # in the SAME directory as this __init__.py file, which is correct for your structure.
    app = Flask(__name__)
    
    # === Load Config ===
    app.config.from_object(config)
    
    # Set secret key for sessions
    app.secret_key = app.config.get('FLASK_SECRET_KEY', os.urandom(32).hex())
    
    # Load password hash into config
    app.config['WARMTH_PASSWORD_HASH'] = get_or_create_password_hash()

    # === Initialize Extensions ===
    cors.init_app(app, origins=app.config.get('ALLOWED_ORIGINS', '*').split(','), supports_credentials=True)
    limiter.init_app(app)

    # === Register Blueprints (Web Routes) ===
    app.register_blueprint(errors.bp)
    app.register_blueprint(main.bp)
    app.register_blueprint(auth.bp)
    app.register_blueprint(memory.bp)
    app.register_blueprint(mood.bp)
    app.register_blueprint(preferences.bp)
    
    # === Inject Services onto the App Object ===
    app.chat_service = chat_service
    app.memory_repo = memory_repo
    
    # === Background Tasks ===
    schedule_memory_expiry(app, memory_repo)

    return app

# --- Background Task (Helper) ---
def schedule_memory_expiry(app, repo: MemoryRepository):
    
    from .config import MEMORY_EXPIRY_ENABLED, MEMORY_EXPIRY_CHECK_INTERVAL
    
    if not MEMORY_EXPIRY_ENABLED:
        return
    
    def _expire_loop():
        while True:
            try:
                time.sleep(MEMORY_EXPIRY_CHECK_INTERVAL)
                with app.app_context():
                    deleted = repo.expire_old_memories()
                    if deleted > 0:
                        app.logger.info(f"Auto-expired {deleted} memory(ies)")
            except Exception as e:
                app.logger.error(f"Memory expiry task error: {e}")
    
    expiry_thread = threading.Thread(target=_expire_loop, daemon=True, name="MemoryExpiry")
    expiry_thread.start()
    app.logger.info(f"Memory expiry task started (check interval: {MEMORY_EXPIRY_CHECK_INTERVAL}s)")