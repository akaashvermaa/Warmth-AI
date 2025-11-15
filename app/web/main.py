# app/web/main.py
import os
import logging
import hashlib
from flask import (
    Blueprint, 
    request, 
    jsonify, 
    render_template,  
    send_from_directory,
    Response,
    make_response,
    current_app
)

# Use relative imports
from ..security import csrf_protect, require_auth, secure_log_message, generate_csrf_token
from ..config import DEFAULT_USER_ID
from .validation import validate_json_request, validate_message

# Import services
# from ..services.whisper_utils import transcribe_audio # Keep this commented out

bp = Blueprint('main', __name__)
logger = logging.getLogger(__name__)

# Cache for rendered templates
_template_cache = {}
def get_template_hash(template_name, **kwargs):
    """Generate hash for template cache key."""
    cache_key = f"{template_name}:{str(sorted(kwargs.items()))}"
    return hashlib.md5(cache_key.encode()).hexdigest()

@bp.route('/transcribe', methods=['POST'])
def transcribe():
    # This route is now disabled
    logger.warning("Attempted to call /transcribe, but Whisper is disabled.")
    return jsonify({"error": "Transcription service is not enabled"}), 503

@bp.route('/')
def home():
    """Serves the main index.html chat UI with caching."""
    try:
        csrf_token = generate_csrf_token() if os.getenv('ENABLE_CSRF', 'false').lower() == 'true' else None
        
        debug_mode = current_app.config.get('FLASK_DEBUG', False)
        cache_key = get_template_hash('index.html', csrf_token=csrf_token)
        
        if not debug_mode and cache_key in _template_cache:
            response = make_response(_template_cache[cache_key])
        else:
            rendered = render_template('index.html', csrf_token=csrf_token)
            if not debug_mode:
                _template_cache[cache_key] = rendered
            response = make_response(rendered)
        
        response.headers['Cache-Control'] = 'public, max-age=3600'
        response.headers['ETag'] = f'"{cache_key}"'
        return response, 200
    except Exception as e:
        logger.error(f"Error rendering index.html: {e}", exc_info=True)
        return "Template not found.", 404

@bp.route('/static/<path:path>')
def send_static(path):
    """Serves static files."""
    try:
        return send_from_directory(current_app.static_folder, path)
    except Exception:
        logger.warning(f"Static file not found: {path}")
        return jsonify({"error": "File not found"}), 404

@bp.route('/chat', methods=['POST'])
@csrf_protect
@require_auth
def chat():
    """
    POST /chat
    Sends a message to the bot and receives a reply.
    """
    try:
        data, error_response, status_code = validate_json_request()
        if error_response: return error_response, status_code
        
        if 'message' not in data:
            return jsonify({"error": "Missing required field: 'message'"}), 400
        
        user_message = data['message']
        is_valid, error_msg = validate_message(user_message, max_length=5000)
        if not is_valid:
            return jsonify({"error": error_msg}), 400
        
        secure_log_message(user_message, "info")
        
        # Use the injected chat_service
        reply = current_app.chat_service.generate_reply(user_message)

        return jsonify({"reply": reply}), 200
        
    except Exception as e:
        logger.error(f"POST /chat - 500 Internal Server Error: {e}", exc_info=True)
        raise

@bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({"status": "healthy", "service": "warmth"}), 200