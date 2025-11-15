# app/web/preferences.py
import logging
from flask import Blueprint, jsonify, request, current_app

# Use relative imports
from ..security import csrf_protect, require_auth
from ..config import DEFAULT_USER_ID
from .validation import validate_json_request

# --- THIS IS THE MISSING LINE ---
bp = Blueprint('preferences', __name__, url_prefix='/preferences')
# ----------------------------------

logger = logging.getLogger(__name__)

@bp.route('/', methods=['GET'])
@require_auth
def get_preferences():
    """ GET /preferences - Gets user preferences. """
    try:
        prefs = current_app.memory_repo.get_user_preferences(DEFAULT_USER_ID)
        return jsonify(prefs), 200
    except Exception as e:
        logger.error(f"Get preferences error: {e}", exc_info=True)
        return jsonify({"error": "Failed to get preferences"}), 500

@bp.route('/listening-mode', methods=['POST'])
@csrf_protect
@require_auth
def toggle_listening_mode():
    """ POST /preferences/listening-mode - Toggles listening mode. """
    try:
        data, error_response, status_code = validate_json_request()
        if error_response: return error_response, status_code
        
        enabled = data.get('enabled', False)
        memory_policy = data.get('memory_policy', None)
        tts_muted = data.get('tts_muted', None)
        
        success = current_app.memory_repo.set_user_preferences(
            DEFAULT_USER_ID,
            listening_mode=enabled,
            listening_memory_policy=memory_policy,
            listening_tts_muted=tts_muted
        )
        
        if success:
            logger.info(f"Listening mode {'enabled' if enabled else 'disabled'} for user {DEFAULT_USER_ID}")
            return jsonify({
                "status": "success",
                "listening_mode": enabled
            }), 200
        else:
            return jsonify({"error": "Failed to update preferences"}), 500
            
    except Exception as e:
        logger.error(f"Toggle listening mode error: {e}", exc_info=True)
        return jsonify({"error": "Failed to update preferences"}), 500

