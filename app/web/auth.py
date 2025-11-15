# app/web/auth.py
import os
import logging
from flask import Blueprint, request, jsonify, session, current_app

# THE FIX: Use '..' to go up to the 'app' folder
from ..security import verify_password, generate_csrf_token, secure_log_message

# THE FIX: Use relative import
from .validation import validate_json_request

bp = Blueprint('auth', __name__, url_prefix='/auth')
logger = logging.getLogger(__name__)



@bp.route('/login', methods=['POST'])
def login():
    """
    POST /auth/login
    Authenticates user with password.
    """
    try:
        data, error_response, status_code = validate_json_request()
        if error_response:
            return error_response, status_code
        
        if 'password' not in data:
            return jsonify({"error": "Missing password"}), 400
        
        password = data['password']
        # Get hash from the app config, set during startup
        password_hash = current_app.config['WARMTH_PASSWORD_HASH']
        
        if verify_password(password, password_hash):
            session['authenticated'] = True
            csrf_token = generate_csrf_token()
            logger.info("User authenticated successfully")
            return jsonify({
                "status": "success",
                "csrf_token": csrf_token
            }), 200
        else:
            secure_log_message("Failed login attempt", "warning")
            return jsonify({"error": "Invalid password"}), 401
            
    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error"}), 500

@bp.route('/logout', methods=['POST'])
def logout():
    """Logs out the current user."""
    session.pop('authenticated', None)
    session.pop('csrf_token', None)
    return jsonify({"status": "success"}), 200

@bp.route('/auth/status', methods=['GET'])
def auth_status():
    """Returns authentication status and CSRF token if authenticated."""
    enable_auth = current_app.config.get('ENABLE_AUTH', False)
    is_authenticated = session.get('authenticated', False)
    
    response = {
        "auth_enabled": enable_auth,
        "authenticated": is_authenticated
    }
    
    if is_authenticated:
        response["csrf_token"] = generate_csrf_token()
    
    return jsonify(response), 200