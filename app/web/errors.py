# app/web/errors.py
import logging
from flask import Blueprint, jsonify, request
from flask_limiter.util import get_remote_address

bp = Blueprint('errors', __name__)
logger = logging.getLogger(__name__)

@bp.app_errorhandler(500)
def handle_500_error(error):
    """Global handler for 500 Internal Server Errors."""
    logger.error(f"500 Internal Server Error: {error}", exc_info=True)
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred. Please try again later."
    }), 500

@bp.app_errorhandler(404)
def handle_404_error(error):
    """Global handler for 404 Not Found errors."""
    logger.warning(f"404 Not Found: {request.path}")
    return jsonify({
        "error": "Not Found",
        "message": "The requested resource was not found."
    }), 404

@bp.app_errorhandler(429)
def handle_429_error(error):
    """Global handler for 429 Too Many Requests (rate limiting)."""
    logger.warning(f"429 Rate Limit Exceeded: {get_remote_address()}")
    return jsonify({
        "error": "Too Many Requests",
        "message": "Rate limit exceeded. Please slow down and try again later."
    }), 429

@bp.app_errorhandler(415)
def handle_415_error(error):
    """Global handler for 415 Unsupported Media Type."""
    return jsonify({
        "error": "Unsupported Media Type",
        "message": "Content-Type must be application/json"
    }), 415

@bp.app_errorhandler(401)
def handle_401_error(error):
    """Global handler for 401 Unauthorized."""
    return jsonify({
        "error": "Unauthorized",
        "message": "Authentication required"
    }), 401

@bp.app_errorhandler(403)
def handle_403_error(error):
    """Global handler for 403 Forbidden (CSRF)."""
    return jsonify({
        "error": "Forbidden",
        "message": "CSRF token validation failed"
    }), 403