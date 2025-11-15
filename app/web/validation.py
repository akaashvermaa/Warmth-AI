# app/web/validation.py
import logging
from flask import request, jsonify

logger = logging.getLogger(__name__)

def validate_json_request():
    """Validates that request contains valid JSON."""
    if not request.is_json:
        return None, jsonify({"error": "Content-Type must be application/json"}), 415
    try:
        data = request.get_json(silent=False)
        if data is None:
            return None, jsonify({"error": "Invalid JSON in request body"}), 400
        return data, None, None
    except Exception as e:
        logger.warning(f"JSON parsing error: {e}")
        return None, jsonify({"error": "Malformed JSON"}), 400

def validate_message(message, max_length=5000, min_length=1):
    """Validates chat message input."""
    if not isinstance(message, str):
        return False, "Message must be a string"
    if len(message.strip()) < min_length:
        return False, f"Message must be at least {min_length} character(s)"
    if len(message) > max_length:
        return False, f"Message exceeds maximum length of {max_length} characters"
    return True, None

def validate_memory_id(mem_id):
    """Validates memory ID input."""
    if not isinstance(mem_id, (int, str)):
        return False, "Memory ID must be an integer or numeric string"
    try:
        int_id = int(mem_id)
        if int_id <= 0:
            return False, "Memory ID must be a positive integer"
        return True, int_id
    except (ValueError, TypeError):
        return False, "Memory ID must be a valid integer"