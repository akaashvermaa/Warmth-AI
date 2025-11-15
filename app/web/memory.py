# app/web/memory.py
import logging
from flask import Blueprint, request, jsonify, Response, current_app
from datetime import datetime

# THE FIX: Use '..' to go up
from ..security import csrf_protect, require_auth, encrypt_data
from ..config import DEFAULT_USER_ID

# THE FIX: Use relative import
from .validation import validate_json_request, validate_memory_id

bp = Blueprint('memory', __name__, url_prefix='/memory')
logger = logging.getLogger(__name__)

@bp.route('/', methods=['GET'])
@require_auth
def get_memories():
    try:
        memories = current_app.memory_repo.get_all_memories(DEFAULT_USER_ID)
        return jsonify({"memories": memories}), 200
    except Exception as e:
        logger.error(f"GET /memory - 500 Internal Server Error: {e}", exc_info=True)
        raise

@bp.route('/forget', methods=['POST'])
@csrf_protect
@require_auth
def forget_memory():
    try:
        data, error_response, status_code = validate_json_request()
        if error_response: return error_response, status_code
        
        if 'id' not in data:
            return jsonify({"error": "Missing required field: 'id'"}), 400
        
        mem_id = data.get('id')
        is_valid, result = validate_memory_id(mem_id)
        if not is_valid:
            return jsonify({"error": result}), 400
        
        validated_id = result
        
        rows_deleted = current_app.memory_repo.forget_memory(DEFAULT_USER_ID, validated_id)
        
        if rows_deleted == 0:
            return jsonify({"error": "Memory ID not found"}), 404
        
        return jsonify({"status": "success"}), 200
        
    except Exception as e:
        logger.error(f"POST /memory/forget - 500 Internal Server Error: {e}", exc_info=True)
        raise

@bp.route('/export-all', methods=['GET'])
@require_auth
def export_all_data():
    try:
        export_data = current_app.memory_repo.export_all_data(DEFAULT_USER_ID)
        return jsonify(export_data), 200
    except Exception as e:
        logger.error(f"Export all data error: {e}", exc_info=True)
        return jsonify({"error": "Export failed"}), 500

@bp.route('/erase-all', methods=['POST'])
@csrf_protect
@require_auth
def erase_all_data():
    try:
        deleted_counts = current_app.memory_repo.erase_all_data(DEFAULT_USER_ID)
        logger.warning(f"User {DEFAULT_USER_ID} erased all data")
        return jsonify({
            "status": "success",
            "deleted": deleted_counts
        }), 200
    except Exception as e:
        logger.error(f"Erase all data error: {e}", exc_info=True)
        return jsonify({"error": "Erase failed"}), 500

@bp.route('/maintenance/vacuum', methods=['POST'])
@csrf_protect
@require_auth
def vacuum_database():
    try:
        success = current_app.memory_repo.vacuum()
        if success:
            return jsonify({"status": "success", "message": "Database optimized"}), 200
        else:
            return jsonify({"error": "VACUUM failed"}), 500
    except Exception as e:
        logger.error(f"VACUUM error: {e}", exc_info=True)
        return jsonify({"error": "VACUUM failed"}), 500

@bp.route('/maintenance/expire', methods=['POST'])
@csrf_protect
@require_auth
def expire_memories():
    try:
        deleted_count = current_app.memory_repo.expire_old_memories(DEFAULT_USER_ID)
        return jsonify({
            "status": "success",
            "expired_count": deleted_count
        }), 200
    except Exception as e:
        logger.error(f"Expire memories error: {e}", exc_info=True)
        return jsonify({"error": "Expire failed"}), 500