# app/web/mood.py
import logging
import json
from flask import Blueprint, jsonify, request, Response, current_app
from datetime import datetime

from ..security import csrf_protect, require_auth, encrypt_data, decrypt_data
from ..config import DEFAULT_USER_ID
from .validation import validate_json_request

# === THE FIX: Remove the url_prefix here ===
bp = Blueprint('mood', __name__)
logger = logging.getLogger(__name__)

# === THE FIX: Add the full path directly to the route ===
@bp.route('/mood-history', methods=['GET'])
@require_auth
def get_mood_history():
    """ GET /mood-history - Retrieves mood history and advice. """
    try:
        history = current_app.memory_repo.get_mood_history(DEFAULT_USER_ID)
        advice = current_app.chat_service.get_smart_advice(history)
        return jsonify({
            "history": history,
            "advice": advice
        }), 200
    except Exception as e:
        logger.error(f"GET /mood-history - 500 Internal Server Error: {e}", exc_info=True)
        raise

@bp.route('/api/journal', methods=['GET'])
@require_auth
def get_journal():
    """ GET /api/journal - Retrieves combined memories and mood logs sorted by timestamp. """
    try:
        # Get memories and mood logs from the repository
        memories = current_app.memory_repo.get_all_memories(DEFAULT_USER_ID)
        mood_history = current_app.memory_repo.get_mood_history(DEFAULT_USER_ID)

        # Create combined journal entries
        journal_entries = []

        # Process memories
        for memory in memories:
            journal_entries.append({
                'id': memory['id'],
                'type': 'memory',
                'key': memory['key'],
                'value': memory['value'],
                'timestamp': memory['timestamp'],
                'importance': memory.get('importance', 0.5)
            })

        # Process mood logs
        for mood in mood_history:
            if isinstance(mood, list) and len(mood) >= 2:
                # mood format: [timestamp, score, count, label, topic]
                timestamp = mood[0]
                score = mood[1]
                count = mood[2] if len(mood) > 2 else 1
                label = mood[3] if len(mood) > 3 else 'Neutral'
                topic = mood[4] if len(mood) > 4 else None

                journal_entries.append({
                    'id': f"mood_{timestamp}",  # Create unique ID for mood entries
                    'type': 'mood',
                    'score': score,
                    'label': label,
                    'topic': topic,
                    'timestamp': timestamp,
                    'count': count
                })

        # Sort all entries by timestamp (descending - newest first)
        journal_entries.sort(key=lambda x: x['timestamp'], reverse=True)

        return jsonify({
            'entries': journal_entries
        }), 200

    except Exception as e:
        logger.error(f"GET /api/journal - 500 Internal Server Error: {e}", exc_info=True)
        raise

# === THE FIX: Add the full path directly to the route ===
@bp.route('/export/mood-history', methods=['GET'])
@require_auth
def export_mood_history():
    """ GET /export/mood-history - Exports mood history as encrypted JSON. """
    try:
        password = request.args.get('password', None)
        
        def _export_task():
            history = current_app.memory_repo.get_mood_history(DEFAULT_USER_ID)
            memories = current_app.memory_repo.get_all_memories(DEFAULT_USER_ID)
            export_data = {
                "mood_history": history,
                "memories": memories,
                "export_timestamp": datetime.utcnow().isoformat()
            }
            json_data = json.dumps(export_data, indent=2)
            encrypted_data = encrypt_data(json_data, password)
            return encrypted_data
        
        # Use the executor from the chat_service (which gets it from llm_service)
        future = current_app.chat_service.llm_service.heavy_executor.submit(_export_task)
        encrypted_data = future.result(timeout=30)
        
        response = Response(
            encrypted_data,
            mimetype='application/octet-stream',
            headers={
                'Content-Disposition': 'attachment; filename=warmth_export_encrypted.txt',
                'X-Export-Format': 'encrypted',
                'X-Password-Required': 'true' if password else 'false'
            }
        )
        logger.info("Mood history exported (encrypted)")
        return response, 200
        
    except Exception as e:
        logger.error(f"Export error: {e}", exc_info=True)
        return jsonify({"error": "Export failed"}), 500

# === THE FIX: Add the full path directly to the route ===
@bp.route('/export/mood-history/decrypt', methods=['POST'])
@csrf_protect
@require_auth
def decrypt_export():
    """ POST /export/mood-history/decrypt - Decrypts exported data. """
    try:
        data, error_response, status_code = validate_json_request()
        if error_response: return error_response, status_code
        
        encrypted_data = data.get('encrypted_data')
        password = data.get('password')
        
        if not encrypted_data or not password:
            return jsonify({"error": "Missing encrypted_data or password"}), 400
        
        decrypted_data = decrypt_data(encrypted_data, password)
        
        return jsonify({"status": "success", "data": decrypted_data}), 200
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Decrypt error: {e}", exc_info=True)
        return jsonify({"error": "Decryption failed"}), 500