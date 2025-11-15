# app/services/whisper_utils.py
# --- WHISPER IS DISABLED ---

import logging
logger = logging.getLogger(__name__)

logger.warning("Whisper (voice transcription) is currently disabled.")
model = None

def transcribe_audio(file_bytes_or_path):
    """
    Transcription is disabled.
    """
    logger.warning("transcribe_audio called, but Whisper is disabled.")
    raise RuntimeError("Transcription model not available")