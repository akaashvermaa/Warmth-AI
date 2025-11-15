"""
Security utilities for Warmth application.
Handles authentication, encryption, secure logging, and CSRF protection.
"""
import os
import hashlib
import secrets
import base64
import logging
from functools import wraps
from flask import session, request, jsonify, abort
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

# === Password Management ===

def hash_password(password: str) -> str:
    """Hashes a password using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{password_hash}"

def verify_password(password: str, stored_hash: str) -> bool:
    """Verifies a password against a stored hash."""
    try:
        salt, password_hash = stored_hash.split(':')
        computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return secrets.compare_digest(computed_hash, password_hash)
    except (ValueError, AttributeError):
        return False

def get_or_create_password_hash() -> str:
    """
    Gets the password hash from environment variable or creates a default one.
    For first-time setup, use WARMTH_PASSWORD env var or create a default.
    """
    password_file = os.getenv('WARMTH_PASSWORD_FILE', '.warmth_password')
    
    # Check environment variable first
    env_password = os.getenv('WARMTH_PASSWORD')
    if env_password:
        return hash_password(env_password)
    
    # Check file
    if os.path.exists(password_file):
        try:
            with open(password_file, 'r') as f:
                stored_hash = f.read().strip()
                if stored_hash and ':' in stored_hash:
                    return stored_hash
        except Exception as e:
            logger.warning(f"Error reading password file: {e}")
    
    # Create default password (user should change this)
    default_password = os.getenv('WARMTH_DEFAULT_PASSWORD', 'warmth')
    default_hash = hash_password(default_password)
    
    try:
        with open(password_file, 'w') as f:
            f.write(default_hash)
        logger.warning(f"Created default password file. Please set WARMTH_PASSWORD env var or change password in {password_file}")
    except Exception as e:
        logger.error(f"Could not write password file: {e}")
    
    return default_hash

# === Encryption for Exported Data ===

def generate_encryption_key(password: str = None) -> bytes:
    """Generates an encryption key from a password or creates a new one."""
    if password:
        # Derive key from password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'warmth_export_salt',  # In production, use random salt per export
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    else:
        # Generate random key
        key = Fernet.generate_key()
    return key

def encrypt_data(data: str, password: str = None) -> str:
    """
    Encrypts data using Fernet symmetric encryption.
    If password is provided, derives key from it. Otherwise uses random key.
    """
    key = generate_encryption_key(password)
    fernet = Fernet(key)
    encrypted = fernet.encrypt(data.encode())
    return base64.urlsafe_b64encode(encrypted).decode()

def decrypt_data(encrypted_data: str, password: str = None, key: bytes = None) -> str:
    """
    Decrypts data. Requires either password or key.
    """
    if key:
        fernet = Fernet(key)
    elif password:
        key = generate_encryption_key(password)
        fernet = Fernet(key)
    else:
        raise ValueError("Either password or key must be provided")
    
    try:
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = fernet.decrypt(encrypted_bytes)
        return decrypted.decode()
    except Exception as e:
        logger.error(f"Decryption error: {e}")
        raise ValueError("Failed to decrypt data. Invalid password or corrupted data.")

# === Secure Logging ===

def hash_message(message: str, max_length: int = 50) -> str:
    """
    Creates a secure hash of a message for logging.
    Returns truncated message + hash for debugging while protecting privacy.
    """
    if not message:
        return "empty"
    
    # Truncate message
    truncated = message[:max_length] if len(message) > max_length else message
    
    # Create hash of full message
    message_hash = hashlib.sha256(message.encode()).hexdigest()[:16]  # First 16 chars of hash
    
    if len(message) > max_length:
        return f"{truncated}...[hash:{message_hash}]"
    else:
        return f"{truncated}[hash:{message_hash}]"

def secure_log_message(message: str, log_level: str = "info") -> None:
    """
    Logs a message securely (truncated + hashed).
    """
    secure_msg = hash_message(message)
    if log_level == "warning":
        logger.warning(f"Message: {secure_msg}")
    elif log_level == "error":
        logger.error(f"Message: {secure_msg}")
    else:
        logger.info(f"Message: {secure_msg}")

# === CSRF Protection ===

def generate_csrf_token() -> str:
    """Generates a CSRF token."""
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(32)
    return session['csrf_token']

def validate_csrf_token(token: str) -> bool:
    """Validates a CSRF token."""
    if 'csrf_token' not in session:
        return False
    return secrets.compare_digest(session['csrf_token'], token)

def csrf_protect(f):
    """
    Decorator to protect routes with CSRF tokens.
    Only applies if ENABLE_CSRF is True (for LAN access).
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        enable_csrf = os.getenv('ENABLE_CSRF', 'false').lower() == 'true'
        
        if not enable_csrf:
            # CSRF protection disabled (localhost only)
            return f(*args, **kwargs)
        
        # CSRF protection enabled (LAN access)
        if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:
            # Get token from header or form data
            token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token') or request.json.get('csrf_token') if request.is_json else None
            
            if not token or not validate_csrf_token(token):
                logger.warning(f"CSRF token validation failed for {request.path}")
                abort(403)
        
        return f(*args, **kwargs)
    return decorated_function

# === Authentication Decorator ===

def require_auth(f):
    """
    Decorator to require authentication for routes.
    Only applies if ENABLE_AUTH is True.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        enable_auth = os.getenv('ENABLE_AUTH', 'false').lower() == 'true'
        
        if not enable_auth:
            # Authentication disabled
            return f(*args, **kwargs)
        
        # Check if user is authenticated
        if not session.get('authenticated', False):
            return jsonify({"error": "Authentication required"}), 401
        
        return f(*args, **kwargs)
    return decorated_function

