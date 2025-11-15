import os

# Model Configuration
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')

# Database Configuration
DB_NAME = os.getenv('DB_NAME', 'warmth_memory.db')

# Chat Configuration
CHAT_HISTORY_LENGTH = int(os.getenv('CHAT_HISTORY_LENGTH', '10'))
MAX_HISTORY_TOKENS = int(os.getenv('MAX_HISTORY_TOKENS', '2000'))  # Approximate token limit for history
OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', '30'))  # Timeout in seconds for ollama.chat()
SLOW_RESPONSE_THRESHOLD = float(os.getenv('SLOW_RESPONSE_THRESHOLD', '5.0'))  # Log slow responses above this (seconds)
AUTO_MEMORIZE_COOLDOWN = int(os.getenv('AUTO_MEMORIZE_COOLDOWN', '10'))  # Seconds between auto-memorize attempts

# User Configuration
DEFAULT_USER_ID = os.getenv('DEFAULT_USER_ID', 'local_user')

# Security Configuration
ENABLE_AUTH = os.getenv('ENABLE_AUTH', 'false').lower() == 'true'  # Enable password authentication
ENABLE_CSRF = os.getenv('ENABLE_CSRF', 'false').lower() == 'true'  # Enable CSRF protection (for LAN access)
FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY', None)  # Secret key for sessions (auto-generated if not set)
WARMTH_PASSWORD = os.getenv('WARMTH_PASSWORD', None)  # Password for authentication
WARMTH_PASSWORD_FILE = os.getenv('WARMTH_PASSWORD_FILE', '.warmth_password')  # File to store password hash
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*')  # CORS allowed origins (comma-separated)

# Performance Configuration
CACHE_MAX_AGE = int(os.getenv('CACHE_MAX_AGE', '31536000'))  # Cache max age in seconds (default: 1 year)
STATIC_CDN_URL = os.getenv('STATIC_CDN_URL', None)  # CDN URL for static assets (optional)
HEAVY_TASK_WORKERS = int(os.getenv('HEAVY_TASK_WORKERS', '4'))  # Number of workers for heavy tasks

# Memory Configuration
MEMORY_EXPIRY_ENABLED = os.getenv('MEMORY_EXPIRY_ENABLED', 'false').lower() == 'true'  # Enable automatic memory expiry
MEMORY_EXPIRY_CHECK_INTERVAL = int(os.getenv('MEMORY_EXPIRY_CHECK_INTERVAL', '86400'))  # Check for expired memories every N seconds (default: daily)