# app/storage/memory_repository.py
import sqlite3
import re
import logging
from datetime import datetime
import numpy as np

# This file ONLY handles database (SQL) operations.

logger = logging.getLogger(__name__)

MAX_MEMORY_VALUE_LENGTH = 1024 

# --- Database Connection Class ---
class _DatabaseConnection:
    """Wraps sqlite3.Connection to provide context manager with auto-commit."""
    def __init__(self, conn):
        self.conn = conn
    
    def __enter__(self):
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.conn.close()
        return False

# --- Repository Class ---
class MemoryRepository:
    """
    Manages the SQLite database for WarmthBot's memory.
    This class ONLY handles database (SQL) operations.
    """
    
    def __init__(self, db_name="warmth_memory.db"):
        self.db_name = db_name
        self.initialize_db()
        logger.info(f"Database repository initialized: {self.db_name}")

    def _get_connection(self):
        """Creates a new database connection."""
        conn = sqlite3.connect(self.db_name, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error as e:
            logger.warning(f"Failed to enable WAL mode: {e}")
        return _DatabaseConnection(conn)

    # --- Schema and Migrations ---
    
    def initialize_db(self):
        """Creates tables if they don't exist and runs schema migrations."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER NOT NULL PRIMARY KEY
                );
            """)
            cursor = conn.execute("SELECT version FROM schema_version")
            row = cursor.fetchone()
            if row is None:
                conn.execute("INSERT INTO schema_version (version) VALUES (0)")
                current_version = 0
            else:
                current_version = row['version']

        with self._get_connection() as conn:
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                logger.info("WAL mode enabled for better concurrency")
            except sqlite3.Error as e:
                logger.warning(f"Failed to enable WAL mode: {e}")
        
        if current_version < 1: self._migration_v1(current_version)
        if current_version < 2: self._migration_v2(current_version)
        if current_version < 3: self._migration_v3(current_version)
        if current_version < 4: self._migration_v4(current_version)
        if current_version < 5: self._migration_v5(current_version)
        if current_version < 6: self._migration_v6(current_version)

    def _migration_v1(self, current_version):
        """Migration to create the main 'memories' (facts) table."""
        if current_version >= 1: return
        logger.info("Running migration v1: Creating 'memories' table...")
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    UNIQUE(user_id, key)
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_user_id ON memories (user_id);")
            conn.execute("UPDATE schema_version SET version = 1")
        logger.info("Migration v1 complete.")

    def _migration_v2(self, current_version):
        """Migration to create the 'mood_logs' table."""
        if current_version >= 2: return
        logger.info("Running migration v2: Creating 'mood_logs' table...")
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mood_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    label TEXT NOT NULL,
                    topic TEXT,
                    timestamp TEXT NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mood_user_time ON mood_logs (user_id, timestamp);")
            conn.execute("UPDATE schema_version SET version = 2")
        logger.info("Migration v2 complete.")
    
    def _migration_v3(self, current_version):
        """Migration to add importance and expiry fields to memories table."""
        if current_version >= 3: return
        logger.info("Running migration v3: Adding importance and expiry fields...")
        with self._get_connection() as conn:
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN importance REAL DEFAULT 0.5")
                conn.execute("ALTER TABLE memories ADD COLUMN expiry_days INTEGER DEFAULT NULL")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_importance ON memories (importance);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_timestamp ON memories (timestamp);")
                conn.execute("UPDATE schema_version SET version = 3")
                logger.info("Migration v3 complete.")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    logger.warning(f"Migration v3 warning: {e}")
                # Always set version even if columns exist, to prevent re-running
                conn.execute("UPDATE schema_version SET version = 3")
    
    def _migration_v4(self, current_version):
        """Migration to create user preferences table for listening mode."""
        if current_version >= 4: return
        logger.info("Running migration v4: Creating user_preferences table...")
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        user_id TEXT NOT NULL PRIMARY KEY,
                        listening_mode INTEGER DEFAULT 0,
                        listening_memory_policy INTEGER DEFAULT 0,
                        listening_tts_muted INTEGER DEFAULT 1,
                        tts_enabled INTEGER DEFAULT 0,
                        updated_at TEXT NOT NULL
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_prefs_user_id ON user_preferences (user_id);")
                conn.execute("UPDATE schema_version SET version = 4")
                logger.info("Migration v4 complete.")
            except sqlite3.Error as e:
                logger.warning(f"Migration v4 warning: {e}")
                conn.execute("UPDATE schema_version SET version = 4")
    
    def _migration_v5(self, current_version):
        """Migration to create embedding tables for vector search."""
        if current_version >= 5: return
        logger.info("Running migration v5: Creating embedding tables for vector search...")
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_embeddings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        memory_id INTEGER NOT NULL UNIQUE,
                        embedding BLOB NOT NULL,
                        embedding_model TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2',
                        embedding_dim INTEGER NOT NULL DEFAULT 384,
                        embedded_at TEXT NOT NULL,
                        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_embed_mem_id ON memory_embeddings(memory_id);")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS mood_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        pattern_name TEXT,
                        pattern_embedding BLOB,
                        member_topics TEXT,
                        relevance_score REAL DEFAULT 0.5,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_mood_pattern_user ON mood_patterns(user_id);")
                conn.execute("UPDATE schema_version SET version = 5")
                logger.info("Migration v5 complete.")
            except sqlite3.Error as e:
                logger.warning(f"Migration v5 warning: {e}")
                conn.execute("UPDATE schema_version SET version = 5")
    
    def _migration_v6(self, current_version):
        """Migration to create memory access tracking for importance learning."""
        if current_version >= 6: return
        logger.info("Running migration v6: Creating memory access tracking...")
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_access_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        memory_id INTEGER NOT NULL,
                        user_id TEXT NOT NULL,
                        accessed_at TEXT NOT NULL,
                        access_type TEXT DEFAULT 'retrieve',
                        relevance_score REAL DEFAULT 0.5,
                        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_access_mem_id ON memory_access_log(memory_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_access_user_time ON memory_access_log(user_id, accessed_at);")
                conn.execute("UPDATE schema_version SET version = 6")
                logger.info("Migration v6 complete.")
            except sqlite3.Error as e:
                logger.warning(f"Migration v6 warning: {e}")
                conn.execute("UPDATE schema_version SET version = 6")

    # --- Sanitization Helpers ---

    def _sanitize_memory_value(self, value: str) -> str:
        """Sanitizes memory values before saving to database."""
        if not value: return ""
        sanitized = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', value)
        sanitized = re.sub(r'\s+', ' ', sanitized)
        sanitized = sanitized.strip()
        if len(sanitized) > MAX_MEMORY_VALUE_LENGTH:
            sanitized = sanitized[:MAX_MEMORY_VALUE_LENGTH]
            logger.warning(f"Memory value truncated from {len(value)} to {MAX_MEMORY_VALUE_LENGTH} chars")
        return sanitized
    
    def _sanitize_memory_key(self, key: str) -> str:
        """Sanitizes memory keys before saving to database."""
        if not key: return ""
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', key)
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        max_key_length = 50
        if len(sanitized) > max_key_length:
            sanitized = sanitized[:max_key_length]
        return sanitized

    # --- Fact Memory Functions (Pure SQL) ---

    def add_or_update_fact(self, user_id: str, key: str, value: str, importance: float = 0.5, expiry_days: int = None) -> int:
        """
        Saves or updates a fact. Returns the memory ID.
        NOTE: This no longer generates embeddings. The service layer does that.
        """
        sanitized_key = self._sanitize_memory_key(key)
        sanitized_value = self._sanitize_memory_value(value)
        
        if not sanitized_key or not sanitized_value:
            logger.warning(f"Attempted to save empty memory: key='{key[:20]}...', value='{value[:20]}...'")
            return -1
        
        importance = max(0.0, min(1.0, importance))
        timestamp = datetime.utcnow().isoformat()
        
        memory_id = -1
        try:
            # Try with RETURNING first (modern SQLite)
            sql = """
                INSERT INTO memories (user_id, key, value, timestamp, importance, expiry_days)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, key) DO UPDATE SET
                    value = excluded.value,
                    timestamp = excluded.timestamp,
                    importance = excluded.importance,
                    expiry_days = excluded.expiry_days
                RETURNING id;
            """
            with self._get_connection() as conn:
                cursor = conn.execute(sql, (user_id, sanitized_key, sanitized_value, timestamp, importance, expiry_days))
                row = cursor.fetchone()
                if row:
                    memory_id = row['id']
        
        except sqlite3.Error as e:
            # Fallback for older sqlite versions
            if "syntax error" in str(e).lower():
                sql_no_return = """
                    INSERT INTO memories (user_id, key, value, timestamp, importance, expiry_days)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, key) DO UPDATE SET
                        value = excluded.value,
                        timestamp = excluded.timestamp,
                        importance = excluded.importance,
                        expiry_days = excluded.expiry_days;
                """
                with self._get_connection() as conn:
                    conn.execute(sql_no_return, (user_id, sanitized_key, sanitized_value, timestamp, importance, expiry_days))
                
                # Get the ID separately
                with self._get_connection() as conn:
                    cursor = conn.execute("SELECT id FROM memories WHERE user_id = ? AND key = ?", (user_id, sanitized_key))
                    row = cursor.fetchone()
                    if row:
                        memory_id = row['id']
            else:
                logger.error(f"Database Error: Failed to save fact. {e}", exc_info=True)
        
        return memory_id

    def get_all_memories(self, user_id: str) -> list:
        """Gets a list of all facts for the API."""
        sql = "SELECT id, key, value, timestamp FROM memories WHERE user_id = ? ORDER BY timestamp DESC"
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(sql, (user_id,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Database Error: Failed to get all memories. {e}")
            return []

    def forget_memory(self, user_id: str, mem_id: int) -> int:
        """Deletes a single fact by its ID."""
        sql = "DELETE FROM memories WHERE user_id = ? AND id = ?"
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(sql, (user_id, mem_id))
                return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f"Database Error: Failed to forget memory. {e}")
            return 0

    # --- Mood Log Functions (Pure SQL) ---

    def log_mood(self, user_id: str, score: float, label: str, topic: str):
        """Saves a single mood entry."""
        timestamp = datetime.utcnow().isoformat()
        sql = "INSERT INTO mood_logs (user_id, score, label, topic, timestamp) VALUES (?, ?, ?, ?, ?)"
        try:
            with self._get_connection() as conn:
                conn.execute(sql, (user_id, score, label, topic, timestamp))
        except sqlite3.Error as e:
            logger.error(f"Database Error: Failed to log mood. {e}")

    def get_recent_mood_scores(self, user_id: str, limit: int = 3) -> list:
        """Gets the most recent mood scores (newest to oldest)."""
        sql = "SELECT score FROM mood_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?"
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(sql, (user_id, limit))
                rows = cursor.fetchall()
                return [row['score'] for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Database Error: Failed to get recent mood scores. {e}")
            return []

    def get_mood_history(self, user_id: str, days: int = 7) -> list:
        """Gets the average mood and entry count per day for the last N days."""
        sql = """
            SELECT
                DATE(timestamp) as day,
                AVG(score) as avg_score,
                COUNT(*) as entry_count
            FROM mood_logs
            WHERE user_id = ?
            AND timestamp >= datetime('now', '-' || ? || ' days')
            GROUP BY DATE(timestamp)
            ORDER BY day DESC;
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(sql, (user_id, days))
                rows = cursor.fetchall()
                return [[row['day'], row['avg_score'], row['entry_count']] for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Database Error: Failed to get mood history. {e}")
            return []
    
    # --- Vector Embedding Functions (Pure SQL) ---
    
    def store_embedding(self, memory_id: int, embedding_bytes: bytes, model_name: str, embedding_dim: int) -> bool:
        """
        Stores an embedding.
        NOTE: This no longer GENERATES the embedding, it just saves it.
        """
        if not embedding_bytes:
             logger.warning(f"Attempted to store empty embedding for memory {memory_id}")
             return False
        
        timestamp = datetime.utcnow().isoformat()
        sql = """
            INSERT INTO memory_embeddings 
            (memory_id, embedding, embedding_model, embedding_dim, embedded_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                embedding = excluded.embedding,
                embedding_model = excluded.embedding_model,
                embedding_dim = excluded.embedding_dim,
                embedded_at = excluded.embedded_at;
        """
        try:
            with self._get_connection() as conn:
                conn.execute(sql, (memory_id, embedding_bytes, model_name, embedding_dim, timestamp))
            logger.debug(f"Embedding stored for memory {memory_id}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Database Error: Failed to store embedding. {e}")
            return False

    def get_all_memories_with_embeddings(self, user_id: str) -> list:
        """
        Gets all memories for a user, JOINING their embedding data.
        This is the raw data the service layer will use.
        """
        sql = """
            SELECT m.id, m.key, m.value, m.importance, me.embedding, me.embedding_dim
            FROM memories m
            LEFT JOIN memory_embeddings me ON m.id = me.memory_id
            WHERE m.user_id = ?
            ORDER BY m.timestamp DESC
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(sql, (user_id,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Database Error: Failed to get all memories with embeddings. {e}")
            return []

    # --- Importance Learning Functions (Pure SQL) ---
    
    def log_memory_access(self, memory_id: int, user_id: str, access_type: str = "retrieve",
                         relevance_score: float = 0.5) -> bool:
        """Log a memory access for importance learning."""
        timestamp = datetime.utcnow().isoformat()
        sql = """
            INSERT INTO memory_access_log (memory_id, user_id, accessed_at, access_type, relevance_score)
            VALUES (?, ?, ?, ?, ?)
        """
        try:
            with self._get_connection() as conn:
                conn.execute(sql, (memory_id, user_id, timestamp, access_type, relevance_score))
            logger.debug(f"Logged access: memory {memory_id}, type {access_type}, score {relevance_score}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to log memory access: {e}")
            return False
    
    def update_importance_from_access(self, user_id: str, days: int = 30) -> int:
        """Update memory importance scores based on recent access patterns."""
        try:
            sql = """
                SELECT 
                    m.id,
                    COUNT(*) as access_count,
                    AVG(mal.relevance_score) as avg_relevance,
                    MAX(mal.accessed_at) as last_accessed
                FROM memory_access_log mal
                JOIN memories m ON mal.memory_id = m.id
                WHERE mal.user_id = ?
                AND mal.accessed_at >= datetime('now', '-' || ? || ' days')
                GROUP BY m.id
            """
            with self._get_connection() as conn:
                cursor = conn.execute(sql, (user_id, days))
                rows = cursor.fetchall()
            
            if not rows:
                logger.debug(f"No access history for user {user_id}")
                return 0
            
            updated_count = 0
            
            for row in rows:
                memory_id = row['id']
                access_count = row['access_count']
                avg_relevance = row['avg_relevance'] or 0.5
                last_accessed = row['last_accessed']
                
                # Logic from original memory_manager.py
                frequency_importance = min(0.9, 0.1 + (access_count * 0.1))
                now = datetime.utcnow()
                accessed_time = datetime.fromisoformat(last_accessed)
                days_ago = (now - accessed_time).days
                recency_bonus = max(0, 0.1 * (1 - days_ago / 30))
                
                new_importance = min(
                    0.95,
                    (0.7 * frequency_importance) + (0.3 * avg_relevance) + recency_bonus
                )
                
                update_sql = "UPDATE memories SET importance = ? WHERE id = ? AND user_id = ?"
                try:
                    with self._get_connection() as conn:
                        conn.execute(update_sql, (new_importance, memory_id, user_id))
                    updated_count += 1
                except sqlite3.Error as e:
                    logger.warning(f"Failed to update importance for memory {memory_id}: {e}")
            
            return updated_count
        
        except Exception as e:
            logger.error(f"Error updating importance from access: {e}")
            return 0
    
    def get_memory_importance_stats(self, user_id: str) -> dict:
        """Get statistics about memory importance distribution."""
        sql = """
            SELECT
                COUNT(*) as total_memories,
                AVG(importance) as avg_importance,
                MIN(importance) as min_importance,
                MAX(importance) as max_importance,
                SUM(CASE WHEN importance >= 0.7 THEN 1 ELSE 0 END) as high_importance_count,
                SUM(CASE WHEN importance < 0.3 THEN 1 ELSE 0 END) as low_importance_count
            FROM memories
            WHERE user_id = ?
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(sql, (user_id,))
                row = cursor.fetchone()
            return {
                'total_memories': row['total_memories'] or 0,
                'avg_importance': round(row['avg_importance'] or 0.5, 3),
                'min_importance': round(row['min_importance'] or 0.0, 3),
                'max_importance': round(row['max_importance'] or 1.0, 3),
                'high_importance': row['high_importance_count'] or 0,
                'low_importance': row['low_importance_count'] or 0
            }
        except sqlite3.Error as e:
            logger.error(f"Failed to get importance stats: {e}")
            return {}
    
    # --- Maintenance Functions ---
    
    def vacuum(self):
        """Runs VACUUM to optimize the database."""
        try:
            logger.info("Starting database VACUUM...")
            with self._get_connection() as conn:
                conn.execute("VACUUM")
            logger.info("Database VACUUM complete.")
            return True
        except sqlite3.Error as e:
            logger.error(f"Database Error: VACUUM failed. {e}")
            return False
    
    def analyze(self):
        """Updates database statistics for query optimizer."""
        try:
            with self._get_connection() as conn:
                conn.execute("ANALYZE")
            logger.info("Database statistics updated.")
            return True
        except sqlite3.Error as e:
            logger.error(f"Database Error: ANALYZE failed. {e}")
            return False
    
    def expire_old_memories(self, user_id: str = None):
        """Deletes memories that have expired."""
        try:
            if user_id:
                sql = """
                    DELETE FROM memories 
                    WHERE user_id = ? 
                    AND expiry_days IS NOT NULL
                    AND date(timestamp, '+' || expiry_days || ' days') < date('now')
                """
                params = (user_id,)
            else:
                sql = """
                    DELETE FROM memories 
                    WHERE expiry_days IS NOT NULL
                    AND date(timestamp, '+' || expiry_days || ' days') < date('now')
                """
                params = ()
            
            with self._get_connection() as conn:
                cursor = conn.execute(sql, params)
                deleted_count = cursor.rowcount
            
            if deleted_count > 0:
                logger.info(f"Expired {deleted_count} memory(ies)")
            
            return deleted_count
        except sqlite3.Error as e:
            logger.error(f"Database Error: Failed to expire memories. {e}")
            return 0
    
    # --- Data Export/Import Functions ---
    
    def export_all_data(self, user_id: str) -> dict:
        """Exports all data for a user."""
        try:
            export_data = {
                "user_id": user_id,
                "export_timestamp": datetime.utcnow().isoformat(),
                "memories": [],
                "mood_logs": []
            }
            export_data["memories"] = self.get_all_memories(user_id)
            sql = """
                SELECT id, score, label, topic, timestamp 
                FROM mood_logs 
                WHERE user_id = ? 
                ORDER BY timestamp DESC
            """
            with self._get_connection() as conn:
                cursor = conn.execute(sql, (user_id,))
                rows = cursor.fetchall()
                export_data["mood_logs"] = [dict(row) for row in rows]
            
            logger.info(f"Exported {len(export_data['memories'])} memories and {len(export_data['mood_logs'])} mood logs for user {user_id}")
            return export_data
        except sqlite3.Error as e:
            logger.error(f"Database Error: Failed to export data. {e}")
            return {}
    
    def erase_all_data(self, user_id: str) -> dict:
        """Erases all data for a user."""
        try:
            deleted_counts = {"memories": 0, "mood_logs": 0}
            
            sql_memories = "DELETE FROM memories WHERE user_id = ?"
            with self._get_connection() as conn:
                cursor = conn.execute(sql_memories, (user_id,))
                deleted_counts["memories"] = cursor.rowcount
            
            sql_moods = "DELETE FROM mood_logs WHERE user_id = ?"
            with self._get_connection() as conn:
                cursor = conn.execute(sql_moods, (user_id,))
                deleted_counts["mood_logs"] = cursor.rowcount
            
            logger.warning(f"Erased all data for user {user_id}: {deleted_counts['memories']} memories, {deleted_counts['mood_logs']} mood logs")
            return deleted_counts
        except sqlite3.Error as e:
            logger.error(f"Database Error: Failed to erase data. {e}")
            return {"memories": 0, "mood_logs": 0}
    
    # --- User Preferences Functions ---
    
    def get_user_preferences(self, user_id: str) -> dict:
        """Gets user preferences."""
        # FIX: Add tts_enabled to the SELECT
        sql = "SELECT listening_mode, listening_memory_policy, listening_tts_muted, tts_enabled FROM user_preferences WHERE user_id = ?"
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(sql, (user_id,))
                row = cursor.fetchone()
                if row:
                    return {
                        "listening_mode": bool(row['listening_mode']),
                        "listening_memory_policy": bool(row['listening_memory_policy']),
                        "listening_tts_muted": bool(row['listening_tts_muted']),
                        "tts_enabled": bool(row['tts_enabled'] if 'tts_enabled' in row.keys() else 0) # Safe access for old DBs
                    }
                else:
                    # Return defaults
                    return {
                        "listening_mode": False,
                        "listening_memory_policy": False,
                        "listening_tts_muted": True,
                        "tts_enabled": False # Default to OFF
                    }
        except sqlite3.Error as e:
            logger.error(f"Database Error: Failed to get user preferences. {e}")
            return {
                "listening_mode": False,
                "listening_memory_policy": False,
                "listening_tts_muted": True,
                "tts_enabled": False # Default to OFF
            }
    
    def set_user_preferences(self, user_id: str, listening_mode: bool = None, 
                            listening_memory_policy: bool = None, 
                            listening_tts_muted: bool = None,
                            tts_enabled: bool = None) -> bool:
        """Updates user preferences."""
        try:
            current = self.get_user_preferences(user_id)
            
            if listening_mode is not None: current["listening_mode"] = listening_mode
            if listening_memory_policy is not None: current["listening_memory_policy"] = listening_memory_policy
            if listening_tts_muted is not None: current["listening_tts_muted"] = listening_tts_muted
            if tts_enabled is not None: current["tts_enabled"] = tts_enabled

            timestamp = datetime.utcnow().isoformat()
            
            sql = """
                INSERT INTO user_preferences (user_id, listening_mode, listening_memory_policy, listening_tts_muted, tts_enabled, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    listening_mode = excluded.listening_mode,
                    listening_memory_policy = excluded.listening_memory_policy,
                    listening_tts_muted = excluded.listening_tts_muted,
                    tts_enabled = excluded.tts_enabled,
                    updated_at = excluded.updated_at;
            """
            with self._get_connection() as conn:
                conn.execute(sql, (
                    user_id,
                    1 if current["listening_mode"] else 0,
                    1 if current["listening_memory_policy"] else 0,
                    1 if current["listening_tts_muted"] else 0,
                    1 if current["tts_enabled"] else 0,
                    timestamp
                ))
            return True
        except sqlite3.Error as e:
            logger.error(f"Database Error: Failed to set user preferences. {e}")
            return False