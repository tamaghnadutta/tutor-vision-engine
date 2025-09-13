"""
Persistence utilities for request/response storage
"""

import json
import sqlite3
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

import logging
import aiofiles

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class RequestResponseStore:
    """Store for request/response auditing"""

    def __init__(self):
        self.settings = get_settings()
        self.db_path = self._get_db_path()
        self._init_database()

    def _get_db_path(self) -> str:
        """Get database path from settings"""
        if self.settings.database_url.startswith("sqlite:///"):
            return self.settings.database_url[10:]  # Remove sqlite:///
        return "./data/requests.db"

    def _init_database(self):
        """Initialize SQLite database"""
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Create tables
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    request_data TEXT NOT NULL,
                    response_data TEXT NOT NULL,
                    duration REAL NOT NULL,
                    error TEXT,
                    user_id_hash TEXT,
                    session_id TEXT,
                    question_id TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_job_id ON requests(job_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON requests(timestamp)
            """)

    async def save_request_response(self,
                                  job_id: str,
                                  request_data: Dict[str, Any],
                                  response_data: Dict[str, Any],
                                  duration: float,
                                  error: Optional[str] = None) -> None:
        """Save request/response pair asynchronously"""
        try:
            # Run database operation in thread pool
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._save_sync,
                job_id,
                request_data,
                response_data,
                duration,
                error
            )
        except Exception as e:
            logger.error(f"Failed to save request/response: {e}")

    def _save_sync(self,
                   job_id: str,
                   request_data: Dict[str, Any],
                   response_data: Dict[str, Any],
                   duration: float,
                   error: Optional[str] = None):
        """Synchronous save operation"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO requests
                (job_id, timestamp, request_data, response_data, duration, error, user_id_hash, session_id, question_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id,
                datetime.utcnow().isoformat(),
                json.dumps(request_data),
                json.dumps(response_data),
                duration,
                error,
                self._hash_user_id(request_data.get('user_id')),
                request_data.get('session_id'),
                request_data.get('question_id')
            ))

    def _hash_user_id(self, user_id: Optional[str]) -> Optional[str]:
        """Hash user ID for privacy"""
        if not user_id:
            return None

        import hashlib
        return hashlib.sha256(user_id.encode()).hexdigest()[:8]

    async def get_request(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get request by job ID"""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._get_request_sync,
                job_id
            )
            return result
        except Exception as e:
            logger.error(f"Failed to get request {job_id}: {e}")
            return None

    def _get_request_sync(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous get operation"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM requests WHERE job_id = ?
            """, (job_id,))

            row = cursor.fetchone()
            if row:
                return {
                    'job_id': row['job_id'],
                    'timestamp': row['timestamp'],
                    'request_data': json.loads(row['request_data']),
                    'response_data': json.loads(row['response_data']),
                    'duration': row['duration'],
                    'error': row['error'],
                    'user_id_hash': row['user_id_hash'],
                    'session_id': row['session_id'],
                    'question_id': row['question_id']
                }
            return None


# Global store instance
_store = None


def get_store() -> RequestResponseStore:
    """Get global store instance"""
    global _store
    if _store is None:
        _store = RequestResponseStore()
    return _store


async def save_request_response(job_id: str,
                              request_data: Dict[str, Any],
                              response_data: Dict[str, Any],
                              duration: float,
                              error: Optional[str] = None) -> None:
    """Convenience function to save request/response"""
    store = get_store()
    await store.save_request_response(job_id, request_data, response_data, duration, error)


class JSONFileStore:
    """Alternative JSON file-based storage for simple deployments"""

    def __init__(self, storage_dir: str = "./data/requests"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def save_request_response(self,
                                  job_id: str,
                                  request_data: Dict[str, Any],
                                  response_data: Dict[str, Any],
                                  duration: float,
                                  error: Optional[str] = None) -> None:
        """Save to JSON file"""
        try:
            data = {
                'job_id': job_id,
                'timestamp': datetime.utcnow().isoformat(),
                'request_data': request_data,
                'response_data': response_data,
                'duration': duration,
                'error': error
            }

            filename = f"{job_id}.json"
            filepath = self.storage_dir / filename

            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(data, indent=2))

        except Exception as e:
            logger.error(f"Failed to save to JSON file: {e}")

    async def get_request(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get request from JSON file"""
        try:
            filename = f"{job_id}.json"
            filepath = self.storage_dir / filename

            if not filepath.exists():
                return None

            async with aiofiles.open(filepath, 'r') as f:
                content = await f.read()
                return json.loads(content)

        except Exception as e:
            logger.error(f"Failed to load from JSON file: {e}")
            return None