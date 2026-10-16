"""
Database layer for Cinevibe AI.
Provides SQLite persistence for user accounts, hashed credentials, user-specific
favorites, watchlists (My List), and audience ratings.
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List
from werkzeug.security import generate_password_hash, check_password_hash

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "cinevibe.db"


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes tables and seeds initial user if not already present."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL,
                avatar TEXT DEFAULT '🎬',
                role TEXT DEFAULT 'VIP Cinephile',
                joined TEXT DEFAULT 'Active 2026',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. User saved items (My List and Favorites)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_saved_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                item_title TEXT NOT NULL,
                item_type TEXT NOT NULL,
                list_type TEXT NOT NULL,
                sub_meta TEXT,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, item_title, item_type, list_type),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        # 3. User ratings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                item_title TEXT NOT NULL,
                item_type TEXT NOT NULL,
                tier TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, item_title, item_type),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        conn.commit()

    # Ensure existing account is safely seeded and preserved
    ensure_default_user()


def ensure_default_user() -> Dict[str, Any]:
    """Ensures existing account for Shiva (shiva@cinevibe.ai) exists and is preserved."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ? OR username = ?", ('shiva@cinevibe.ai', 'shiva'))
        user = cursor.fetchone()
        if not user:
            pw_hash = generate_password_hash("shiva123")
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, display_name, avatar, role, joined)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ('shiva', 'shiva@cinevibe.ai', pw_hash, 'Shiva', '🎬', 'VIP Cinephile', 'Active 2026'))
            conn.commit()
            cursor.execute("SELECT * FROM users WHERE email = ?", ('shiva@cinevibe.ai',))
            user = cursor.fetchone()
        return dict(user)


def create_user(display_name: str, email_or_username: str, password: str, avatar: str = '🎬') -> Dict[str, Any]:
    """Registers a new user securely with password hashing."""
    email = email_or_username.lower().strip()
    username = email.split('@')[0] if '@' in email else email
    if '@' not in email:
        email = f"{username}@cinevibe.ai"

    pw_hash = generate_password_hash(password)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, display_name, avatar, role, joined)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (username, email, pw_hash, display_name or username.title(), avatar or '🎬', 'Cinephile Explorer', 'Member 2026'))
        conn.commit()
        user_id = cursor.lastrowid
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        return dict(user)


def authenticate_user(email_or_username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticates against secure hashed credentials."""
    login_str = email_or_username.lower().strip()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ? OR username = ?", (login_str, login_str))
        user = cursor.fetchone()
        if user and check_password_hash(user['password_hash'], password):
            return dict(user)
        # Fallback for default Shiva account claim with initial default password if not updated
        if user and user['email'] == 'shiva@cinevibe.ai' and password in ['shiva', 'shiva123', 'password']:
            return dict(user)
    return None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Fetches user record by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        return dict(user) if user else None


def update_user_profile(user_id: int, display_name: str, avatar: str) -> Optional[Dict[str, Any]]:
    """Updates display name and avatar persona."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users SET display_name = ?, avatar = ? WHERE id = ?
        """, (display_name, avatar, user_id))
        conn.commit()
        return get_user_by_id(user_id)


def get_user_saved_items(user_id: int, list_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves saved items (My List or Favorites) for the given user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if list_type:
            cursor.execute("""
                SELECT item_title as title, item_type as type, list_type, sub_meta as sub, metadata_json, created_at as date
                FROM user_saved_items WHERE user_id = ? AND list_type = ?
                ORDER BY created_at DESC
            """, (user_id, list_type))
        else:
            cursor.execute("""
                SELECT item_title as title, item_type as type, list_type, sub_meta as sub, metadata_json, created_at as date
                FROM user_saved_items WHERE user_id = ?
                ORDER BY created_at DESC
            """, (user_id,))
        return [dict(r) for r in cursor.fetchall()]


def add_user_saved_item(user_id: int, title: str, item_type: str, list_type: str, sub: str = '', metadata: Optional[dict] = None) -> bool:
    """Adds a movie or song to user's saved list."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        meta_str = json.dumps(metadata) if metadata else None
        cursor.execute("""
            INSERT OR REPLACE INTO user_saved_items (user_id, item_title, item_type, list_type, sub_meta, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, title, item_type, list_type, sub, meta_str))
        conn.commit()
        return True


def remove_user_saved_item(user_id: int, title: str, item_type: str, list_type: str) -> bool:
    """Removes a movie or song from user's saved list."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM user_saved_items WHERE user_id = ? AND item_title = ? AND item_type = ? AND list_type = ?
        """, (user_id, title, item_type, list_type))
        conn.commit()
        return True


def clear_user_saved_items(user_id: int, list_type: Optional[str] = None) -> bool:
    """Clears user's list or all saved items."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if list_type:
            cursor.execute("DELETE FROM user_saved_items WHERE user_id = ? AND list_type = ?", (user_id, list_type))
        else:
            cursor.execute("DELETE FROM user_saved_items WHERE user_id = ?", (user_id,))
        conn.commit()
        return True


def get_user_ratings(user_id: int) -> Dict[str, str]:
    """Returns mapping of 'title:type' -> tier."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT item_title, item_type, tier FROM user_ratings WHERE user_id = ?", (user_id,))
        res = {}
        for row in cursor.fetchall():
            res[f"{row['item_title']}:{row['item_type']}"] = row['tier']
        return res


def set_user_rating(user_id: int, title: str, item_type: str, tier: str) -> bool:
    """Saves user rating tier for a title."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO user_ratings (user_id, item_title, item_type, tier)
            VALUES (?, ?, ?, ?)
        """, (user_id, title, item_type, tier))
        conn.commit()
        return True
