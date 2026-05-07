"""
utils/db.py
SQLite database utilities for conversation persistence.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import os


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "conversations.db")


def init_db():
    """Initialize the database schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tools_used TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversation_id ON messages(conversation_id)
    """)
    
    conn.commit()
    conn.close()


def create_conversation(title: str) -> int:
    """Create a new conversation and return its ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO conversations (title, created_at, updated_at)
        VALUES (?, ?, ?)
    """, (title, datetime.now(), datetime.now()))
    
    conversation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return conversation_id


def save_message(conversation_id: int, role: str, content: str, tools_used: List[str] = None):
    """Save a message to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    tools_json = json.dumps(tools_used) if tools_used else None
    
    cursor.execute("""
        INSERT INTO messages (conversation_id, role, content, tools_used, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (conversation_id, role, content, tools_json, datetime.now()))
    
    # Update the conversation's updated_at timestamp
    cursor.execute("""
        UPDATE conversations SET updated_at = ? WHERE id = ?
    """, (datetime.now(), conversation_id))
    
    conn.commit()
    conn.close()


def get_conversations() -> List[Dict]:
    """Get all conversations with their metadata."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, title, created_at, updated_at
        FROM conversations
        ORDER BY updated_at DESC
    """)
    
    conversations = []
    for row in cursor.fetchall():
        conversations.append({
            "id": row[0],
            "title": row[1],
            "created_at": row[2],
            "updated_at": row[3]
        })
    
    conn.close()
    return conversations


def get_conversation_messages(conversation_id: int) -> List[Dict]:
    """Get all messages for a specific conversation."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT role, content, tools_used, timestamp
        FROM messages
        WHERE conversation_id = ?
        ORDER BY timestamp ASC
    """, (conversation_id,))
    
    messages = []
    for row in cursor.fetchall():
        tools_used = json.loads(row[2]) if row[2] else []
        messages.append({
            "role": row[0],
            "content": row[1],
            "tools_used": tools_used,
            "timestamp": row[3]
        })
    
    conn.close()
    return messages


def delete_conversation(conversation_id: int):
    """Delete a conversation and all its messages."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        DELETE FROM messages WHERE conversation_id = ?
    """, (conversation_id,))
    
    cursor.execute("""
        DELETE FROM conversations WHERE id = ?
    """, (conversation_id,))
    
    conn.commit()
    conn.close()


def update_conversation_title(conversation_id: int, title: str):
    """Update the title of a conversation."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?
    """, (title, datetime.now(), conversation_id))
    
    conn.commit()
    conn.close()


def create_conversation_with_title(title: str) -> int:
    """Create a new conversation with a custom title and return its ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO conversations (title, created_at, updated_at)
        VALUES (?, ?, ?)
    """, (title, datetime.now(), datetime.now()))
    
    conversation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return conversation_id


def get_latest_conversation() -> Optional[Dict]:
    """Get the most recently updated conversation."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, title, created_at, updated_at
        FROM conversations
        ORDER BY updated_at DESC
        LIMIT 1
    """)
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "id": row[0],
            "title": row[1],
            "created_at": row[2],
            "updated_at": row[3]
        }
    return None
