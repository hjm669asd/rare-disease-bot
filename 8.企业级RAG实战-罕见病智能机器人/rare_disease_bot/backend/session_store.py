"""会话历史存储 - 基于SQLite持久化"""

import sqlite3
from pathlib import Path


class SessionStore:
    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds
        self.db_path = Path(__file__).parent.parent / "storage" / "sessions.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id, id)
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def get_messages(self, session_id: str) -> list[dict[str, str]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

    def get_recent_history_text(self, session_id: str) -> str:
        messages = self.get_messages(session_id)
        recent_messages = messages[-self.max_rounds * 2:]
        if not recent_messages:
            return "无历史对话。"

        lines = []
        for message in recent_messages:
            role = "用户" if message["role"] == "user" else "助手"
            lines.append(f"{role}: {message['content']}")
        return "\n".join(lines)

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, "user", question),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, "assistant", answer),
            )

            # 只保留最近 max_rounds * 2 条
            limit = self.max_rounds * 2
            conn.execute("""
                DELETE FROM messages WHERE session_id = ? AND id NOT IN (
                    SELECT id FROM messages WHERE session_id = ?
                    ORDER BY id DESC LIMIT ?
                )
            """, (session_id, session_id, limit))

    def clear(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
