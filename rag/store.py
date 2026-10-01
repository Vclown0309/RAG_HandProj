"""SQLite 存储：chunks 表（id / vector BLOB / content / source）。

向量以 float32 数组序列化为 BLOB 存储（每条 4096 维 = 16 KB），
检索时取出再还原为 list[float]。
"""

import sqlite3
from array import array

DB_PATH = 'kb.db'


def init_db(path: str = DB_PATH) -> None:
    """建表。换模型（维度变化）时先删表重建。"""
    with sqlite3.connect(path) as conn:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS chunks ('
            'id INTEGER PRIMARY KEY AUTOINCREMENT, '
            'vector BLOB NOT NULL, '
            'content TEXT NOT NULL, '
            'source TEXT NOT NULL)'
        )


def store_chunk(content: str, vector: list[float], source: str, path: str = DB_PATH) -> None:
    """插入一条知识块（content + 向量 + 来源）。"""
    blob = array('f', vector).tobytes()
    with sqlite3.connect(path) as conn:
        conn.execute(
            'INSERT INTO chunks (vector, content, source) VALUES (?, ?, ?)',
            (blob, content, source),
        )


def fetch_all(path: str = DB_PATH) -> list[tuple[int, list[float], str]]:
    """取全部块：(id, vector, content)。"""
    with sqlite3.connect(path) as conn:
        rows = conn.execute('SELECT id, vector, content FROM chunks').fetchall()
    out = []
    for row_id, blob, content in rows:
        vec = array('f')
        vec.frombytes(blob)
        out.append((row_id, vec.tolist(), content))
    return out


def clear_chunks(path: str = DB_PATH) -> None:
    """清空知识库（重建前调用）。"""
    with sqlite3.connect(path) as conn:
        conn.execute('DELETE FROM chunks')
