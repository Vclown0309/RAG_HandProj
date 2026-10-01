"""SQLite 存储：chunks 表（id / vector BLOB / content / source / chunk_hash）。

向量以 float32 数组序列化为 BLOB 存储（每条 4096 维 = 16 KB），
检索时取出再还原为 list[float]。

去重策略：按 (source, content) 的哈希建 UNIQUE 约束，重复写入退化为「更新向量」，
所以可以任意分次追加，不需要每次清空整表。
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
import sys
from array import array
from collections.abc import Iterable, Sequence
from contextlib import closing
from typing import cast

DB_PATH = "kb.db"

# 'f' = C float，即 IEEE-754 单精度，4 字节
_TYPECODE = "f"
_BIG_ENDIAN = sys.byteorder == "big"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    vector     BLOB    NOT NULL,
    content    TEXT    NOT NULL,
    source     TEXT    NOT NULL,
    chunk_hash TEXT    NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);

CREATE TABLE IF NOT EXISTS history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    query      TEXT NOT NULL,
    answer     TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
"""


# ---------------------------------------------------------------- 连接 / 建表

def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(path: str = DB_PATH, reset: bool = False) -> None:
    """建表。reset=True 时先 DROP 再建（换模型导致维度变化时用）。"""
    with closing(_connect(path)) as conn:
        if reset:
            conn.execute("DROP TABLE IF EXISTS chunks")
        conn.executescript(_SCHEMA)
        conn.commit()


def drop_table(path: str = DB_PATH) -> None:
    """删表。适合原理阶段的快速重来。"""
    with closing(_connect(path)) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.commit()


def clear_chunks(path: str = DB_PATH, reset_id: bool = True) -> None:
    """清空数据但保留表结构。适合大表重建时保留 schema。"""
    with closing(_connect(path)) as conn:
        conn.execute("DELETE FROM chunks")
        if reset_id:
            conn.execute("DELETE FROM sqlite_sequence WHERE name='chunks'")
        conn.commit()


def count_chunks(path: str = DB_PATH) -> int:
    with closing(_connect(path)) as conn:
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


# ---------------------------------------------------------------- 序列化层

def to_blob(vector: Sequence[float], dim: int | None = None) -> bytes:
    """list[float] -> BLOB（固定输出小端字节序）。"""
    if dim is not None and len(vector) != dim:
        raise ValueError(f"维度不符：期望 {dim}，实际 {len(vector)}")
    buf = array(_TYPECODE, vector)
    if _BIG_ENDIAN:
        buf.byteswap()
    return buf.tobytes()


def from_blob(blob: bytes) -> list[float]:
    """BLOB -> list[float]。"""
    buf = array(_TYPECODE)
    buf.frombytes(blob)
    if _BIG_ENDIAN:
        buf.byteswap()
    return cast(list[float], buf.tolist())


def _hash(source: str, content: str) -> str:
    """chunk 的稳定身份：同来源 + 同内容 => 同一条记录。"""
    return hashlib.sha256(f"{source}\x00{content}".encode()).hexdigest()


# ---------------------------------------------------------------- 写入

def store_chunk(
    content: str,
    vector: Sequence[float],
    source: str,
    path: str = DB_PATH,
    dim: int | None = None,
    on_conflict: str = "update",
) -> None:
    """写入一条知识块。

    on_conflict:
        'update' -> 内容已存在则只更新向量（默认，幂等）
        'ignore' -> 已存在则跳过
        'error'  -> 已存在则抛 IntegrityError
    """
    store_chunks([(content, vector, source)], path=path, dim=dim, on_conflict=on_conflict)


def store_chunks(
    rows: Iterable[tuple[str, Sequence[float], str]],
    path: str = DB_PATH,
    dim: int | None = None,
    on_conflict: str = "update",
) -> int:
    """批量写入，单事务。rows 元素为 (content, vector, source)。

    返回写入 / 更新的行数。
    """
    if on_conflict == "error":
        sql = ("INSERT INTO chunks (vector, content, source, chunk_hash) "
               "VALUES (?, ?, ?, ?)")
    elif on_conflict == "ignore":
        sql = ("INSERT INTO chunks (vector, content, source, chunk_hash) "
               "VALUES (?, ?, ?, ?) "
               "ON CONFLICT(chunk_hash) DO NOTHING")
    elif on_conflict == "update":
        sql = ("INSERT INTO chunks (vector, content, source, chunk_hash) "
               "VALUES (?, ?, ?, ?) "
               "ON CONFLICT(chunk_hash) DO UPDATE SET vector = excluded.vector")
    else:
        raise ValueError(f"未知 on_conflict: {on_conflict!r}")

    payload = [
        (to_blob(vec, dim), content, source, _hash(source, content))
        for content, vec, source in rows
    ]

    with closing(_connect(path)) as conn:
        cur = conn.executemany(sql, payload)
        conn.commit()
        return cur.rowcount


# ---------------------------------------------------------------- 读取 / 检索

def fetch_all(path: str = DB_PATH) -> list[tuple[int, list[float], str, str]]:
    """取全部块：(id, vector, content, source)。"""
    with closing(_connect(path)) as conn:
        rows = conn.execute(
            "SELECT id, vector, content, source FROM chunks ORDER BY id"
        ).fetchall()
    return [(rid, from_blob(blob), content, source) for rid, blob, content, source in rows]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def search(
    query_vector: Sequence[float],
    top_k: int = 3,
    path: str = DB_PATH,
) -> list[tuple[int, float, str, str]]:
    """暴力余弦检索。返回 (id, score, content, source)，按得分降序。

    演示用：全表扫描 + Python 计算。数据量大了再换 sqlite-vec / FAISS。
    """
    scored = [
        (rid, _cosine(query_vector, vec), content, source)
        for rid, vec, content, source in fetch_all(path)
    ]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------- 问答历史

def add_history(query: str, answer: str, path: str = DB_PATH) -> int:
    """记录一条问答历史，返回新行 id。"""
    with closing(_connect(path)) as conn:
        cur = conn.execute("INSERT INTO history (query, answer) VALUES (?, ?)", (query, answer))
        conn.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid


def get_history(limit: int = 20, path: str = DB_PATH) -> list[tuple[int, str, str, str]]:
    """最近问答历史：(id, query, answer, created_at)，最新在前。"""
    with closing(_connect(path)) as conn:
        return conn.execute(
            "SELECT id, query, answer, created_at FROM history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()


# ---------------------------------------------------------------- 文档管理

def list_sources(path: str = DB_PATH) -> list[tuple[str, int]]:
    """知识库文档清单：(source, 块数)，按来源名排序。"""
    with closing(_connect(path)) as conn:
        return conn.execute(
            "SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source"
        ).fetchall()


def delete_source(source: str, path: str = DB_PATH) -> int:
    """删除某文档的全部块，返回删除行数。"""
    with closing(_connect(path)) as conn:
        cur = conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
        conn.commit()
        return cur.rowcount
