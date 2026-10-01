"""SQLite 存储单元测试：用临时库验证 roundtrip。"""

import pytest

from rag.store import clear_chunks, fetch_all, init_db, store_chunk


def test_store_and_fetch_roundtrip(tmp_path):
    """写入一条块，取回后向量与内容一致（float32 语义：用可精确表示的值）。"""
    db = str(tmp_path / 'test_kb.db')
    init_db(db)
    vec = [0.5, 0.25, 0.75] * 100  # float32 可精确表示的分数
    store_chunk('测试内容', vec, 'test.md', db)

    rows = fetch_all(db)
    assert len(rows) == 1
    row_id, back_vec, content = rows[0]
    assert content == '测试内容'
    assert len(back_vec) == len(vec)
    assert back_vec == vec


def test_float64_truncated_to_float32(tmp_path):
    """float64 字面量经 float32 存储后按 1e-6 容差近似（模型输出本就是 float32）。"""
    db = str(tmp_path / 'test_kb.db')
    init_db(db)
    vec = [0.1, 0.2, 0.3] * 100
    store_chunk('内容', vec, 'test.md', db)

    _, back_vec, _ = fetch_all(db)[0]
    assert back_vec == pytest.approx(vec, abs=1e-6)


def test_multiple_chunks_preserve_order(tmp_path):
    """多条写入按 id 顺序取回。"""
    db = str(tmp_path / 'test_kb.db')
    init_db(db)
    for i in range(3):
        store_chunk(f'块 {i}', [float(i)] * 4, 'test.md', db)

    rows = fetch_all(db)
    assert [c for _, _, c in rows] == ['块 0', '块 1', '块 2']
    assert [rid for rid, _, _ in rows] == [1, 2, 3]


def test_clear_chunks_empties_table(tmp_path):
    """清空后无残留。"""
    db = str(tmp_path / 'test_kb.db')
    init_db(db)
    store_chunk('内容', [1.0] * 4, 'test.md', db)
    clear_chunks(db)
    assert fetch_all(db) == []


def test_empty_db_returns_empty(tmp_path):
    """空库返回空列表。"""
    db = str(tmp_path / 'test_kb.db')
    init_db(db)
    assert fetch_all(db) == []
