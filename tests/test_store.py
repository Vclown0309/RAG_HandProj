"""SQLite 存储单元测试：临时库验证序列化/去重/检索。"""

import sqlite3
import struct

import pytest

from rag.store import (
    add_history,
    clear_chunks,
    count_chunks,
    delete_source,
    drop_table,
    fetch_all,
    from_blob,
    get_history,
    init_db,
    list_sources,
    search,
    store_chunk,
    store_chunks,
    to_blob,
)

DIM = 4096


# ------------------------------------------------------------------ fixtures

@pytest.fixture
def db(tmp_path):
    """每个用例一个干净的空库。"""
    path = str(tmp_path / "kb.db")
    init_db(path, reset=True)
    return path


def vec_of(value: float, dim: int = DIM) -> list[float]:
    return [value] * dim


# ------------------------------------------------------------------ 序列化

def test_blob_size_is_dim_times_4():
    blob = to_blob(vec_of(0.1))
    assert len(blob) == DIM * 4 == 16384


def test_roundtrip_returns_f32_precision():
    v = [i * 0.001 for i in range(DIM)]
    back = from_blob(to_blob(v))
    assert len(back) == DIM
    assert max(abs(a - b) for a, b in zip(v, back)) < 1e-6


def test_blob_is_little_endian():
    """小端：低有效字节排在前。float32 的 1.0 = 0x3F800000 -> 00 00 80 3F"""
    assert to_blob([1.0]) == bytes.fromhex("0000803f")
    assert to_blob([1.0]) == struct.pack("<f", 1.0)


def test_endianness_roundtrip_is_symmetric():
    """不管本机是什么序，写进去读出来必须一致。"""
    v = [1.0, -2.5, 3.14]
    assert from_blob(to_blob(v)) == pytest.approx(v, abs=1e-6)


def test_dim_mismatch_raises():
    with pytest.raises(ValueError, match="维度不符"):
        to_blob([1.0, 2.0], dim=DIM)


# ------------------------------------------------------------------ 写入 / 去重

def test_repeat_insert_is_idempotent(db):
    rows = [("RAG 是检索增强生成", vec_of(0.1), "doc/a.md")]
    store_chunks(rows, path=db)
    store_chunks(rows, path=db)          # 完全重跑一遍
    assert count_chunks(db) == 1


def test_incremental_append_keeps_old_data(db):
    store_chunk("第一条", vec_of(0.1), "doc/a.md", path=db)
    store_chunk("第二条", vec_of(0.2), "doc/b.md", path=db)
    store_chunks([("第三条", vec_of(0.3), "doc/c.md")], path=db)
    assert count_chunks(db) == 3
    assert [c for _, _, c, _ in fetch_all(db)] == ["第一条", "第二条", "第三条"]


def test_same_content_different_source_are_distinct(db):
    store_chunk("相同文本", vec_of(0.1), "doc/a.md", path=db)
    store_chunk("相同文本", vec_of(0.2), "doc/b.md", path=db)
    assert count_chunks(db) == 2


def test_same_content_updates_vector(db):
    store_chunk("待更新", vec_of(0.1), "doc/a.md", path=db)
    store_chunk("待更新", vec_of(0.9), "doc/a.md", path=db)
    assert count_chunks(db) == 1
    assert fetch_all(db)[0][1][0] == pytest.approx(0.9, abs=1e-6)


@pytest.mark.parametrize(
    "mode,expected_first_value",
    [("ignore", 0.1), ("update", 0.9)],
)
def test_on_conflict_modes(db, mode, expected_first_value):
    """ignore 保留旧向量；update 覆盖为新向量。"""
    store_chunk("内容", vec_of(0.1), "doc/a.md", path=db)
    store_chunk("内容", vec_of(0.9), "doc/a.md", path=db, on_conflict=mode)
    assert count_chunks(db) == 1
    assert fetch_all(db)[0][1][0] == pytest.approx(expected_first_value, abs=1e-6)


def test_on_conflict_error_raises(db):
    store_chunk("内容", vec_of(0.1), "doc/a.md", path=db)
    with pytest.raises(sqlite3.IntegrityError):
        store_chunk("内容", vec_of(0.2), "doc/a.md", path=db, on_conflict="error")


def test_batch_write_is_single_transaction(db):
    store_chunks(
        [(f"块{i}", vec_of(i * 0.01), "doc/x.md") for i in range(50)], path=db
    )
    assert count_chunks(db) == 50


# ------------------------------------------------------------------ 读取 / 检索

def test_fetch_all_returns_source(db):
    store_chunk("内容", vec_of(0.1), "doc/a.md", path=db)
    _rid, vector, content, source = fetch_all(db)[0]
    assert (content, source) == ("内容", "doc/a.md")
    assert len(vector) == DIM


def test_search_ranks_by_cosine(db):
    store_chunk("不像", [-1.0] + [0.0] * (DIM - 1), "doc/a.md", path=db)
    store_chunk("一模一样", [1.0] + [0.0] * (DIM - 1), "doc/b.md", path=db)
    store_chunk("有点像", [0.5] + [0.5] * (DIM - 1), "doc/c.md", path=db)

    hits = search([1.0] + [0.0] * (DIM - 1), top_k=3, path=db)
    assert [h[2] for h in hits] == ["一模一样", "有点像", "不像"]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-6)
    assert hits[0][1] > hits[1][1] > hits[2][1]


def test_search_top_k_limits_results(db):
    store_chunks([(f"块{i}", vec_of(i * 0.01), "doc/x.md") for i in range(10)], path=db)
    assert len(search(vec_of(0.1), top_k=3, path=db)) == 3


def test_search_on_empty_db_returns_empty(db):
    assert search(vec_of(0.1), path=db) == []


# ------------------------------------------------------------------ 重置方式

def test_clear_keeps_table(db):
    store_chunk("内容", vec_of(0.1), "doc/a.md", path=db)
    clear_chunks(db)
    assert count_chunks(db) == 0
    store_chunk("重新插入", vec_of(0.2), "doc/b.md", path=db)  # 表还在，能直接用
    assert count_chunks(db) == 1


def test_drop_removes_table(db):
    store_chunk("内容", vec_of(0.1), "doc/a.md", path=db)
    drop_table(db)
    with sqlite3.connect(db) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "chunks" not in tables


def test_init_db_reset_rebuilds_from_scratch(db):
    store_chunk("内容", vec_of(0.1), "doc/a.md", path=db)
    init_db(db, reset=True)
    assert count_chunks(db) == 0


def test_init_db_is_idempotent(db):
    """不传 reset 时，重复建表不应报错也不应丢数据。"""
    store_chunk("内容", vec_of(0.1), "doc/a.md", path=db)
    init_db(db)
    assert count_chunks(db) == 1


# ------------------------------------------------------------------ 问答历史

def test_add_and_get_history(db):
    add_history("第一个问题", "答案一", path=db)
    add_history("第二个问题", "答案二", path=db)
    rows = get_history(path=db)
    assert len(rows) == 2
    # 最新在前
    assert rows[0][1:3] == ("第二个问题", "答案二")
    assert rows[1][1:3] == ("第一个问题", "答案一")
    # 带时间戳
    assert rows[0][3]


def test_history_limit(db):
    for i in range(5):
        add_history(f"问题{i}", f"答案{i}", path=db)
    assert len(get_history(limit=2, path=db)) == 2


# ------------------------------------------------------------------ 文档管理

def test_list_sources_groups_by_source(db):
    store_chunk("a1", vec_of(0.1), "a.md", path=db)
    store_chunk("a2", vec_of(0.1), "a.md", path=db)
    store_chunk("b1", vec_of(0.1), "b.md", path=db)
    assert list_sources(path=db) == [("a.md", 2), ("b.md", 1)]


def test_delete_source_only_removes_target(db):
    store_chunk("a1", vec_of(0.1), "a.md", path=db)
    store_chunk("b1", vec_of(0.1), "b.md", path=db)
    deleted = delete_source("a.md", path=db)
    assert deleted == 1
    assert count_chunks(db) == 1
    assert list_sources(path=db) == [("b.md", 1)]


def test_delete_missing_source_returns_zero(db):
    assert delete_source("ghost.md", path=db) == 0
