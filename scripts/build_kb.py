"""构建知识库 CLI：文件 → 归一化 → 切块 → 向量化 → 入库（幂等）。

用法：
    uv run python scripts/build_kb.py <文件路径> [--reset]

- 不传 --reset：建表（如无），增量入库，同 (source, content) 只更新向量，不重复插入
- 传 --reset：先 DROP 表再建（换模型导致维度变化时必须重建）
"""

import argparse
from pathlib import Path

from rag.chunking import chunk_markdown
from rag.embed import embed
from rag.normalize import normalize_md
from rag.store import count_chunks, init_db, store_chunks


def build_kb(file: str, reset: bool = False) -> int:
    """构建知识库，返回本次入库块数。"""
    init_db(reset=reset)
    path = Path(file)
    text = path.read_text(encoding='utf-8')
    chunks = list(chunk_markdown(normalize_md(text)))
    source = path.name

    rows = []
    for chunk in chunks:
        vec = embed(chunk)
        rows.append((chunk, vec, source))
    store_chunks(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='构建 RAG 知识库')
    parser.add_argument('file', help='源文档路径（txt/md，UTF-8）')
    parser.add_argument('--reset', action='store_true', help='重建表（换模型后必须）')
    args = parser.parse_args()

    n = build_kb(args.file, reset=args.reset)
    print(f'本次入库 {n} 块，库内共 {count_chunks()} 条')


if __name__ == '__main__':
    main()
