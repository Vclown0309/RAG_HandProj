# -*- coding: utf-8 -*-
"""切块器：两级策略（标题分节 + 超长节内固定窗口 + 无标题兜底）。

策略来源（M2 设计）：
    - 标题是语义边界：`# `（ATX，anydoc/md 输出）与 `【】`（学习手册原生）都是分节点。
    - 一节一块；节超长（> WINDOW）时节内再固定窗口切（两级）。
    - 无标题文档退化为固定窗口切（旧 read_split_text 行为），标题兜底分支。
    - 标题行保留在块内——它是内容的锚点，也是检索时的命中信号。
"""

import re
from collections.abc import Iterator

WINDOW = 500
OVERLAP = 50

_ATX_HEADING = re.compile(r'^#{1,6}\s+\S')


def _is_heading(line: str) -> bool:
    """标题判定：ATX（# 开头）或学习手册风格（【】包裹）。"""
    s = line.strip()
    return bool(_ATX_HEADING.match(s)) or (s.startswith('【') and s.endswith('】'))


def _split_window(text: str) -> Iterator[str]:
    """固定窗口切：WINDOW 长、OVERLAP 重叠，含头不含尾。"""
    step = WINDOW - OVERLAP
    for i in range(0, len(text), step):
        yield text[i : i + WINDOW]


def chunk_markdown(text: str) -> Iterator[str]:
    """两级切块：按标题分节；超长节内窗口切；无标题退化为窗口切。"""
    lines = text.splitlines()
    if not any(_is_heading(line) for line in lines):
        yield from _split_window(text)
        return

    current: list[str] = []
    for line in lines:
        if _is_heading(line):
            if current:
                section = '\n'.join(current).strip()
                if len(section) > WINDOW:
                    yield from _split_window(section)
                else:
                    yield section
            current = [line]
        else:
            current.append(line)

    if current:
        section = '\n'.join(current).strip()
        if len(section) > WINDOW:
            yield from _split_window(section)
        else:
            yield section
