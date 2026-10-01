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
    # 直接通过range来实现 步长 以及 始末点的限制
    # 利用yield 加 每次窗口移动的列表切片 同时完成了 500个每块 每块重复50个字的目标
    for i in range(0, len(text), step):
        yield text[i: i + WINDOW]


def chunk_markdown(text: str) -> Iterator[str]:
    """两级切块：按标题分节；超长节内窗口切；无标题退化为窗口切。"""
    lines = text.splitlines()
    # any 函数实现判断是否文件内容为空
    if not any(_is_heading(line) for line in lines):
        yield from _split_window(text)
        return

    current: list[str] = []
    for line in lines:
        # 判断是否是标题行
        if _is_heading(line):
            # 判断是否含内容 含有内容 先对上一步的标题+内容收尾
            if current:
                section = '\n'.join(current).strip()
                # 二级切块拆分 标题下的内容大于设定窗口长度 进行无标题切块逻辑
                if len(section) > WINDOW:
                    yield from _split_window(section)
                else:
                    yield section
            # 不含内容，那么就作为 current 列表的第一个元素
            current = [line]
        else:
            # 非标题行，在current列表尾部追加内容即可
            current.append(line)
    # 最后一部分，下面没有标题了，那么跳出循环的这最后一部分，再来执行下yield返回就完成了所有内容的切块了
    if current:
        section = '\n'.join(current).strip()
        # 二级切块拆分 标题下的内容大于设定窗口长度 进行无标题切块逻辑
        if len(section) > WINDOW:
            yield from _split_window(section)
        else:
            yield section
