"""切块器：两级策略（标题分节 + 超长节内固定窗口 + 无标题兜底）。

策略来源（M2 设计）：
    - 标题是语义边界：`# `（ATX，anydoc/md 输出）与 `【】`（学习手册原生）都是分节点。
    - 一节一块；节超长（> WINDOW）时节内再固定窗口切（两级）。
    - 无标题文档退化为固定窗口切（旧 read_split_text 行为），标题兜底分支。
    - 标题行保留在块内——它是内容的锚点，也是检索时的命中信号。

M3 修正（真实链路暴露）：
    - 多级标题过度切分：`### 4.1` 子节也被切成独立块，导致顶级节只剩标题没正文。
    - 改为动态分节级别：只按文档最浅标题层级分节，次级标题留在节内作内容锚点。
"""

import re
from collections import Counter
from collections.abc import Iterator

WINDOW = 500
OVERLAP = 50

_ATX_HEADING = re.compile(r'^#{1,6}\s+\S')


def _is_heading(line: str) -> bool:
    """标题判定：ATX（# 开头）或学习手册风格（【】包裹）。"""
    s = line.strip()
    return bool(_ATX_HEADING.match(s)) or (s.startswith('【') and s.endswith('】'))


def _heading_level(line: str) -> int:
    """标题层级：ATX 用 # 数量；【】视为 1 级（最浅）。"""
    s = line.strip()
    if s.startswith('#'):
        return len(s) - len(s.lstrip('#'))
    return 1


def _split_level_of(heads: list[int]) -> int:
    """分节级别：出现 >= 2 次的最浅标题级别（章节标题）。

    单条的极浅标题（如文档主标题 `#`）不作为分节点——
    它是整篇文档的名字，不是章节边界。
    """
    counter = Counter(heads)
    for level in sorted(counter):
        if counter[level] >= 2:
            return level
    return min(heads)


def _split_window(text: str) -> Iterator[str]:
    """固定窗口切：WINDOW 长、OVERLAP 重叠，含头不含尾。"""
    step = WINDOW - OVERLAP
    # 直接通过range来实现 步长 以及 始末点的限制
    # 利用yield 加 每次窗口移动的列表切片 同时完成了 500个每块 每块重复50个字的目标
    for i in range(0, len(text), step):
        yield text[i: i + WINDOW]


def _emit(section: str) -> Iterator[str]:
    """节输出：超长节内窗口切，否则原样一块。"""
    if len(section) > WINDOW:
        yield from _split_window(section)
    else:
        yield section


def chunk_markdown(text: str) -> Iterator[str]:
    """两级切块：按最浅标题分节；次级标题留在节内；超长节内窗口切；无标题退化为窗口切。"""
    lines = text.splitlines()
    # 扫描全部标题层级，取「出现 >= 2 次的最浅级」作为分节级别
    # （### 及更深并入所属节；单条主标题不作为分节点）
    heads = [_heading_level(line) for line in lines if _is_heading(line)]
    if not heads:
        # 文档无任何标题，退化为固定窗口切
        yield from _split_window(text)
        return
    split_level = _split_level_of(heads)

    current: list[str] = []
    started = False
    for line in lines:
        # 只有恰好等于分节级别的标题才新开一节；
        # 更浅的单条主标题并入首节，更深（###+）留在节内作内容锚点
        if _is_heading(line) and _heading_level(line) == split_level:
            if not started:
                # 文档头（主标题/序言）并进第一节，不单独成块
                current.append(line)
                started = True
            else:
                if current:
                    yield from _emit('\n'.join(current).strip())
                current = [line]
        else:
            current.append(line)
    if current:
        yield from _emit('\n'.join(current).strip())
