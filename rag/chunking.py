"""切块器：两级策略（标题分节 + 超长节内窗口切 + 无标题兜底）。

策略来源（M2 设计）：
    - 标题是语义边界：`# `（ATX，anydoc/md 输出）与 `【】`（学习手册原生）都是分节点。
    - 一节一块；节超长（> WINDOW）时节内再窗口切（两级）。
    - 无标题文档退化为窗口切（旧 read_split_text 行为），标题兜底分支。
    - 标题行保留在块内——它是内容的锚点，也是检索时的命中信号。

M3 修正（真实链路暴露）：
    - 多级标题过度切分：`### 4.1` 子节也被切成独立块，导致顶级节只剩标题没正文。
    - 改为动态分节级别：只按文档最浅标题层级分节，次级标题留在节内作内容锚点。

M4 修正（真实链路暴露）：
    - 单条极浅标题（文档主标题 `#`）不能当分节点：全篇并成一节被窗口切。
    - 分节级别 = 出现 >= 2 次的最浅层级；文档头（主标题/序言）并入首节。

M5 修正（真实链路暴露）：
    - 字符级硬切会把 markdown 表格拦腰切断（半截表格无意义）。
    - 窗口切改为行级：段落边界断块 + 表格行保护 + 超长单行行内兜底。
    - 行级切在语义边界（段落/行）断，不再需要字符级重叠。
"""

import re
from collections import Counter
from collections.abc import Iterator

WINDOW = 500

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
    """行级窗口切：段落边界断块 + 表格行保护 + 超长单行行内兜底。

    字符级硬切会把句子/表格从中间切断（表格变半截就没意义了）。
    行级切在语义边界（段落/行）断块，语义完整，不再需要字符级重叠；
    表格行（`|` 开头）成组保留，不在表格中间断块。
    """
    lines = text.splitlines()
    buf: list[str] = []
    buf_len = 0

    def flush() -> Iterator[str]:
        if buf:
            yield '\n'.join(buf)

    for line in lines:
        # 超长单行（行级无法切）：行内字符切兜底
        if len(line) > WINDOW:
            yield from flush()
            buf, buf_len = [], 0
            for i in range(0, len(line), WINDOW):
                yield line[i:i + WINDOW]
            continue

        stripped = line.strip()
        is_table_row = stripped.startswith('|')
        prev_is_table = bool(buf) and buf[-1].strip().startswith('|')
        over = buf_len >= WINDOW

        # 断块：超窗 + 表格已完整收尾 +（段落边界空行，或连续长文兜底上限）
        if over and not is_table_row and not prev_is_table and (
            not stripped or buf_len >= WINDOW + 200
        ):
            yield '\n'.join(buf)
            buf, buf_len = [], 0

        # 断块后的空行是段落分隔符，不进入下一块
        if not stripped and not buf:
            continue

        buf.append(line)
        buf_len += len(line) + 1

    yield from flush()


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
