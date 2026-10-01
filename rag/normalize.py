# -*- coding: utf-8 -*-
"""Markdown 方言清理：把扩展方言降级到 CommonMark/GFM 子集。

设计（站在巨人肩膀上）：
    - 标题结构交给转换器（anydoc）原生保留：
      真实 Heading 样式 -> `#` 标题；行内加粗 -> 转义为强调。
      不做"粗体标题提升"启发式——假结构补不出来，且会误伤正文强调。
    - 本模块只清理转换器覆盖不到的扩展方言：
      * Obsidian callout `> [!note] 内容` -> `> 内容`（普通引用）
      * 脚注引用 `[^1]` -> 删除
    未来若需要严格方言收敛（wiki-link、内嵌 HTML 等），升级路径：
    引入 markdown-it-py 解析 AST + 白名单渲染器。
"""

import re

_CALLOUT_PREFIX = re.compile(r'^>\s*\[![a-zA-Z]+\]\s*')
_FOOTNOTE_REF = re.compile(r'\[\^\d+\]')


def normalize_md(text: str) -> str:
    """清理 Markdown 扩展方言，输出 CommonMark/GFM 子集。"""
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        # callout 降级：`> [!note] 内容` -> `> 内容`
        if stripped.startswith('>') and _CALLOUT_PREFIX.search(stripped):
            line = _CALLOUT_PREFIX.sub('> ', line)
        # 脚注引用降级：删除 `[^1]` 标记
        line = _FOOTNOTE_REF.sub('', line)
        out.append(line)
    return '\n'.join(out)
