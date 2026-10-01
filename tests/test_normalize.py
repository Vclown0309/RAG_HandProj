# -*- coding: utf-8 -*-
"""normalize_md 单元测试：扩展方言降级到 CommonMark/GFM 子集。"""

from rag.normalize import normalize_md


def test_real_heading_from_anydoc_preserved():
    """anydoc 输出的真实 # 标题必须原样保留（核心链路：标题切块依赖它）。"""
    src = '# 1. 环境事实\n\n正文内容'
    assert normalize_md(src) == src


def test_md_native_heading_untouched():
    """原生 ## 标题原样保留。"""
    src = '## 2.1 两个服务\n\n正文'
    assert normalize_md(src) == src


def test_inline_bold_kept():
    """行内粗体（anydoc 转义形式）原样保留，绝不提升为标题。"""
    src = '这是一个\\*\\*加粗强调\\**的段落。'
    assert normalize_md(src) == src


def test_callout_downgraded_to_plain_quote():
    """Obsidian callout 降级：去 [!note] 标记，保留引用文本。"""
    src = '> [!note] 这是一个重要提示'
    assert normalize_md(src) == '> 这是一个重要提示'


def test_callout_with_type_downgraded():
    """带类型的 callout（[!warning]）同样降级。"""
    src = '> [!warning] 磁盘空间不足'
    assert normalize_md(src) == '> 磁盘空间不足'


def test_footnote_reference_removed():
    """脚注引用 [^1] 降级删除。"""
    src = '正文内容[^1]'
    assert normalize_md(src) == '正文内容'


def test_table_untouched():
    """表格原样保留（切块器需要表格边界）。"""
    src = '| **角色** | **模型** |\n| --- | --- |\n| 嵌入 | Qwen |'
    assert normalize_md(src) == src


def test_normal_paragraph_untouched():
    """普通段落原样保留。"""
    src = '这是普通正文段落，包含中文标点。'
    assert normalize_md(src) == src
