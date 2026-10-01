"""chunk_markdown 单元测试：两级切块策略。"""

from rag.chunking import WINDOW, chunk_markdown


def test_atx_heading_sections():
    """# 标题分节：一节一块。"""
    text = '# 一、引言\n正文A\n\n# 二、正文\n正文B'
    chunks = list(chunk_markdown(text))
    assert len(chunks) == 2
    assert chunks[0].startswith('# 一、引言')
    assert chunks[1].startswith('# 二、正文')


def test_bracket_heading_sections():
    """【】标题分节（学习手册原生格式）。"""
    text = '【一、引言】\n正文A\n\n【二、正文】\n正文B'
    chunks = list(chunk_markdown(text))
    assert len(chunks) == 2
    assert chunks[0].startswith('【一、引言】')
    assert chunks[1].startswith('【二、正文】')


def test_mixed_headings():
    """# 与【】混合出现时都识别为分节点。"""
    text = '# Markdown 标题\n正文A\n\n【学习手册节】\n正文B'
    chunks = list(chunk_markdown(text))
    assert len(chunks) == 2
    assert chunks[0].startswith('# Markdown 标题')
    assert chunks[1].startswith('【学习手册节】')


def test_heading_kept_as_anchor():
    """标题行必须保留在块内（检索命中信号）。"""
    text = '# 检索最佳实践\n正文内容'
    chunk = next(iter(chunk_markdown(text)))
    assert chunk.startswith('# 检索最佳实践')


def test_no_heading_falls_back_to_window():
    """无标题文档退化为固定窗口切。"""
    body = '这是一段没有标题的纯正文。' * 200  # 约 2800 字
    chunks = list(chunk_markdown(body))
    assert len(chunks) > 1
    assert all(len(c) <= WINDOW for c in chunks)


def test_overlong_section_split_inside():
    """超长节内再固定窗口切（两级策略）。"""
    text = '# 超长节\n' + '正文内容。' * 400  # 约 1800 字
    chunks = list(chunk_markdown(text))
    assert len(chunks) > 1
    assert all(len(c) <= WINDOW for c in chunks)


def test_empty_input():
    """空输入返回空迭代。"""
    assert list(chunk_markdown('')) == []


def test_single_section_without_heading_plus_heading_later():
    """文档开头无标题段落 + 后续标题：开头段落并入第一节。"""
    text = '前言段落\n\n# 正式章节\n正文内容'
    chunks = list(chunk_markdown(text))
    assert len(chunks) == 2
    assert '# 正式章节' in chunks[1]
