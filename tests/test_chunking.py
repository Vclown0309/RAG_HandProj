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
    body = '这是一段没有标题的纯正文。' * 200  # 约 2600 字
    chunks = list(chunk_markdown(body))
    assert len(chunks) > 1
    assert all(len(c) <= WINDOW for c in chunks)


def test_overlong_section_split_inside():
    """超长节内再固定窗口切（两级策略）。"""
    text = '# 超长节\n' + '正文内容。' * 400  # 约 2000 字
    chunks = list(chunk_markdown(text))
    assert len(chunks) > 1
    assert all(len(c) <= WINDOW for c in chunks)


def test_empty_input():
    """空输入返回空迭代。"""
    assert list(chunk_markdown('')) == []


def test_leading_text_merged_into_first_section():
    """文档开头无标题段落：并入首个分节，不单独成块（检索时前言不孤立）。"""
    text = '前言段落\n\n# 正式章节\n正文内容'
    chunks = list(chunk_markdown(text))
    assert len(chunks) == 1
    assert chunks[0].startswith('前言段落')
    assert '# 正式章节' in chunks[0]


def test_subheading_merged_into_section():
    """次级标题（###）不单独分节，并入所属 ## 节（M3 修复：防路标块）。"""
    text = '## 4. 向量化\n### 4.1 结论\n结论正文\n## 5. 切块\n切块正文'
    chunks = list(chunk_markdown(text))
    assert len(chunks) == 2
    assert '### 4.1 结论' in chunks[0]  # 次级标题留在节内作锚点
    assert '结论正文' in chunks[0]
    assert chunks[1].startswith('## 5. 切块')


def test_deepest_heading_is_split_level():
    """最浅标题 # 出现多次时是分节级别，##/### 全部并入该节。"""
    text = '# 总览\n## 子节\n### 孙节\n正文\n# 第二总览\n内容'
    chunks = list(chunk_markdown(text))
    assert len(chunks) == 2
    assert '### 孙节' in chunks[0]
    assert chunks[1].startswith('# 第二总览')


def test_single_document_title_not_split_level():
    """单条 # 主标题不作为分节点；## 章节才是（M4 修复：防整篇并一节）。"""
    text = '# 文档主标题\n## 第一章\n### 1.1 小节\n内容A\n## 第二章\n内容B'
    chunks = list(chunk_markdown(text))
    assert len(chunks) == 2
    # 第一章块包含主标题 + 第一章（含其子节）
    assert chunks[0].startswith('# 文档主标题')
    assert '### 1.1 小节' in chunks[0]
    assert '内容A' in chunks[0]
    assert chunks[1].startswith('## 第二章')
