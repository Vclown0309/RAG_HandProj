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
    """无标题文档退化为窗口切。"""
    body = '这是一段没有标题的纯正文。' * 200  # 约 2600 字
    chunks = list(chunk_markdown(body))
    assert len(chunks) > 1
    assert all(len(c) <= WINDOW for c in chunks)


def test_overlong_section_split_inside():
    """超长节内再窗口切（两级策略）。"""
    text = '# 超长节\n' + '正文内容。' * 400  # 约 2000 字
    chunks = list(chunk_markdown(text))
    assert len(chunks) > 1
    assert all(len(c) <= WINDOW for c in chunks)


def test_table_not_cut_in_half():
    """表格行不可切断：| 连续行成组保留，窗口切不在表格中间断（M5 修复）。"""
    rows = ''.join('| 参数%d | 值%d |\n' % (i, i) for i in range(60))
    text = '# 参数表\n| 参数 | 值 |\n|---|---|\n' + rows + '\n收尾正文。' * 40
    chunks = list(chunk_markdown(text))
    assert len(chunks) >= 2
    # 表格完整保留在首块（表头到末行不缺）
    assert chunks[0].startswith('# 参数表')
    assert '| 参数 | 值 |' in chunks[0]
    assert '| 参数59 | 值59 |' in chunks[0]
    # 其余块不含表格行（表格未被切断/切碎）
    assert all(
        not any(l.strip().startswith('|') for l in c.splitlines())
        for c in chunks[1:]
    )


def test_window_cuts_at_paragraph_boundary():
    """行级窗口切在段落边界（空行）断，不切句子。"""
    para = '这是第%d段。' * 50 % tuple(range(1, 51))  # 约 300 字一段
    text = '\n\n'.join([para, para, para])  # 三段共 900+ 字
    chunks = list(chunk_markdown(text))
    assert len(chunks) >= 2
    # 每块开头是完整段落（以'这是'起头），没有被拦腰切断的句子
    assert all(c.startswith('这是') for c in chunks)


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
