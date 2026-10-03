"""/ask /history /docs 接口测试：链路用 monkeypatch，存储用临时库（不碰 kb.db）。"""

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

FAKE_HITS = [
    (1, 0.9, '资料内容A', 'a.md#0'),
    (2, 0.8, '资料内容B', 'b.md#1'),
]


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """把 main.DB_LOCAL 指向临时库并建表，所有 store 调用落到测试库。"""
    db = tmp_path / 'test.db'
    monkeypatch.setattr('main.DB_LOCAL', str(db))
    from rag.store import init_db

    init_db(str(db))
    return str(db)


def test_root() -> None:
    resp = client.get('/')
    assert resp.status_code == 200
    assert 'text/html' in resp.headers['content-type']
    assert 'RAG 演示台' in resp.text


def test_static_page_has_source_panel() -> None:
    """演示页含知识库管理 + 参考源侧边栏 + 提问表单。"""
    resp = client.get('/')
    assert resp.status_code == 200
    assert 'sources-section' in resp.text
    assert 'library-section' in resp.text
    assert 'ask-form' in resp.text


def test_ask_full_pipeline(monkeypatch, tmp_db) -> None:
    """全链路正常：返回答案 + 来源 + 落历史。"""
    monkeypatch.setattr('main.embed', lambda q: [0.1, 0.2])
    monkeypatch.setattr('main.search', lambda v, top_k=3, path=None: FAKE_HITS)
    monkeypatch.setattr('main.chat', lambda p: '根据资料，RAG 是检索增强生成。')

    resp = client.post('/ask', json={'query': '什么是 RAG？'})
    assert resp.status_code == 200
    body = resp.json()
    assert body['query'] == '什么是 RAG？'
    assert 'RAG 是检索增强生成' in body['answer']
    assert len(body['sources']) == 2

    hist = client.get('/history').json()['history']
    assert len(hist) == 1
    assert hist[0]['query'] == '什么是 RAG？'


def test_ask_no_hits(monkeypatch, tmp_db) -> None:
    """空命中：提示先入库，不调生成，仍落历史。"""
    monkeypatch.setattr('main.embed', lambda q: [0.1])
    monkeypatch.setattr('main.search', lambda v, top_k=3, path=None: [])
    monkeypatch.setattr('main.chat', lambda p: (_ for _ in ()).throw(AssertionError('不该调生成')))

    resp = client.post('/ask', json={'query': '未知问题'})
    assert resp.status_code == 200
    assert '知识库暂无相关内容' in resp.json()['answer']
    assert resp.json()['sources'] == []
    assert len(client.get('/history').json()['history']) == 1


def test_ask_embedding_error(monkeypatch) -> None:
    """向量化服务挂了：502，不落历史。"""
    from rag.embed import EmbeddingError

    def boom(q):
        raise EmbeddingError('8081 没起来')

    monkeypatch.setattr('main.embed', boom)
    resp = client.post('/ask', json={'query': '问题'})
    assert resp.status_code == 502
    assert '8081 没起来' in resp.json()['detail']


def test_ask_generation_error(monkeypatch, tmp_db) -> None:
    """生成服务挂了：502，不落历史。"""
    from rag.generate import GenerateError

    monkeypatch.setattr('main.embed', lambda q: [0.1])
    monkeypatch.setattr('main.search', lambda v, top_k=3, path=None: FAKE_HITS)

    def boom(p):
        raise GenerateError('8080 没起来')

    monkeypatch.setattr('main.chat', boom)
    resp = client.post('/ask', json={'query': '问题'})
    assert resp.status_code == 502
    assert '8080 没起来' in resp.json()['detail']


def test_docs_upload_list_delete(tmp_db, monkeypatch) -> None:
    """上传 txt 入库 → 列表可见 → 删除 → 列表为空。"""
    monkeypatch.setattr('main.embed', lambda chunk: [0.1, 0.2])

    resp = client.post(
        '/library',
        files={'file': ('notes.txt', '# 第一节\n正文内容\n\n# 第二节\n更多正文'.encode(), 'text/plain')},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body['source'] == 'notes.txt'
    assert body['added'] == 2

    sources = client.get('/library').json()['sources']
    assert sources == [{'source': 'notes.txt', 'chunks': 2}]

    resp = client.delete('/library/notes.txt')
    assert resp.status_code == 200
    assert resp.json()['deleted'] == 2
    assert client.get('/library').json()['sources'] == []


def test_docs_delete_missing(tmp_db) -> None:
    """删除不存在的文档：404。"""
    resp = client.delete('/library/ghost.md')
    assert resp.status_code == 404


def test_docs_empty_file(tmp_db, monkeypatch) -> None:
    """空文件：400，不入库。"""
    monkeypatch.setattr('main.embed', lambda chunk: [0.1])
    resp = client.post('/library', files={'file': ('empty.txt', b'', 'text/plain')})
    assert resp.status_code == 400
    assert '内容为空' in resp.json()['detail']


def test_docs_gbk_ok(tmp_db, monkeypatch) -> None:
    """GBK 编码 txt：编码自适应，应入库成功（200）。"""
    monkeypatch.setattr('main.embed', lambda chunk: [0.1])
    resp = client.post('/library', files={'file': ('gbk.txt', '中文GBK内容'.encode('gbk'), 'text/plain')})
    assert resp.status_code == 200, resp.text
    assert resp.json()['added'] >= 1
    sources = client.get('/library').json()['sources']
    assert any(s['source'] == 'gbk.txt' for s in sources)


def test_docs_bad_binary(tmp_db) -> None:
    """txt 扩展名但内容是完全无法识别的二进制：400（三个编码都解不出）。"""
    resp = client.post('/library', files={'file': ('bad.txt', bytes(range(255)), 'text/plain')})
    assert resp.status_code == 400
    assert '编码' in resp.json()['detail']


def test_docs_unsupported_ext(tmp_db) -> None:
    """不支持扩展名：400。"""
    resp = client.post('/library', files={'file': ('a.pdf', b'%PDF-1.4', 'application/pdf')})
    assert resp.status_code == 400
    assert '暂支持' in resp.json()['detail']


def test_docs_docx_ok(tmp_db, monkeypatch) -> None:
    """docx 上传：提取为 markdown 并入库（200）。"""
    import io
    import zipfile

    xml = ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           '<w:body>'
           '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>第一章 测试</w:t></w:r></w:p>'
           '<w:p><w:r><w:t>这是一段正文。</w:t></w:r></w:p>'
           '</w:body></w:document>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('word/document.xml', xml)

    monkeypatch.setattr('main.embed', lambda chunk: [0.1])
    resp = client.post('/library', files={'file': ('a.docx', buf.getvalue(), 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')})
    assert resp.status_code == 200, resp.text
    assert resp.json()['added'] >= 1
    sources = client.get('/library').json()['sources']
    assert any(s['source'] == 'a.docx' for s in sources)
