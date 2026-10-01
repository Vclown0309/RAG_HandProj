"""/ask 接口测试：monkeypatch 掉链路（embed/search/chat），验证接口契约。"""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

FAKE_HITS = [
    (1, 0.9, '资料内容A', 'a.md#0'),
    (2, 0.8, '资料内容B', 'b.md#1'),
]


def test_root() -> None:
    resp = client.get('/')
    assert resp.status_code == 200
    assert resp.json() == {'service': 'RAG 问答', 'status': 'ok'}


def test_ask_full_pipeline(monkeypatch) -> None:
    """全链路正常：返回答案 + 来源。"""
    monkeypatch.setattr('main.embed', lambda q: [0.1, 0.2])
    monkeypatch.setattr('main.search', lambda v, top_k=3: FAKE_HITS)
    monkeypatch.setattr('main.chat', lambda p: '根据资料，RAG 是检索增强生成。')

    resp = client.post('/ask', json={'query': '什么是 RAG？'})
    assert resp.status_code == 200
    body = resp.json()
    assert body['query'] == '什么是 RAG？'
    assert 'RAG 是检索增强生成' in body['answer']
    assert len(body['sources']) == 2
    assert body['sources'][0]['source'] == 'a.md#0'


def test_ask_no_hits(monkeypatch) -> None:
    """空命中：提示先入库，不调生成。"""
    monkeypatch.setattr('main.embed', lambda q: [0.1])
    monkeypatch.setattr('main.search', lambda v, top_k=3: [])
    monkeypatch.setattr('main.chat', lambda p: (_ for _ in ()).throw(AssertionError('不该调生成')))

    resp = client.post('/ask', json={'query': '未知问题'})
    assert resp.status_code == 200
    assert '知识库暂无相关内容' in resp.json()['answer']
    assert resp.json()['sources'] == []


def test_ask_embedding_error(monkeypatch) -> None:
    """向量化服务挂了：502。"""
    from rag.embed import EmbeddingError

    def boom(q):
        raise EmbeddingError('8081 没起来')

    monkeypatch.setattr('main.embed', boom)
    resp = client.post('/ask', json={'query': '问题'})
    assert resp.status_code == 502
    assert '8081 没起来' in resp.json()['detail']


def test_ask_generation_error(monkeypatch) -> None:
    """生成服务挂了：502。"""
    from rag.generate import GenerateError

    monkeypatch.setattr('main.embed', lambda q: [0.1])
    monkeypatch.setattr('main.search', lambda v, top_k=3: FAKE_HITS)

    def boom(p):
        raise GenerateError('8080 没起来')

    monkeypatch.setattr('main.chat', boom)
    resp = client.post('/ask', json={'query': '问题'})
    assert resp.status_code == 502
    assert '8080 没起来' in resp.json()['detail']
