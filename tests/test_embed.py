"""embed 客户端单元测试：mock 掉 requests，不依赖真实服务。"""

import pytest

from rag.embed import DEFAULT_URL, embed


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f'HTTP {self.status_code}')


def test_embed_returns_vector(monkeypatch):
    """正常请求：返回 data[0].embedding。"""

    def fake_post(url, json=None, timeout=None):
        assert url == DEFAULT_URL
        assert json == {'model': 'qwen3-embed', 'input': '你好世界'}
        return FakeResponse({'data': [{'embedding': [0.1, 0.2, 0.3]}]})

    monkeypatch.setattr('rag.embed.requests.post', fake_post)
    assert embed('你好世界') == [0.1, 0.2, 0.3]


def test_embed_respects_custom_model(monkeypatch):
    """自定义模型名与 URL 生效。"""

    def fake_post(url, json=None, timeout=None):
        assert url == 'http://127.0.0.1:8099/v1/embeddings'
        assert json == {'model': 'kalm-12b', 'input': 'x'}
        return FakeResponse({'data': [{'embedding': [0.5]}]})

    monkeypatch.setattr('rag.embed.requests.post', fake_post)
    assert embed('x', url='http://127.0.0.1:8099/v1/embeddings', model='kalm-12b') == [0.5]


def test_embed_raises_on_service_error(monkeypatch):
    """服务返回 500：raise_for_status 抛 HTTPError。"""
    import requests

    def fake_post(url, json=None, timeout=None):
        return FakeResponse({}, status=500)

    monkeypatch.setattr('rag.embed.requests.post', fake_post)
    with pytest.raises(requests.HTTPError):
        embed('触发错误')
