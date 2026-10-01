"""生成客户端单元测试：mock 掉 requests，不依赖真实服务。"""

import pytest

from rag.generate import DEFAULT_URL, GenerateError, build_answer_prompt, chat


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


def test_chat_returns_content(monkeypatch):
    """正常请求：返回 choices[0].message.content。"""

    def fake_post(url, json=None, timeout=None):
        assert url == DEFAULT_URL
        assert json['messages'] == [{'role': 'user', 'content': '你好'}]
        return FakeResponse({
            'choices': [{'message': {'content': '你好，我是助手'}}]
        })

    monkeypatch.setattr('rag.generate.requests.post', fake_post)
    assert chat('你好') == '你好，我是助手'


def test_chat_ignores_reasoning_content(monkeypatch):
    """只认 content，不认 reasoning_content（Qwen3.5 坑）。"""

    def fake_post(url, json=None, timeout=None):
        return FakeResponse({
            'choices': [{
                'message': {
                    'content': '正确答案',
                    'reasoning_content': '一大坨英文思考过程……',
                }
            }]
        })

    monkeypatch.setattr('rag.generate.requests.post', fake_post)
    assert chat('问题') == '正确答案'


def test_chat_raises_on_service_error(monkeypatch):
    """服务 500：包成 GenerateError 并保留异常链。"""

    def fake_post(url, json=None, timeout=None):
        return FakeResponse({}, status=500)

    monkeypatch.setattr('rag.generate.requests.post', fake_post)
    with pytest.raises(GenerateError, match='生成请求出问题了'):
        chat('触发错误')


def test_build_answer_prompt_assembles_sources():
    """prompt 拼装：资料块在前、问题在后、禁止编造指令在首。"""
    hits = [(1, 0.9, '资料内容A', 'a.md#0'), (2, 0.8, '资料内容B', 'b.md#1')]
    prompt = build_answer_prompt('什么是 RAG？', hits)
    assert '[资料1] 资料内容A' in prompt
    assert '[资料2] 资料内容B' in prompt
    assert '问题：什么是 RAG？' in prompt
    assert prompt.startswith('根据以下资料回答问题')


def test_build_answer_prompt_empty_hits():
    """空命中也拼出可用 prompt（无资料块）。"""
    prompt = build_answer_prompt('问题', [])
    assert '问题：问题' in prompt
    assert '[资料1]' not in prompt
