"""embed 客户端：调用 llama-server 嵌入服务（默认 8081）。

服务端：llama-server -m Qwen3-Embedding-8B-Q5_K_M.gguf --embedding
        --pooling last -c 8192 -b 8192 -ub 8192 --host 127.0.0.1 --port 8081
实测：8B 输出 4096 维，4B 输出 2560 维。换模型后维度变化需重建知识库。
"""

import requests

DEFAULT_URL = 'http://127.0.0.1:8081/v1/embeddings'
DEFAULT_MODEL = 'qwen3-embed'


def embed(text: str, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL) -> list[float]:
    """把文本向量化，返回浮点向量列表。"""
    resp = requests.post(url, json={'model': model, 'input': text}, timeout=30)
    resp.raise_for_status()
    return resp.json()['data'][0]['embedding']
