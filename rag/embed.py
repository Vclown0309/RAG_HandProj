"""embed 客户端：调用 llama-server 嵌入服务（默认 8081）。

服务端：llama-server -m Qwen3-Embedding-8B-Q5_K_M.gguf --embedding
        --pooling last -c 8192 -b 8192 -ub 8192 --host 127.0.0.1 --port 8081
实测：8B 输出 4096 维，4B 输出 2560 维。换模型后维度变化需重建知识库。
"""

import requests

DEFAULT_URL = 'http://127.0.0.1:8081/v1/embeddings'
DEFAULT_MODEL = 'qwen3-embed'

class EmbeddingError(Exception):
    pass

def embed(text: str, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL) -> list[float]:
    """把文本向量化，返回浮点向量列表。错误统一抛 EmbeddingError。"""
    try:
        resp = requests.post(url, json={'model': model, 'input': text}, timeout=30)
        # 状态码是4XX，5XX都会直接报错，不会继续执行下面的代码
        resp.raise_for_status()
        return resp.json()['data'][0]['embedding']
    except requests.exceptions.RequestException as err:
        # HTTPError、Timeout、ConnectionError 都是他的子类，一个 except 全兜住
        raise EmbeddingError(f"向量化请求出问题了：{err}") from err
    except (KeyError, IndexError, ValueError) as err:
        # 服务返回 200 但结构不对：缺 data / data 为空 / 不是合法 JSON
        raise EmbeddingError(f'向量化响应格式异常：{err}') from err
