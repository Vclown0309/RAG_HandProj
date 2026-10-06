"""生成客户端：调用 llama-server 聊天服务（默认 8080）。

坑（旧项目踩过）：Qwen3.5 的思考过程在 reasoning_content 里，
取答案只认 choices[0].message.content，别被思考过程带偏。

llama-server -m <对话模型> -c 8192 --port 8080
"""

import requests

DEFAULT_URL = 'http://127.0.0.1:8080/v1/chat/completions'
DEFAULT_MODEL = 'qwen3.5'


class GenerateError(Exception):
    pass


def chat(prompt: str, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL) -> str:
    """发一条 user 消息，返回模型 content。错误统一抛 GenerateError。"""
    try:
        resp = requests.post(
            url,
            json={'model': model, 'messages': [{'role': 'user', 'content': prompt}]},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as err:
        raise GenerateError(f'生成请求出问题了：{err}') from err
    except (KeyError, IndexError, ValueError) as err:
        raise GenerateError(f'生成响应格式异常：{err}') from err


def build_answer_prompt(query: str, hits: list[tuple[int, float, str, str]]) -> str:
    """把检索结果拼成 prompt：资料 + 问题，要求基于资料回答（RAG 生成）。

    编号用【资料N】强调；并约束引用编号与实际内容对应——
    LLM 会被块内的目录/链接文字误导（引用幻觉），指令里明说。
    """
    parts = [
        '根据以下资料回答问题，只依据资料，不要编造资料外的内容。',
        ('回答时如需引用依据，用【资料N】标注，N 必须与实际内容所在的资料编号一致；'
         '拿不准编号时不要标编号。资料里的目录、链接文字不算内容本身。'),
    ]
    for i, (_rid, _score, content, _source) in enumerate(hits, 1):
        parts.append(f'【资料{i}】\n{content}')
    parts.append(f'问题：{query}')
    return '\n\n'.join(parts)
