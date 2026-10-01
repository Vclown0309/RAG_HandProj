from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.embed import EmbeddingError, embed
from rag.generate import GenerateError, build_answer_prompt, chat
from rag.store import search

app = FastAPI()


class AskRequest(BaseModel):
    query: str


@app.get('/')
async def root():
    return {'service': 'RAG 问答', 'status': 'ok'}


@app.post('/ask')
def ask(req: AskRequest):
    """RAG 全链路：向量化问题 → 检索 top-k → 拼 prompt → 生成答案。"""
    try:
        query_vector = embed(req.query)
    except EmbeddingError as err:
        raise HTTPException(status_code=502, detail=str(err)) from err

    hits = search(query_vector, top_k=3)
    if not hits:
        return {'query': req.query, 'answer': '知识库暂无相关内容，先入库再问。', 'sources': []}

    prompt = build_answer_prompt(req.query, hits)
    try:
        answer = chat(prompt)
    except GenerateError as err:
        raise HTTPException(status_code=502, detail=str(err)) from err

    return {
        'query': req.query,
        'answer': answer,
        'sources': [{'content': content, 'source': source} for _rid, _score, content, source in hits],
    }
