from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from rag.chunking import chunk_markdown
from rag.embed import EmbeddingError, embed
from rag.generate import GenerateError, build_answer_prompt, chat
from rag.normalize import normalize_md
from rag.store import (
    DB_PATH,
    add_history,
    delete_source,
    get_history,
    init_db,
    list_sources,
    search,
    store_chunks,
)

# 数据库路径：显式传参，测试可替换成临时库，避免污染真实 kb.db
DB_LOCAL = DB_PATH


@asynccontextmanager
async def lifespan(_: FastAPI):
    """服务启动时确保表存在（幂等建表）。"""
    init_db(path=DB_LOCAL)
    yield


app = FastAPI(lifespan=lifespan)

# 数据库路径：显式传参，测试可替换成临时库，避免污染真实 kb.db
DB_LOCAL = DB_PATH


class AskRequest(BaseModel):
    query: str


@app.get('/')
async def root():
    return {'service': 'RAG 问答', 'status': 'ok'}


@app.post('/ask')
def ask(req: AskRequest):
    """RAG 全链路：向量化问题 → 检索 top-k → 拼 prompt → 生成答案 → 落历史。"""
    try:
        query_vector = embed(req.query)
    except EmbeddingError as err:
        raise HTTPException(status_code=502, detail=str(err)) from err

    hits = search(query_vector, top_k=3, path=DB_LOCAL)
    if not hits:
        answer = '知识库暂无相关内容，先入库再问。'
    else:
        prompt = build_answer_prompt(req.query, hits)
        try:
            answer = chat(prompt)
        except GenerateError as err:
            raise HTTPException(status_code=502, detail=str(err)) from err

    add_history(req.query, answer, path=DB_LOCAL)
    return {
        'query': req.query,
        'answer': answer,
        'sources': [{'content': content, 'source': source} for _rid, _score, content, source in hits],
    }


@app.get('/history')
def history(limit: int = 20):
    """最近问答记录，最新在前。"""
    return {
        'history': [
            {'id': rid, 'query': query, 'answer': answer, 'created_at': created_at}
            for rid, query, answer, created_at in get_history(limit, path=DB_LOCAL)
        ]
    }


@app.get('/library')
def library():
    """知识库文档清单：(source, 块数)。"""
    return {
        'sources': [{'source': source, 'chunks': n} for source, n in list_sources(path=DB_LOCAL)]
    }


@app.post('/library')
async def add_doc(file: Annotated[UploadFile, File()]):
    """上传 txt/md，归一化 → 切块 → 向量化 → 追加入库（幂等）。"""
    source = Path(file.filename or 'untitled.txt').name
    try:
        raw = (await file.read()).decode('utf-8')
    except UnicodeDecodeError as err:
        raise HTTPException(status_code=400, detail='仅支持 UTF-8 编码的文本文件') from err

    chunks = list(chunk_markdown(normalize_md(raw)))
    if not chunks:
        raise HTTPException(status_code=400, detail='文件内容为空，无可入库块')

    rows = []
    for chunk in chunks:
        try:
            vec = embed(chunk)
        except EmbeddingError as err:
            raise HTTPException(status_code=502, detail=str(err)) from err
        rows.append((chunk, vec, source))

    n = store_chunks(rows, path=DB_LOCAL)
    return {'source': source, 'added': len(chunks), 'updated_or_added': n}


@app.delete('/library/{source}')
def del_doc(source: str):
    """删除某文档的全部块。"""
    deleted = delete_source(source, path=DB_LOCAL)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f'知识库中未找到文档 {source}')
    return {'source': source, 'deleted': deleted}
