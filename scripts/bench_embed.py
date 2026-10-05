"""嵌入模型横向对比：固定测试集 × 同一知识库 × 各嵌入模型 → top-3 命中率。

用法（需先起对应 llama-server，见 bench_embed 顶部 MODEL_POOLING）：
    uv run python -m scripts.bench_embed

输出：bench_results/embed_bench.md（每模型：命中率 / 维度 / 体积 / 耗时）
判定：top-3 块内容包含该题锚句 = 命中（不依赖块 ID，块 ID 每次重建会变）
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import request

from rag.chunking import chunk_markdown
from rag.embed import embed

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / 'docs' / '源码导读.md', ROOT / 'docs' / 'RAG学习笔记01-模型服务与向量化输入.md', ROOT / 'docs' / '快速上手.md']
OUT = ROOT / 'bench_results'
LLAMA_SERVER = r'D:\software\llama-b11337-bin-win-cuda-12.4-x64\llama-server.exe'
PORT = 8091  # 避开用户常驻的 8081/8080，防止误连残留服务
EMBED_URL = f'http://127.0.0.1:{PORT}/v1/embeddings'

# 模型文件 → (pooling, 说明)
MODEL_POOLING: dict[str, str] = {
    r'D:\llamacpp_models\qwen3-embed\Qwen3-Embedding-0.6B-Q8_0.gguf': 'last',
    r'D:\llamacpp_models\qwen3-embed\Qwen3-Embedding-4B-Q4_K_M.gguf': 'last',
    r'D:\llamacpp_models\qwen3-embed\Qwen3-Embedding-4B-Q5_K_M.gguf': 'last',
    r'D:\llamacpp_models\qwen3-embed\Qwen3-Embedding-8B-Q5_K_M.gguf': 'last',
    r'D:\llamacpp_models\bge-m3-embed\bge-m3-q8_0.gguf': 'cls',
    r'D:\llamacpp_models\KaLM-embed-Gemma3\KaLM-Embedding-Gemma3-12B-2511.Q4_K_M.gguf': 'last',
    r'D:\llamacpp_models\KaLM-embed-Gemma3\KaLM-Embedding-Gemma3-12B-2511.Q5_K_M.gguf': 'last',
    r'D:\llamacpp_models\harrier-oss-v1-embed\harrier-oss-v1-0.6B-Q8_0.gguf': 'last',
    r'D:\llamacpp_models\harrier-oss-v1-embed\harrier-oss-v1-0.6B-BF16.gguf': 'last',
}

# 12 题：问题 → 判定锚句（答案块必须包含）
QUESTIONS: list[tuple[str, str]] = [
    ('切块为什么要设 50 字重叠？', '为什么要重叠'),
    ('检索时怎么衡量两个向量像不像？', '余弦相似度'),
    ('我不想让模型瞎编，回答前要喂它什么？', '根据以下资料回答'),
    ('向量是一串数字，数字本身没意义，那什么才有意义？', '方向才表达语义'),
    ('为什么要用余弦相似度而不是欧氏距离？', '为什么用余弦不用距离'),
    ('文档主标题为什么不能当分节点？', '不能当分节点'),
    ('嵌入服务和聊天服务各监听什么端口？', '8081'),
    ('SQLite 怎么存向量？', '向量存二进制'),
    ('启动嵌入模型服务必须加哪两个参数？', '--pooling last'),
    ('向量化用什么格式输入最佳？', 'Markdown'),
    ('没有显卡能跑这个演示台吗？', 'CPU 版'),
    ('换嵌入模型后检索结果不对怎么办？', '重建知识库'),
]


@dataclass
class ModelResult:
    name: str
    pooling: str
    dim: int
    size_gb: float
    hits: int
    total: int
    secs: float
    errors: list[str]


def wait_ready(proc: subprocess.Popen, timeout: int = 180) -> bool:
    """等服务就绪。端口被占或启动失败时子进程会退出，poll() 非 None 即报失败，
    避免误连到端口上的残留服务。"""
    url = f'http://127.0.0.1:{PORT}/health'
    for _ in range(timeout):
        if proc.poll() is not None:
            return False
        try:
            with request.urlopen(url, timeout=3):
                return proc.poll() is None
        except OSError:
            time.sleep(2)
    return False


def run_bench(model: str, pooling: str) -> ModelResult:
    """对单个嵌入模型跑一轮：起服务 → 建库 → 12 题 top-3 命中。"""
    name = Path(model).name
    start = time.time()
    log_path = OUT / f'bench_{Path(model).stem}.log'
    log_file = log_path.open('w', encoding='utf-8')
    proc = subprocess.Popen(
        [LLAMA_SERVER, '-m', model, '--embedding', '--pooling', pooling,
         '-c', '8192', '-b', '8192', '-ub', '8192', '--host', '127.0.0.1', '--port', str(PORT)],
        stdout=log_file, stderr=log_file,
    )
    errors: list[str] = []
    try:
        if not wait_ready(proc):
            log_file.flush()
            tail = log_path.read_text(encoding='utf-8', errors='ignore').splitlines()[-8:]
            errors.append('服务未就绪；日志尾：' + ' ⏎ '.join(tail))
            return ModelResult(name, pooling, 0, 0, 0, 0, time.time() - start, errors)

        # 1) 结构切块（生产同一策略）
        chunks: list[dict] = []
        for doc in DOCS:
            for block in chunk_markdown(doc.read_text(encoding='utf-8')):
                chunks.append({'content': block, 'source': doc.name})

        # 2) 全部块向量化
        vectors = [embed(c['content'], url=EMBED_URL) for c in chunks]
        dim = len(vectors[0]) if vectors else 0
        for c, v in zip(chunks, vectors):
            c['vector'] = v

        # 3) 12 题逐题检索 top-3
        hits = 0
        for q, anchor in QUESTIONS:
            qv = embed(q, url=EMBED_URL)
            scored = sorted(chunks, key=lambda c: _cosine(c['vector'], qv), reverse=True)[:3]
            if any(anchor in c['content'] for c in scored):
                hits += 1

        total = len(QUESTIONS)
        size_gb = Path(model).stat().st_size / 2**30
        return ModelResult(name, pooling, dim, size_gb, hits, total, time.time() - start, errors)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _cosine(v1: list[float], v2: list[float]) -> float:
    dot = sum(x * y for x, y in zip(v1, v2))
    n1 = sum(x * x for x in v1) ** 0.5
    n2 = sum(y * y for y in v2) ** 0.5
    return dot / (n1 * n2) if n1 and n2 else 0.0


def main() -> None:
    OUT.mkdir(exist_ok=True)
    targets = MODEL_POOLING
    is_partial = len(sys.argv) > 1  # 过滤模式：只跑匹配模型，写 *_partial 文件，不覆盖全量报告
    if is_partial:
        targets = {m: p for m, p in MODEL_POOLING.items() if sys.argv[1] in m}
        if not targets:
            print(f'无匹配模型：{sys.argv[1]}')
            return
    results: list[ModelResult] = []
    for model, pooling in targets.items():
        r = run_bench(model, pooling)
        results.append(r)
        status = 'OK' if not r.errors else 'FAIL'
        print(f'[{status}] {Path(model).name}  {r.hits}/{r.total}  dim={r.dim}  {r.secs:.0f}s', flush=True)
    results.sort(key=lambda r: (r.hits / r.total if r.total else 0.0), reverse=True)

    lines = ['# 嵌入模型横向对比（top-3 命中率）\n', f'语料：{", ".join(d.name for d in DOCS)}\n', f'测试集：{len(QUESTIONS)} 题 · 判定：top-3 含锚句\n', '']
    lines.append('| 排名 | 模型 | 命中 | 维度 | 体积 | 耗时 | pooling |')
    lines.append('|---|---|---|---|---|---|---|')
    for i, r in enumerate(results, 1):
        flag = ' ✅' if r.errors else ''
        lines.append(f'| {i} | {r.name} | {r.hits}/{r.total} | {r.dim} | {r.size_gb:.2f}GB | {r.secs:.0f}s | {r.pooling}{flag} |')
    lines.append('')
    for r in results:
        if r.errors:
            lines.append(f'- ⚠️ {r.name}：{"；".join(r.errors)}')

    text = '\n'.join(lines)
    out_md = OUT / ('embed_bench_partial.md' if is_partial else 'embed_bench.md')
    out_raw = OUT / ('embed_bench_raw_partial.json' if is_partial else 'embed_bench_raw.json')
    out_md.write_text(text, encoding='utf-8')
    print(text)
    with open(out_raw, 'w', encoding='utf-8') as f:
        json.dump([r.__dict__ for r in results], f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
