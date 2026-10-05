"""聊天模型交叉矩阵：两档嵌入 × 五档聊天 = 10 组。

阶段 A（检索排名稳定性）：同嵌入档内换聊天模型，top-3 排名应完全一致
    → 证明检索结果由嵌入模型决定，聊天模型不"代考"
    （大聊天模型若靠常识补答案，检索波动会被掩盖——这是本矩阵要消除的黑箱）。
阶段 B（生成质量代理指标）：
    - 可追溯：答案含【资料N】且 N 在合法范围（引用幻觉检测的弱判据）
    - 拒答：3 道库外题是否诚实说"资料里没有"（RAG 防幻觉核心）
    - 速度：每题生成耗时 + completion token/s（数据驱动，不凭感觉）
    - 忠实度弱判据：答案中的数字是否都能在参考源文本里找到
    + 人工抽样复核兜底（每模型 3 题，报告标注"抽样非全量"）

用法：uv run python -m scripts.bench_chat
输出：bench_results/chat_matrix.md + bench_results/chat_matrix_raw.json
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib import request

import requests

from rag.chunking import chunk_markdown
from rag.embed import embed
from rag.generate import build_answer_prompt
from scripts.bench_embed import DOCS, MODEL_POOLING, QUESTIONS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'bench_results'
LLAMA_SERVER = r'D:\software\llama-b11337-bin-win-cuda-12.4-x64\llama-server.exe'
EMBED_PORT = 8093  # 避开常驻 8081/8080/8000 与嵌入对比 8091
CHAT_PORT = 8092
EMBED_URL = f'http://127.0.0.1:{EMBED_PORT}/v1/embeddings'
CHAT_URL = f'http://127.0.0.1:{CHAT_PORT}/v1/chat/completions'

# 嵌入模型：默认全 9 档（与嵌入对比同一清单，pooling 官方实证）。
# 缺省跑 9×5=45 组太久；用 CLI 过滤收敛：如 9 嵌入 × 单聊天 `bench_chat <聊天关键词>`，
# 或单组 `bench_chat <聊天关键词> <嵌入关键词>`。
EMBED_MODELS: dict[str, str] = MODEL_POOLING

# 五档聊天（用户拍板档位：1.7B / 4B / 4B-3.5 / 9B / 27B）
CHAT_MODELS: list[str] = [
    r'D:\llamacpp_models\qwen3\unsloth\Qwen3-1.7B-Q4_K_M.gguf',
    r'D:\llamacpp_models\qwen3\Qwen3-4B-Q4_K_M.gguf',
    r'D:\llamacpp_models\qwen3.5\Qwen3.5-4B-Q5_K_M.gguf',
    r'D:\llamacpp_models\qwen3.5\Qwen3.5-9B.Q5_K_M.gguf',
    r'D:\llamacpp_models\Qwen3.8-27B\Qwen3.8-27B-UD-Q5_K_M.gguf',
]

# 3 道库外题：知识库没有答案，测拒答/幻觉（搜到的最相似块与问题无关）
OUT_OF_LIBRARY: list[str] = [
    '今天成都的天气怎么样？',
    '介绍一下量子计算的基本原理',
    '怎么给电脑安装 Linux 双系统？',
]
ALL_QUESTIONS = [q for q, _anchor in QUESTIONS] + OUT_OF_LIBRARY

REFUSE_MARKS = ('没有', '未找到', '资料中', '无法回答', '不确定', '没有提到', '不知道', '未能', '未提供', '资料里没有', '资料中没有')


@dataclass
class GroupResult:
    embed: str
    chat: str
    ranks: list[tuple[tuple[str, ...], ...]] = field(default_factory=list)  # 12 题 top-3 content
    answers: list[dict] = field(default_factory=list)  # 15 题生成记录
    errors: list[str] = field(default_factory=list)


def wait_ready(proc: subprocess.Popen, port: int, timeout: int = 180) -> bool:
    """等服务就绪。子进程退出（端口被占/启动失败）即判失败，避免误连残留服务。"""
    url = f'http://127.0.0.1:{port}/health'
    for _ in range(timeout):
        if proc.poll() is not None:
            return False
        try:
            with request.urlopen(url, timeout=3):
                return proc.poll() is None
        except OSError:
            time.sleep(2)
    return False


def _cosine(v1: list[float], v2: list[float]) -> float:
    dot = sum(x * y for x, y in zip(v1, v2))
    n1 = sum(x * x for x in v1) ** 0.5
    n2 = sum(y * y for y in v2) ** 0.5
    return dot / (n1 * n2) if n1 and n2 else 0.0


def _ctx_args(chat_path: str) -> list[str]:
    """显存策略：27B(18.4GB) 与 8B 嵌入(5.05GB) 同载时小上下文防 OOM。"""
    if '27b' in Path(chat_path).name.lower():
        return ['-c', '4096', '-b', '2048', '-ub', '2048']
    return ['-c', '8192', '-b', '4096', '-ub', '4096']


def chat_with_meta(prompt: str) -> tuple[str, dict, float]:
    """生成一条答案，返回 (content, usage, 耗时秒)。超时/失败抛错由调用方处理。"""
    t0 = time.time()
    resp = requests.post(CHAT_URL, json={'model': 'chat', 'messages': [{'role': 'user', 'content': prompt}]}, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    content = data['choices'][0]['message']['content']
    return content, data.get('usage', {}), time.time() - t0


def _traceable(answer: str) -> tuple[bool, list[int]]:
    """答案是否含【资料N】且 N 合法（1..3）。"""
    refs = sorted({int(r) for r in re.findall(r'【资料(\d+)】', answer) if 1 <= int(r) <= 3})
    return bool(refs), refs


def _refused(answer: str) -> bool:
    return any(m in answer for m in REFUSE_MARKS)


def _digit_faithful(answer: str, src_text: str) -> tuple[int, int]:
    """答案中能在参考源文本里找到的数字数 / 总数字数（弱判据）。"""
    nums = re.findall(r'\d+(?:\.\d+)?', answer)
    if not nums:
        return 0, 0
    hit = sum(1 for n in nums if n in src_text)
    return hit, len(nums)


def _retrieve_top3(chunks: list[dict], qv: list[float]) -> list[dict]:
    return sorted(chunks, key=lambda c: _cosine(c['vector'], qv), reverse=True)[:3]


def run_group(embed_path: str, pooling: str, chat_path: str) -> GroupResult:
    """单组（一个嵌入 × 一个聊天）：起双服务 → 建库 → 12 题检索排名 → 15 题生成。"""
    g = GroupResult(embed=Path(embed_path).name, chat=Path(chat_path).name)
    elog = (OUT / f'chat_embed_{Path(embed_path).stem}_{Path(chat_path).stem}.log').open('w', encoding='utf-8')
    clog = (OUT / f'chat_{Path(embed_path).stem}_{Path(chat_path).stem}.log').open('w', encoding='utf-8')
    ep = subprocess.Popen(
        [LLAMA_SERVER, '-m', embed_path, '--embedding', '--pooling', pooling,
         '-c', '8192', '-b', '8192', '-ub', '8192', '--host', '127.0.0.1', '--port', str(EMBED_PORT)],
        stdout=elog, stderr=elog,
    )
    cp = subprocess.Popen(
        [LLAMA_SERVER, '-m', chat_path, *_ctx_args(chat_path), '--host', '127.0.0.1', '--port', str(CHAT_PORT)],
        stdout=clog, stderr=clog,
    )
    try:
        if not wait_ready(ep, EMBED_PORT) or not wait_ready(cp, CHAT_PORT):
            for log_path, tag in ((OUT / f'chat_embed_{Path(embed_path).stem}_{Path(chat_path).stem}.log', 'embed'), (OUT / f'chat_{Path(embed_path).stem}_{Path(chat_path).stem}.log', 'chat')):
                tail = log_path.read_text(encoding='utf-8', errors='ignore').splitlines()[-5:]
                g.errors.append(f'{tag} 服务未就绪；日志尾：' + ' ⏎ '.join(tail))
            return g

        # 建库（结构切块 + 全量向量化，内存直比不落库）
        chunks: list[dict] = []
        for doc in DOCS:
            for block in chunk_markdown(doc.read_text(encoding='utf-8')):
                chunks.append({'content': block, 'source': doc.name})
        for c in chunks:
            c['vector'] = embed(c['content'], url=EMBED_URL)

        # 阶段 A：12 题检索排名（top-3 的 content 元组，供同档内比对）
        for q, _anchor in QUESTIONS:
            qv = embed(q, url=EMBED_URL)
            g.ranks.append(tuple(c['content'] for c in _retrieve_top3(chunks, qv)))

        # 阶段 B：15 题生成（12 库内 + 3 库外）
        for q in ALL_QUESTIONS:
            qv = embed(q, url=EMBED_URL)
            hits = [(i, _cosine(c['vector'], qv), c['content'], c['source']) for i, c in enumerate(_retrieve_top3(chunks, qv))]
            prompt = build_answer_prompt(q, hits)
            try:
                content, usage, secs = chat_with_meta(prompt)
            except Exception as err:  # noqa: BLE001 - 单题失败不中断整组
                content, usage, secs = f'[生成失败] {err}', {}, 0.0
            src_text = '\n'.join(h[2] for h in hits)
            faithful = _digit_faithful(content, src_text)
            g.answers.append({
                'q': q,
                'answer': content,
                'secs': round(secs, 2),
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'traceable': _traceable(content)[0],
                'refs': _traceable(content)[1],
                'refused': _refused(content),
                'faithful': faithful,
                'sources': [h[3] for h in hits],
            })
        return g
    finally:
        for p in (ep, cp):
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


def _stability_section(groups: list[GroupResult]) -> list[str]:
    lines = ['## 阶段 A：检索排名稳定性（同嵌入档内 top-3 一致性）', '',
             '| 嵌入档 | 聊天模型 | 与档内基准一致(12题) |', '|---|---|---|']
    for embed_key in (Path(p).name for p in EMBED_MODELS):
        subset = [g for g in groups if g.embed == embed_key]
        if not subset:
            continue
        base = subset[0].ranks
        for g in subset:
            same = sum(1 for a, b in zip(base, g.ranks) if a == b)
            lines.append(f'| {embed_key} | {g.chat} | {same}/12 |')
        lines.append('')
    lines.append('判读：同档内应全为 12/12——检索由嵌入决定；出现波动说明上下文污染或流程 bug。')
    lines.append('')
    return lines


def _gen_section(groups: list[GroupResult]) -> list[str]:
    path_by_name = {Path(p).name: p for p in CHAT_MODELS}
    lines = ['## 阶段 B：生成质量代理指标', '',
             '| 聊天模型 | 体积 | 可追溯(12库内题) | 拒答(3库外题) | 平均耗时 | token/s | 忠实度(弱判据) |',
             '|---|---|---|---|---|---|---|']
    for g in groups:
        if g.embed != Path(next(iter(EMBED_MODELS))).name:
            continue  # 生成指标只展示 4B 嵌入档（口径与边界声明写进报告）
        if g.errors:
            lines.append(f'| {g.chat} | 启动失败 | - | - | - | - | - |')
            continue
        size_gb = Path(path_by_name[g.chat]).stat().st_size / 2**30
        trace = sum(1 for a in g.answers[:12] if a['traceable'])
        refuse = sum(1 for a in g.answers[12:] if a['refused'])
        secs = [a['secs'] for a in g.answers if a['secs'] > 0]
        gen_secs = sum(a['secs'] for a in g.answers if a['secs'] > 0)
        tok = sum(a['completion_tokens'] for a in g.answers) / gen_secs if gen_secs else 0.0
        hit_n = sum(a['faithful'][0] for a in g.answers)
        tot_n = sum(a['faithful'][1] for a in g.answers)
        faithful = hit_n / tot_n if tot_n else 0.0
        avg_secs = f'{sum(secs) / len(secs):.1f}s' if secs else '-'
        lines.append(f'| {g.chat} | {size_gb:.2f}GB | {trace}/12 | {refuse}/3 | {avg_secs} | {tok:.1f} | {faithful:.0%} |')
    lines.append('')
    lines.append('忠实度为弱判据（答案数字能否在参考源找到），盲区由人工抽样兜底（见下）。')
    lines.append('')
    return lines


def _human_sample_section(groups: list[GroupResult]) -> list[str]:
    lines = ['## 人工抽样复核表（待填：每模型抽 3 题，判断忠实 / 轻微扩展 / 幻觉）', '',
             '| 聊天模型 | 题号 | 模型答案摘录 | 人工判定 |', '|---|---|---|---|']
    for g in groups:
        if g.embed != Path(next(iter(EMBED_MODELS))).name:
            continue
        for idx in (0, 4, 12):  # 库内首题/库内中段/库外首题
            if idx < len(g.answers):
                a = g.answers[idx]
                excerpt = a['answer'][:60].replace('\n', ' ')
                lines.append(f'| {g.chat} | Q{idx + 1} | {excerpt} | |')
    lines.append('')
    return lines


def _dump_raw(groups: list[GroupResult]) -> None:
    with open(OUT / 'chat_matrix_raw.json', 'w', encoding='utf-8') as f:
        json.dump([{'embed': g.embed, 'chat': g.chat, 'errors': g.errors, 'ranks': g.ranks, 'answers': g.answers} for g in groups], f, ensure_ascii=False, indent=2)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    # 过滤：python -m scripts.bench_chat [聊天关键词] [嵌入关键词]（如补跑单组用）
    chat_filter = sys.argv[1] if len(sys.argv) > 1 else None
    embed_filter = sys.argv[2] if len(sys.argv) > 2 else None
    groups: list[GroupResult] = []
    for embed_path, pooling in EMBED_MODELS.items():
        if embed_filter and embed_filter not in Path(embed_path).name:
            continue
        for chat_path in CHAT_MODELS:
            if chat_filter and chat_filter not in Path(chat_path).name:
                continue
            g = run_group(embed_path, pooling, chat_path)
            groups.append(g)
            status = 'OK' if not g.errors else 'FAIL'
            trace = sum(1 for a in g.answers[:12] if a['traceable']) if g.answers else 0
            print(f'[{status}] {g.chat}  ranks={len(g.ranks)}  trace={trace}/12  err={len(g.errors)}', flush=True)
            _dump_raw(groups)  # 即时落盘：中途崩溃也不丢已完成组

    lines = ['# 聊天模型交叉矩阵（两档嵌入 × 五档聊天 = 10 组）', '',
             f'语料：{", ".join(d.name for d in DOCS)}', f'库内题 {len(QUESTIONS)} · 库外题 {len(OUT_OF_LIBRARY)}', '']
    lines += _stability_section(groups)
    lines += _gen_section(groups)
    lines += _human_sample_section(groups)
    lines += ['## 边界声明', '',
              '结论基于本机实测：Windows 11 / i9-14900K / 64GB / RTX 4090 D 24GB（CUDA 12.4 llama.cpp b11337）。',
              '未测 ROCm（AMD Linux 独显）与 arm CPU；绝对速度按消费硬件折算，相对排名与硬件无关。',
              '生成侧结论 = 代理指标 + 人工抽样，只能区分明显好坏，不能区分神仙打架。', '']

    text = '\n'.join(lines)
    OUT.joinpath('chat_matrix.md').write_text(text, encoding='utf-8')
    print(text)
    _dump_raw(groups)


if __name__ == '__main__':
    main()
