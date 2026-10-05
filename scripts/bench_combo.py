"""聊天矩阵综合命中分析：从 chat_matrix_raw.json 重算，不重跑模型。

回答"命中率到底是纯嵌入匹配还是配合聊天之后的"：
- 检索命中：锚句出现在 top-3 检索块里（纯嵌入质量，= 嵌入对比口径）
- 生成命中：锚句出现在模型最终答案里（端到端，含生成环节）
- 综合命中：同一题上二者同时命中（题级 AND）

锚句是弱判据（长锚句模型可能改写而不含原词），供横向排序，人工抽样兜底。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'bench_results' / 'chat_matrix_raw.json'
BACKUP = ROOT / 'bench_results' / 'chat_matrix_raw_2embed_backup.json'
OUT = ROOT / 'bench_results' / 'chat_combo.md'

# 与 bench_embed.QUESTIONS 同源（12 库内题；3 库外题无锚句，不参与）
# 锚句取核心词（空格分隔 = AND）：模型会改写问法（如"为什么要重叠"答成"防止一句话被切断"），
# 整句子串会漏判；核心词命中更接近真实"答对关键点"。
QUESTIONS: list[tuple[str, str]] = [
    ('切块为什么要设 50 字重叠？', '重叠'),
    ('检索时怎么衡量两个向量像不像？', '余弦相似度'),
    ('我不想让模型瞎编，回答前要喂它什么？', '资料'),
    ('向量是一串数字，数字本身没意义，那什么才有意义？', '方向'),
    ('为什么要用余弦相似度而不是欧氏距离？', '余弦'),
    ('文档主标题为什么不能当分节点？', '分节点'),
    ('嵌入服务和聊天服务各监听什么端口？', '8081'),
    ('SQLite 怎么存向量？', '二进制'),
    ('启动嵌入模型服务必须加哪两个参数？', 'pooling'),
    ('向量化用什么格式输入最佳？', 'Markdown'),
    ('没有显卡能跑这个演示台吗？', 'CPU'),
    ('换嵌入模型后检索结果不对怎么办？', '重建'),
]
ANCHORS = [anchor for _, anchor in QUESTIONS]


def norm(s: str) -> str:
    return re.sub(r'\s+', '', s)


def hit_anchor(text: str, anchor: str) -> bool:
    """核心词 AND 判定：锚句每个词（去空白）都出现在文本中。"""
    t = norm(text)
    words = [w for w in anchor.split() if w]
    return bool(words) and all(norm(w) in t for w in words)


def load_groups() -> list[dict]:
    """读新 raw，并合并旧备份（同 embed×chat 以新为准）——兼容分批补测。"""
    if not RAW.exists():
        print(f'缺 raw：{RAW}')
        sys.exit(1)
    groups = json.loads(RAW.read_text(encoding='utf-8'))
    if BACKUP.exists():
        old = json.loads(BACKUP.read_text(encoding='utf-8'))
        keys = {(g['embed'], g['chat']) for g in groups}
        groups += [g for g in old if (g['embed'], g['chat']) not in keys]
    return groups


def main() -> None:
    groups = load_groups()
    if len(groups) < 10:
        print(f'合并后仅 {len(groups)} 组，期望 ≥10（可能被过滤跑覆盖，重跑全量）')
        sys.exit(1)

    rows: list[dict[str, Any]] = []
    for g in groups:
        ranks = g['ranks']  # 12 题 × top-3 content
        answers = g['answers']  # 15 题（前 12 库内 + 后 3 库外）
        if len(ranks) != 12 or len(answers) < 12:
            print(f"跳过异常组：{g['embed']} × {g['chat']}（ranks={len(ranks)}, answers={len(answers)}）")
            continue
        ret_hit = gen_hit = combo_hit = 0
        secs = tokens = 0.0
        misses: list[str] = []
        for i, anchor in enumerate(ANCHORS):
            ret_ok = any(hit_anchor(c, anchor) for c in ranks[i])
            gen_ok = hit_anchor(answers[i]['answer'], anchor)
            if ret_ok:
                ret_hit += 1
            if gen_ok:
                gen_hit += 1
            if ret_ok and gen_ok:
                combo_hit += 1
            if not gen_ok:
                misses.append(f'Q{i + 1}({anchor})')
        # 速度/速率：15 题（含库外）耗时与 token 产出；排除失败题（secs=0）
        ok_answers = [a for a in answers if a.get('secs', 0) > 0]
        if ok_answers:
            secs = sum(a['secs'] for a in ok_answers) / len(ok_answers)
            total_secs = sum(a['secs'] for a in ok_answers)
            total_tokens = sum(a.get('completion_tokens', 0) for a in ok_answers)
            tokens = total_tokens / total_secs if total_secs else 0.0
        rows.append({
            'embed': g['embed'], 'chat': g['chat'],
            'ret': ret_hit, 'gen': gen_hit, 'combo': combo_hit,
            'secs': secs, 'tps': tokens,
            'miss': '、'.join(misses),
        })

    rows.sort(key=lambda r: (r['combo'], r['gen'], -r['secs']), reverse=True)  # type: ignore[arg-type,operator]

    lines = ['# 聊天矩阵综合命中（检索 × 生成 双层 + 速度，12 库内题）\n',
             '口径：检索命中=锚句在 top-3 检索块内（纯嵌入）；生成命中=锚句在最终答案内（端到端）；综合=同题双命中。',
             '锚句取核心词（AND），模型改写问法不漏判；平均耗时/token 速率按 15 题成功生成统计。\n',
             '| 排名 | 嵌入 | 聊天 | 检索 | 生成 | 综合 | 平均耗时 | token/s | 生成未命中题 |',
             '|---|---|---|---|---|---|---|---|---|']
    for i, r in enumerate(rows, 1):
        embed_short = str(r['embed']).replace('Qwen3-Embedding-', '').replace('.gguf', '')
        chat_short = str(r['chat']).replace('Qwen3', 'Q').replace('Q3.5-', '3.5-').replace('.gguf', '')
        secs_txt = f"{r['secs']:.1f}s" if r['secs'] else '-'
        tps_txt = f"{r['tps']:.0f}" if r['tps'] else '-'
        lines.append(f"| {i} | {embed_short} | {chat_short} | {r['ret']}/12 | {r['gen']}/12 | **{r['combo']}/12** | {secs_txt} | {tps_txt} | {r['miss']} |")

    text = '\n'.join(lines) + '\n'
    OUT.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
