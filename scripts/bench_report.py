"""模型选型总报告：嵌入 9 模型 × 聊天 5 模型 实测汇总。

数据源：
- bench_results/embed_bench_raw.json        9 嵌入纯检索对比
- bench_results/chat_matrix_raw.json        新补测（9 嵌入 × Qwen3-4B-Q4 聊天）
- bench_results/chat_matrix_raw_2embed_backup.json  旧 10 组（2 嵌入 × 5 聊天）

输出 docs/模型选型报告.md，供 README / 整合包 / 开发者文档直接引用选型理由。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.bench_chat import QUESTIONS
from scripts.bench_combo import ANCHORS, hit_anchor, load_groups

ROOT = Path(__file__).resolve().parent.parent
EMBED_RAW = ROOT / 'bench_results' / 'embed_bench_raw.json'
OUT = ROOT / 'docs' / '模型选型报告.md'


def size_gb(path: str) -> float:
    p = Path(path)
    return round(p.stat().st_size / 1024**3, 2) if p.exists() else 0.0


def embed_rows() -> list[dict[str, Any]]:
    items = json.loads(EMBED_RAW.read_text(encoding='utf-8'))
    return sorted(items, key=lambda r: (-r['hits'], r['size_gb']))  # type: ignore[arg-type]


def chat_rows(groups: list[dict]) -> list[dict[str, Any]]:
    """聊天维度 10 组：4B-Q4 / 8B-Q5 嵌入 × 5 聊天。"""
    rows: list[dict[str, Any]] = []
    for g in groups:
        if 'Qwen3-Embedding-4B-Q4' not in g['embed'] and 'Qwen3-Embedding-8B-Q5' not in g['embed']:
            continue
        ans = g['answers']
        trace = sum(1 for a in ans[: len(QUESTIONS)] if a.get('traceable'))
        refuse = sum(1 for a in ans[len(QUESTIONS) :] if a.get('refused'))
        hit_n = sum(a['faithful'][0] for a in ans)
        tot_n = sum(a['faithful'][1] for a in ans)
        ok = [a for a in ans if a.get('secs', 0) > 0]
        secs = sum(a['secs'] for a in ok) / len(ok) if ok else 0.0
        total_secs = sum(a['secs'] for a in ok)
        tokens = sum(a.get('completion_tokens', 0) for a in ok) / total_secs if total_secs else 0.0
        rows.append({
            'embed': g['embed'], 'chat': g['chat'],
            'trace': trace, 'refuse': refuse,
            'faithful': hit_n / tot_n if tot_n else 0.0,
            'secs': secs, 'tps': tokens,
        })
    return rows


def combo_rows(groups: list[dict]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for g in groups:
        ranks, ans = g['ranks'], g['answers']
        if len(ranks) != 12 or len(ans) < 12:
            continue
        ret = gen = combo = 0
        for i, anchor in enumerate(ANCHORS):
            ret_ok = any(hit_anchor(c, anchor) for c in ranks[i])
            gen_ok = hit_anchor(ans[i]['answer'], anchor)
            ret += ret_ok
            gen += gen_ok
            combo += ret_ok and gen_ok
        ok = [a for a in ans if a.get('secs', 0) > 0]
        secs = sum(a['secs'] for a in ok) / len(ok) if ok else 0.0
        rows.append({'embed': g['embed'], 'chat': g['chat'],
                     'ret': ret, 'gen': gen, 'combo': combo, 'secs': secs})
    return rows


def short_name(path: str) -> str:
    return path.replace('Qwen3-Embedding-', '').replace('KaLM-Embedding-', 'KaLM-').replace('.gguf', '')


def chat_short(path: str) -> str:
    return path.replace('Qwen3', 'Q').replace('Q3.5-', '3.5-').replace('.gguf', '')


def main() -> None:
    groups = load_groups()
    lines: list[str] = [
        '# 模型选型与横向对比报告（实测）',
        '',
        '> 结论基于本机实测：Windows 11 / i9-14900K（24C/32T）/ 64GB / RTX 4090 D 24GB，',
        '> llama.cpp b11337（CUDA 12.4 / CPU / Vulkan 三后端已备）。',
        '> **未测 ROCm（AMD Linux 独显）与 arm CPU**；相对排名与硬件无关，绝对速度按消费硬件折算。',
        '> 语料：源码导读.md / RAG学习笔记01 / 快速上手.md；库内题 12 + 库外题 3。',
        '',
        '---',
        '',
        '## 一、嵌入模型纯检索对比（9 模型 × 12 题）',
        '',
        '口径：结构切块 → 全量向量化 → 余弦 top-3，锚句在 top-3 内即命中。只换嵌入模型。',
        '',
        '| 排名 | 嵌入模型 | 池化方式 | 向量维度 | 体积 | 检索命中 | 单组耗时 |',
        '|---|---|---|---|---|---|---|',
    ]
    for i, r in enumerate(embed_rows(), 1):
        lines.append(
            f"| {i} | {short_name(str(r['name']))} | {r['pooling']} | {r['dim']} | {r['size_gb']:.2f}GB | "
            f"**{r['hits']}/{r['total']}** | {r['secs']:.1f}s |"
        )
    lines += ['',
              '要点：4B-Q4/Q5 并列 10/12 最优；8B 与 0.6B、harrier 同为 9/12；KaLM-12B 最低（6~7/12）且体积 3 倍。',
              '',
              '---',
              '',
              '## 二、聊天模型生成质量（4B-Q4 / 8B-Q5 嵌入 × 5 聊天 = 10 组）',
              '',
              '口径：可追溯=12 库内题答案带【资料N】引用；拒答=3 库外题正确拒答；忠实度=答案数字能在参考源找到的比例（弱判据）。',
              '',
              '| 嵌入 | 聊天 | 可追溯 | 拒答 | 忠实度 | 平均耗时 | token/秒 |',
              '|---|---|---|---|---|---|---|']
    for r in sorted(chat_rows(groups), key=lambda r: (-r['trace'], r['secs'])):  # type: ignore[arg-type]
        lines.append(
            f"| {short_name(str(r['embed']))} | {chat_short(str(r['chat']))} | {r['trace']}/12 | {r['refuse']}/3 | "
            f"{r['faithful']:.0%} | {r['secs']:.1f}s | {r['tps']:.0f} |"
        )
    lines += ['',
              '要点：可追溯满分只有 4B-Q4×{4B,9B} 与 8B-Q5×9B；3.5-4B 内容答对但常忘标引用（可追溯 5~6/12）；27B 最慢（78~93s/题）。',
              '',
              '---',
              '',
              '## 三、综合命中（检索 × 生成 双层，17 组实测）',
              '',
              '口径：检索命中=锚句核心词在 top-3 块内（纯嵌入）；生成命中=核心词在最终答案内（端到端）；综合=同题双命中。',
              '补齐：9 嵌入 × Qwen3-4B-Q4 聊天（控制聊天变量），叠加 4B-Q4/8B-Q5 × 其余 4 聊天。',
              '',
              '| 排名 | 嵌入 | 聊天 | 检索命中 | 生成命中 | 综合命中 | 平均耗时 |',
              '|---|---|---|---|---|---|---|---|']
    for i, r in enumerate(sorted(combo_rows(groups), key=lambda r: (-r['combo'], r['secs']))):  # type: ignore[arg-type]
        lines.append(
            f"| {i + 1} | {short_name(str(r['embed']))} | {chat_short(str(r['chat']))} | {r['ret']}/12 | "
            f"{r['gen']}/12 | **{r['combo']}/12** | {r['secs']:.1f}s |"
        )
    lines += ['',
              '要点：4B-Q4/Q5 嵌入是唯一 12/12（其余 11/12 且全丢 Q11"CPU 版"场景题）；KaLM-12B 垫底（9~10/12）。',
              '',
              '---',
              '',
              '## 四、为什么 45 组全矩阵只测了 17 组',
              '',
              '全矩阵 = 9 嵌入 × 5 聊天 = 45 组，实测 17 组；未测的 28 组 = 其余 7 嵌入 × {1.7B, 3.5-4B, 9B, 27B}。原因不是偷懒，是控制变量法：',
              '',
              '1. **阶段 A 已证明检索与聊天无交互**：同嵌入档内换任意聊天模型，top-3 检索结果完全一致（12/12）。',
              '   检索质量只由嵌入模型决定——所以 9 个嵌入只需配一个代表聊天模型（选 4B-Q4）即可排序。',
              '2. **生成质量只由聊天模型决定**：top-3 一旦包含答案块，能否答对取决于聊天模型本身——',
              '   所以聊天只需在两档代表性嵌入（4B-Q4/8B-Q5）上测 5 个即可排序。',
              '3. **唯一需要担心的交互效应**（嵌入检索差 1 题 → 综合掉 1 题）已由 8B 档验证：检索 11/12 → 综合 9~11/12。',
              '4. 剩余 28 组是已测两个变量"主效应"之后的笛卡尔积冗余，增量信息 ≈ 0；成本约 4 小时（27B 两组占大半，且 27B 已淘汰）。',
              '',
              '一句话：**主效应测全了，交互效应已被 8B 档探明**——17 组足以支撑选型。',
              '',
              '---',
              '',
              '## 五、推荐选型（供 README / 整合包 / 开发者文档引用）',
              '',
              '### GPU / CUDA 档（推荐）',
              '- 聊天：**Qwen3.5-9B-Q5_K_M**（6.07GB）——综合 12/12 + 可追溯 12/12 双满分，3.4s/题',
              '- 嵌入：**Qwen3-Embedding-4B-Q4_K_M**（2.33GB）——综合 12/12，2.33GB 三档通用',
              '',
              '### CPU 档',
              '- 聊天：**Qwen3-4B-Q4_K_M**（2.33GB）——综合 12/12，1.6s/题，206 tok/s',
              '- 嵌入：**Qwen3-Embedding-4B-Q4_K_M**（2.33GB）满配；**0.6B-Q8_0**（0.60GB）省内存（综合 11/12，丢 Q11"CPU 版"场景题）',
              '',
              '### Vulkan / 核显档',
              '- 聊天：**Qwen3-4B-Q4_K_M**（2.33GB）——综合 12/12，可追溯 11/12（比 3.5-4B 的 5/12 稳）',
              '- 嵌入：**Qwen3-Embedding-4B-Q4_K_M**（2.33GB）',
              '',
              '### 淘汰结论',
              '- **Qwen3.8-27B**：综合 11/12 无优势，77.8s/题慢 20 倍——淘汰',
              '- **KaLM-Embedding-Gemma3-12B**：综合 9~10/12 垫底，6.8~7.9GB 体积 3 倍——淘汰',
              '- **Qwen3-1.7B**：综合 11/12 但可追溯 8/12，仅极端省内存场景考虑',
              '- **bge-m3 / harrier / 0.6B**：综合 11/12，可作 CPU 嵌入平替，但会丢"CPU 版"场景题',
              '',
    ]
    OUT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'已生成：{OUT}')
    print(f'表一 {len(embed_rows())} 行 · 表二 {len(chat_rows(groups))} 行 · 表三 {len(combo_rows(groups))} 行')


if __name__ == '__main__':
    main()
