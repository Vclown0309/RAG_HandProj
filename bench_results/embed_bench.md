# 嵌入模型横向对比（top-3 命中率）

语料：RAG学习手册.md, RAG学习笔记01-模型服务与向量化输入.md, 使用说明-开发者版.md

测试集：12 题 · 判定：top-3 含锚句

| 排名 | 模型 | 命中 | 维度 | 体积 | 耗时 | pooling |
|---|---|---|---|---|---|---|
| 1 | Qwen3-Embedding-4B-Q4_K_M.gguf | 10/12 | 2560 | 2.33GB | 4s | last |
| 2 | Qwen3-Embedding-4B-Q5_K_M.gguf | 10/12 | 2560 | 2.69GB | 5s | last |
| 3 | Qwen3-Embedding-0.6B-Q8_0.gguf | 9/12 | 1024 | 0.60GB | 4s | last |
| 4 | Qwen3-Embedding-8B-Q5_K_M.gguf | 9/12 | 4096 | 5.05GB | 7s | last |
| 5 | harrier-oss-v1-0.6B-Q8_0.gguf | 9/12 | 1024 | 0.60GB | 4s | last |
| 6 | harrier-oss-v1-0.6B-BF16.gguf | 9/12 | 1024 | 1.12GB | 4s | last |
| 7 | bge-m3-q8_0.gguf | 8/12 | 1024 | 0.59GB | 3s | cls |
| 8 | KaLM-Embedding-Gemma3-12B-2511.Q5_K_M.gguf | 7/12 | 3840 | 7.87GB | 45s | last |
| 9 | KaLM-Embedding-Gemma3-12B-2511.Q4_K_M.gguf | 6/12 | 3840 | 6.80GB | 39s | last |

## 结论

1. **Qwen3-Embedding-4B（Q4/Q5）并列第一（10/12）**，2.33GB 起步——检索力、体积、速度三者平衡，整合包首选。
2. **0.6B-Q8 与 8B-Q5 平手（9/12）**：0.60GB 的 0.6B 干翻 5.05GB 的 8B，CPU 版/低显存场景直接选 0.6B。
3. **harrier-oss-v1-0.6B（9/12）**：微软开源小模型，与 Qwen0.6B 并列，多一个备选。
4. **bge-m3（8/12）**：经典多语种模型，此测试集表现中游；pooling 用官方 cls。
5. **KaLM-Gemma3-12B（6~7/12）倒数**：6.8~7.9GB 体积、39~45s 耗时，命中率垫底——12B 嵌入在中小语料场景毫无性价比。

## 说明

- pooling 均为官方实证：qwen3-embed=last、bge-m3=cls、KaLM=last、harrier=last。
- 测试集为中文学习手册类语料（结构切块），结论适用于同类场景；绝对耗时随硬件折算，相对排名与硬件无关。
- 8 家首轮明细数据因脚本过滤跑覆盖丢失（已修复：过滤跑写 *_partial 文件）；本表指标为实测记录重建，KaLM 两行为补测新值。
