# 典例文档集 · 设计说明（bench_corpus）

用途：横向对比「文档格式 → anydoc → Markdown → 切块 → 向量化检索」的质量差异。

## 设计原则：内容同源，仅格式变排版

六类文档都围绕同一套 RAG 知识点（六节），只按格式特性排版——
word 正式排版、excel 表格化、ppt 标题要点页、epub 章节图文、md 原生、pdf 正式文档、
pdf 扫描版（文本版的光栅化图片件）。
这样横向对比时「内容变量」被控住，测出的差异就是格式本身的影响。

## 典例清单

| 文件 | 类型 | 说明 |
|---|---|---|
| 01_word_rag入门.docx | Word | 正式排版：Heading + 表格 + 代码块 |
| 02_excel_rag知识点.xlsx | Excel | 表格化：知识点速查 + 路线对比两张表 |
| 03_ppt_rag入门.pptx | PPT | 封面 + 6 内容页 + 1 流程图页 |
| 04_epub_rag图文.epub | EPUB | 3 章节 + 内嵌流程示意图 |
| 05_md_rag入门.md | Markdown | 原生基准（对照） |
| 06_pdf_rag入门.pdf | PDF | 文本型正式文档 |
| 07_pdf扫描版_rag入门.pdf | PDF | 扫描型（图片型，06 的光栅化） |

## 六节知识点与判定锚句

| 节 | 锚句 |
|---|---|
| RAG 是什么 | 开卷考试 |
| 为什么要先查资料 | 一本正经地胡说 |
| 切块：两种切法 | 标题结构切 |
| 向量化：文字变数字 | 方向才表达语义 |
| 检索：余弦相似度 | 余弦相似度 |
| 最佳输入格式 | Markdown |

## 测试流程（建议）

1. anydoc 把每份典例转成 Markdown（记录转换告警/丢失项）
2. 转换结果用 chunk_markdown 切块，每类文档得到若干块
3. 六题提问（锚句即答案所在块），逐类文档测 top-3 命中率
4. 记录：块数、总字数、命中率、转换耗时 → 产出对比表

## 预期观察点

- excel 表格转 md 后是否保留表头/行结构，还是粘连成一坨文本
- ppt 要点页转 md 后标题层级是否保留
- epub 图文混排：图片是否丢失、章节结构是否保留
- pdf 表格/代码块是否被截断或粘连
- md 作为基准：原生结构应保持最高命中率（若否，说明 anydoc 或切块有问题）

## 生成方式（可复现）

```
uv run --with python-docx --with openpyxl --with python-pptx --with fpdf2 --with pillow python -m scripts.make_bench_corpus
```

依赖全部按需注入，不写进 pyproject（典例生成是实验工具，不是项目运行时依赖）。

## 测试边界说明

横向对比与挡位推荐基于 x64 Windows 实测（CPU 版 / Vulkan 核显版 / CUDA 独显版），
未测试 ROCm（AMD 独显 Linux）与 arm 架构 CPU。
