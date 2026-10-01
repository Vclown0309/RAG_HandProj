# RAG 手写全链路

> 从零手写 RAG（检索增强生成）全链路 + FastAPI 整合包：本地私有知识库问答，**答案可追溯**。

项目用最朴素的 Python 实现 RAG 的每一环——切块、向量化、存储、余弦检索、Prompt 拼装、生成——不用任何 RAG 框架封装，代码量小、可读性强，既是求职作品，也是学习路线图。

## 亮点

- **手写核心链路**：切块（窗口切 + 标题结构切）、向量化（本地 embedding 模型）、SQLite 存储（BLOB + 去重幂等）、手写余弦相似度 top-k、RAG prompt 拼装——每块代码都能逐行解释
- **答案可追溯**：回答附参考源，演示页右侧侧边栏点击展开原文块，模型依据一验即知
- **开箱即用**：启动本地模型服务后，打开演示页即可上传文档、提问、查看来源、删改知识库
- **全本地运行**：模型用 GGUF 跑在 llama.cpp，数据不出机器
- **工程规范**：uv 环境 + ruff 格式 + mypy 静态检查 + pytest 63 个测试全绿；git 提交遵循 Conventional Commits

## 架构

```
              ┌────────────────────────────────────────────────────┐
              │                    RAG 管线（手写）                  │
  上传文档     │                                                    │
 ─────────►  │  归一化 ─► 切块 ─► 向量化 ─► SQLite 存储             │
  提问        │                    │                                │
 ─────────►  │              余弦检索 top-k ◄─────── 向量化提问      │
              │                    │                                │
              │           拼 Prompt（资料+问题+不编造指令）          │
              │                    │                                │
              │            本地模型生成 ─► 回答 + 参考源            │
              └────────────────────────────────────────────────────┘
    输入                  FastAPI（/ask /library /history /static）
```

## 快速开始

两条路，按需选择：

### 路径 A：纯测试模式（不需要模型、不需要显卡）

```powershell
uv sync
uv run pytest -q        # 63 passed
uv run ruff check rag tests main.py scripts
uv run mypy
```

测试全部用 mock，不依赖真实模型，适合快速了解代码与跑通 CI 姿势。

### 路径 B：完整体验（本地模型 + 演示页）

前置：Windows + uv + llama.cpp（CUDA 版），模型文件放在本地模型目录。

1. **启动嵌入模型**（8081 端口，`--embedding --pooling last` 是编码模型必带参数）：

```powershell
llama-server.exe -m D:\llamacpp_models\qwen3-embed\Qwen3-Embedding-8B-Q5_K_M.gguf --embedding --pooling last -c 8192 -b 8192 -ub 8192 --host 127.0.0.1 --port 8081
```

2. **启动对话模型**（8080 端口，普通聊天模型，不带 `--embedding`）：

```powershell
llama-server.exe -m D:\llamacpp_models\qwen3.5\Qwen3.5-4B-Q5_K_M.gguf -c 8192 -b 8192 -ub 8192 --host 127.0.0.1 --port 8080
```

3. **构建知识库**（可选，内置示例文档 `docs/RAG学习笔记01-模型服务与向量化输入.md`）：

```powershell
uv run python -m scripts.build_kb "docs\RAG学习笔记01-模型服务与向量化输入.md" --reset
```

4. **启动服务**：

```powershell
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

5. 浏览器打开 <http://127.0.0.1:8000> 开始体验：提问、点击参考源展开原文、上传自己的 txt/md 追加入库。

详细步骤与常见问题见 [docs/使用说明.md](docs/使用说明.md)。

## 项目结构

```
RAG_HandProj/
├── main.py              # FastAPI 入口：/ask 问答、/library 知识库管理、/history、演示页
├── rag/
│   ├── normalize.py     # Markdown 方言归一化（callout / 脚注降级）
│   ├── chunking.py      # 切块器：标题结构切 + 超长节窗口切 + 无标题兜底
│   ├── embed.py         # 向量化客户端（POST /v1/embeddings）
│   ├── store.py         # SQLite 存储：chunks（去重幂等）+ history + 余弦检索
│   └── generate.py      # 生成客户端：RAG prompt 拼装 + 调对话模型
├── scripts/build_kb.py  # 构建知识库 CLI
├── static/index.html    # 离线演示页（答案 + 可点击参考源侧边栏）
├── tests/               # 63 个测试（mock 链路 + 契约）
├── docs/                # 学习笔记、使用说明、学习手册
└── CONTRIBUTING.md      # 提交规范（Conventional Commits + 分支语义）
```

## API 一览

| 接口 | 说明 |
|---|---|
| `GET /` | 演示页（离线单页） |
| `POST /ask` | 问答：`{"query": "..."}` → `{"answer": "...", "sources": [{source, content}]}` |
| `GET /library` | 知识库清单（文档 + 块数） |
| `POST /library` | 上传 txt/md 追加入库（自动归一化 → 切块 → 向量化） |
| `DELETE /library/{source}` | 删除文档 |
| `GET /history` | 问答历史（最新在前） |
| `GET /docs` | Swagger UI（接口文档） |

## 分支语义（git log 就是课程地图）

- `main`：里程碑主线，每次合并 = 一个可验收的知识点
- `dev`：碎步工作区，过程细节在这里

```
main  M1 工具链 ─► M2 核心链路 ─► M3 /ask 全链路 ─► M4 知识库管理 ─► M5 门面文档
```

## 技术栈与工具

FastAPI / SQLite / requests / uv / ruff / mypy / pytest / llama.cpp（本地推理）

## License

**待定（review 时拍板）**：MIT（宽松，利于学习传播）/ GPLv3（免费商用 + 二次分发开源 + 禁止闭源倒卖，防别人打包卖钱）。模型文件与文档内容版权归各自来源。
