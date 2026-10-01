# CONTRIBUTING

## 分支语义

- `main`：里程碑主线。每个 commit = 一个可验收的步骤 / 知识点。
  历史即课程地图——给招聘者看，git log main 一眼一条知识。
- `dev`：碎步工作区。小步迭代、过程留痕；攒够一个里程碑，合成后 no-ff 合并进 main。

## 提交规范（Conventional Commits）

统一使用约定式提交格式，保持 git 历史的机器可读性（changelog / 版本自动化）与人类可读性（git log 一目了然）。

```
<type>(<scope>): <subject>
```

- `type`：本次提交的性质（必填，见下表）
- `scope`：影响范围（可选，如 `rag` / `tests` / `main`）
- `subject`：祈使句，言简意赅，中文或英文均可，尽量 ≤ 50 字符

### type 最小集

| type | 含义 |
|---|---|
| `feat` | 新功能 |
| `fix` | 修 bug |
| `docs` | 文档 |
| `refactor` | 重构（行为不变） |
| `test` | 补测试 |
| `chore` | 杂务（依赖、配置） |
| `perf` | 性能 |
| `style` | 格式（不改行为） |

### 可选的 body / footer

- **body**：一行讲"为什么"（动机），不写"怎么做"（看 diff 就知道怎么做）。
- **BREAKING CHANGE**：破坏性变更必须写（例如换模型导致知识库重建）。

### 示例

```bash
# 功能
feat(rag): 方言归一化，降级 callout/脚注
标题结构由 anydoc 原生保留，正则仅清理扩展方言

# 修 bug
fix(rag): 修复无标题文档退化为窗口切时重叠丢失

# 破坏性变更
refactor(kb): chunks 表按 4096 维重建
BREAKING CHANGE: 换模型后旧向量维度不匹配，需重建知识库
```

## 本地模板（可选）

仓库内置 `.gitmessage` 提交模板，一条命令启用：

```powershell
git config commit.template .gitmessage
```

之后 `git commit` 会自动带格式提示（`#` 开头为注释，不会进入提交信息）。
