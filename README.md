# Tifa

Tifa 是依据当前工作区保存的 Pico 设计资料、schema、fixture 和 measured artifact 重建的本地代码 Agent Harness。它面向代码仓库长任务，提供有状态 Agent Loop、分层上下文、结构化记忆、受治理工具、checkpoint/resume、运行工件、trace 和确定性 Offline Replay。

本仓库不是遗失 Pico 源码的逐行恢复。能够确认的历史能力按 Pico 资料重建；Tifa 在此基础上完成命名迁移、版本化工件、恢复一致性校验以及 Runtime 与 Offline Replay 的集成。

## 快速开始

```powershell
git clone https://github.com/cuangzhou/tifa-harness.git
cd tifa-harness
python -m pip install -e ".[test]"
python -m tifa "介绍当前仓库"
```

默认使用确定性的 `FakeModelClient`，不需要 API Key，并会在目标工作区生成：

```text
.tifa/
├── sessions/<session-id>.json
├── memory/memory.json
└── runs/<run-id>/
    ├── task_state.json
    ├── trace.jsonl
    ├── checkpoint.json
    ├── report.json
    └── evidence_bundle.json
```

SDK 示例：

```python
from tifa import FakeModelClient, build_agent

agent = build_agent(
    ".",
    FakeModelClient(["<final>offline demo completed</final>"]),
    approval_policy="never",
)
result = agent.ask("检查仓库结构")
print(result.answer, result.run_dir)
```

真实 provider：

```powershell
python -m tifa --provider openai --model gpt-4.1-mini "检查测试"
python -m tifa --provider anthropic --model claude-sonnet-4-20250514 "检查测试"
python -m tifa --provider ollama --model qwen2.5-coder "检查测试"
```

OpenAI-compatible、Anthropic-compatible 和 Ollama 分别读取其标准 API Key、base URL 和 model 环境变量。Tifa 不把凭据写入 trace 或 session。

## 模型控制协议

模型输出必须采用以下一种形式：

```text
<tool>{"name":"read_file","arguments":{"path":"README.md"}}</tool>
<final>最终回答</final>
```

工具包括 `list_files`、`read_file`、`search`、`run_shell`、`write_file`、`patch_file` 和受限只读 `delegate`。高风险工具默认采用 `on-risk` 审批；路径被限制在工作区内，重复调用通过调用指纹拦截。

## 恢复与兼容

```powershell
python -m tifa --resume latest "继续任务"
python -m tifa --resume <session-id> "继续任务"
```

恢复前会比较 workspace fingerprint、provider/model、approval policy 和工具签名。存在不一致时拒绝恢复，避免静默重复副作用。旧 `.pico/sessions/*.json` 仅作为导入源读取，新状态始终写入 `.tifa/`。公共 API 保留 `Pico = Tifa` 兼容别名。

## Offline Replay 与 benchmark

```powershell
python -m tifa replay evaluation/fixtures/doc_01.json --mode offline
python -m tifa benchmark replay --mode smoke
python -m tifa benchmark replay --mode full
python -m pytest
```

Offline Replay 不调用 provider、不执行真实工具、不修改 fixture 工作区。项目保留 12 个实际执行的历史 Pico fixture；矩阵中的 24 是规划规模，benchmark 使用 `executed_fixture_count=12` 单独披露实际数量。

## 事实边界

- 已实现：Harness、Agent Loop、上下文预算、结构化记忆、工具治理、版本化 session/run 工件、checkpoint/resume 一致性检查、EvidenceBundle、只读 Offline Replay 和确定性 benchmark。
- 未实现：Forked Replay、Counterfactual Replay、案例辅助、案例自动晋升以及 checkpoint 重新执行率。相关调用返回结构化 `NOT_IMPLEMENTED`。
- 未使用 LangGraph，也没有 MCP Server。
- 历史上下文指标仅作为既有证据记录：平均 prompt 长度 7082→5664，平均压缩率 16.19%，最高 33.28%。本仓库不把它改写成新实现的重新测量结果。
