# Tifa

Tifa 是面向代码仓库长任务的本地 Agent Harness，提供有状态 Agent Loop、结构化 Tool Calling、分层上下文、结构化记忆、受治理工具、durable checkpoint/resume、证据工件和可隔离 Replay。

仓库中的能力与指标只以当前源码、测试和可重新生成的 measured artifact 为证据。

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

真实 Provider 优先返回结构化 ToolCall；FakeModel 和历史适配仍可使用以下文本协议：

```text
<tool>{"name":"read_file","arguments":{"path":"README.md"}}</tool>
<final>最终回答</final>
```

工具包括 `list_files`、`read_file`、`search`、`run_shell`、`write_file`、`patch_file` 和受限只读 `delegate`。高风险工具默认采用 `on-risk` 审批；路径被限制在工作区内，重复调用通过调用指纹拦截。

## 会话恢复、运行续跑与兼容

```powershell
python -m tifa --resume latest "继续任务"
python -m tifa --resume <session-id> "继续任务"
python -m tifa resume-run <run-id> --checkpoint <checkpoint-id> "继续中断任务"
```

`from_session()` 只恢复会话；`resume_run()` 从不可变 checkpoint 创建 continuation run。恢复前会比较 digest、workspace fingerprint、provider/model、approval policy 和工具签名，写工具通过 prepare/commit、文件 digest 和调用指纹避免重复副作用。

## Offline Replay 与 benchmark

```powershell
python -m tifa replay evaluation/fixtures/doc_01.json --mode offline
python -m tifa replay evaluation/fixtures/doc_01.json --mode forked --workspace .
python -m tifa replay-diff original_bundle.json replay_bundle.json
python -m tifa benchmark replay --mode smoke
python -m tifa benchmark replay --mode full
python -m pytest
```

Offline Replay 不调用 provider 或真实工具。Forked Replay 校验源 digest 后使用临时快照副本，测试保证源目录零写入。Counterfactual Replay 强制单变量 override，额外差异会标记 `confounded`。当前 24 个独立确定性 fixture 全部实际执行并可由生成器重新构建。

## CaseCard

```powershell
python -m tifa cases list
python -m tifa cases search bugfix --top-k 3
python -m tifa cases promote candidate.json replay_result.json
python -m tifa cases reject <case-id>
```

只有同任务合同、同 snapshot、单变量修正、verifier 通过、预算合规且证据完整的案例才能晋升为 `verified`；Runtime 只注入兼容的 verified 摘要。

## 事实边界

- 已实现并由本仓库测试支撑：结构化 Tool Calling、不可变 checkpoint、continuation run、工具副作用幂等、v2 EvidenceBundle、可配置 verifier、Offline/Forked/Counterfactual Replay、CaseCard 门禁与 24 fixture benchmark。
- Forked/Counterfactual 与案例辅助的结果来自确定性工程合同和隔离测试，不代表真实模型任务收益；live provider smoke test 默认跳过，需通过 `TIFA_LIVE_TEST_PROVIDER` 显式开启。
- 未使用 LangGraph，也没有 MCP Server。
- 尚未完成的生产化工作包括容器级命令隔离、资源配额、大仓库压力测试和多 Provider 大规模真实任务评测。
