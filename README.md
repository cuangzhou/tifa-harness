# Tifa

Tifa is a verifiable local coding-agent harness for Python 3.11+. Version 0.5 adds task contracts, verifier-gated completion, bounded repair, semantic code indexing, hardened local execution, durable recovery, replay, and professional evaluation contracts.

## Quick start

```powershell
python -m pip install -e ".[test]"
python -m tifa "Describe this repository"
python -m pytest
```

The deterministic default client requires no credentials. Versioned session, trace, checkpoint, report, metrics, logs, and evidence artifacts are written under `.tifa/`.

## Execution isolation

Docker is the strong-isolation backend. It runs as a non-root user with a read-only root filesystem, a separate workspace mount, networking disabled by default, all capabilities dropped, `no-new-privileges`, and CPU, memory, PID, and timeout limits.

```powershell
docker build -t tifa-runner:0.5.0 -f docker/runner.Dockerfile docker
python -m tifa doctor --cwd .
python -m tifa --execution-backend docker --network-policy deny "Run the tests"
python -m tifa index build --cwd .
```

The local backend executes argv with a filtered environment and process-tree termination. It always reports `isolation_level=local_degraded`; it is not described as a security sandbox.

## Providers and budgets

OpenAI-compatible, Anthropic-compatible, and Ollama clients map structured tool calls and results. Recoverable timeout, HTTP 429, and 5xx failures use bounded exponential backoff; auth, schema, and invalid calls fail immediately. Runtime budgets cover model calls, tokens, tool calls, elapsed time, and optional cost.

```powershell
python -m tifa --provider ollama --model qwen2.5-coder:3b "Inspect the tests"
python -m tifa eval live --provider ollama --output evaluation/artifacts/measured-live/local.json
python -m tifa eval provider --provider openai --repetitions 3 --output evaluation/artifacts/measured-live/openai.json
python -m tifa eval validate evaluation/artifacts/measured-live/openai.json
python -m tifa eval compare left.json right.json --output comparison.json
python -m tifa eval harness-ab --provider openai --model deepseek-v4-flash --cases 30 --repetitions 3 --seed 20260721 --output evaluation/artifacts/harness-ab/deepseek-30x3
```

Provider evaluations use the `tifa-evaluation-result.v2` contract and write summary, case, manifest, and release-decision artifacts. Smoke task selection is category-stratified and reproducible from its recorded seed. Release comparisons require matching code, task set, repetitions, runner, budgets, and model settings; otherwise they are marked `INVALID_FOR_RELEASE_COMPARISON` while retaining descriptive paired counts.

`eval harness-ab` performs an interleaved, same-commit comparison between a safety-constrained minimal function-calling loop and the full Tifa harness. It records invariant and treatment digests, persistent per-run evidence, clustered bootstrap confidence intervals, exact McNemar statistics, safety and efficiency deltas, and optional `--ablations standard` capability-group ablations. Infrastructure-failed pairs remain visible and are excluded from causal capability deltas rather than counted as model failures.

Evaluation is split into regression, capability, and real-repository tracks. The real-repository manifest is currently `NOT_RUN` until licensed, commit-pinned offline repository snapshots are approved; it is not represented as implemented evidence.

Credentials and full raw responses are not persisted. Stable request IDs and summarized usage support correlation.

## Resume, replay, cases, and benchmarks

```powershell
python -m tifa --resume latest "Continue"
python -m tifa resume-run <run-id> --checkpoint <checkpoint-id> "Continue"
python -m tifa replay evaluation/fixtures/doc_01.json --mode offline
python -m tifa benchmark replay --mode full
python -m tifa benchmark replay --mode innovation
python -m tifa benchmark context
python -m tifa benchmark workspace --output evaluation/artifacts/workspace-performance.json
python -m tifa cases search bugfix --top-k 3
python -m tifa cases propose <failed-run-id>
python -m tifa cases verify <case-id> --override memory_enabled=false
python -m tifa cases promote <case-id>
python -m tifa inspect run <run-id> --lineage
python -m tifa report <run-id> --format html --output run-report.html
python -m tifa gc --dry-run
```

Offline replay never invokes a provider or real tools. Forked replay uses a temporary workspace copy and verifies the source stays unchanged. Counterfactual replay permits one registered override; extra differences are `confounded`. The deterministic suite contains 24 independent fixtures. Live-model results are separate measured artifacts created only by an actual run.

Failed runs create CaseCard v2 candidates automatically. Verification is an explicit single-variable counterfactual replay and promotion remains a separate human action. Only verified, fresh cases can be injected, and case assistance stays opt-in through `--case-assistance verified`; selected, dropped, and stale case IDs are recorded in the context manifest. Current context benchmark measurements are separate from the historical PICO compression artifact and do not reuse its numbers.

## Current boundary

Version 0.5.0 is a local CLI and Python SDK, not a remote multi-tenant service. Docker/Linux provides the strong command boundary; Windows local execution remains degraded. Live provider measurements are never mixed with deterministic fixture metrics. Resource limits are container controls, not a claim of formal isolation.

The current release candidate does not yet satisfy its live-model release gates. The measured 30-task `qwen2.5-coder:3b` run at temperature 0 achieved a 40% verifier pass rate (12/30), below the 50% non-regression target; duplicate side effects were 0, recovery success was 100%, and provider/schema failures were 0. A later measured `deepseek-v4-flash` run achieved 63.33% (19/30), with zero duplicate side effects and one transport failure, but remained below the 80% cloud-model gate. The two artifacts use different Tifa commits, so the observed 23.33 percentage-point difference is descriptive rather than a controlled model-only effect. Full 100-task, three-repetition cloud evaluation remains required for release.

Measured evidence is stored under `evaluation/artifacts/measured-live/`; the professional suite definition is `evaluation/professional_suite.v1.json`.

See [configuration](docs/configuration.md), [security policy](SECURITY.md), and [changelog](CHANGELOG.md).

For a claim-to-code index, see [capability evidence](docs/capability-evidence.md).
