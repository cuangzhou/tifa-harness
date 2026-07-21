# Capability evidence

This page maps the public Tifa capability claims to code, tests, and reproducible evidence. It is an implementation index, not a claim that the current release candidate has passed every live-model release gate.

| Capability | Implementation | Verification |
|---|---|---|
| Stateful Agent Loop | `tifa/runtime.py`, `tifa/models.py` | `tests/test_runtime.py`, `tests/test_complete_spec.py` |
| Checkpoint and resume | versioned checkpoint store, workspace digest validation, explicit resume entry points | interruption, round-trip, mismatch, and tamper tests in `tests/test_complete_spec.py` |
| Tool governance | JSON-schema arguments, workspace-safe paths, approval policy, writable-path contracts, duplicate-call fingerprints | `tests/test_tools_context.py`, `tests/test_runtime.py` |
| Layered context and freshness | `tifa/context_manager.py`, `tifa/memory.py`, `tifa/cases.py` | `tests/test_tools_context.py`, `tests/test_inheritance.py` |
| Runtime evidence | task state, trace, checkpoint, report, metrics, logs, and evidence bundle under `.tifa/` | `tests/test_professional.py`, `evaluation/evidence_bundle.schema.json` |
| Replay | Offline, Forked, and one-variable Counterfactual modes in `tifa/replay.py` | `tests/test_replay.py`, `tests/test_complete_spec.py` |
| Deterministic benchmark | 24 independent fixtures with structured assertions | `python -m tifa benchmark replay --mode full` |
| Provider evaluation | versioned result, case, manifest, and release-decision contracts | `tests/test_evaluation_v2.py`, `evaluation/evaluation_result.schema.json` |
| Harness A/B and ablation | interleaved minimal-loop comparison and capability-group ablations | `tests/test_harness_ab.py`, `python -m tifa eval harness-ab --help` |

## Reproduce locally

```powershell
python -m pip install -e ".[test]"
python -m pytest
python -m tifa benchmark replay --mode full
python -m tifa benchmark context
```

## Evidence boundaries

- The historical PICO measurement reduced average prompt length from 7,082 to 5,664, with 16.19% average compression and 33.28% maximum compression. It is historical evidence, not a Tifa 0.5.0 remeasurement.
- Deterministic fixtures verify harness invariants; they are not real-model task success rates.
- The measured 30-task `qwen2.5-coder:3b` run passed 12/30 verifiers (40%), below its 50% release gate. Duplicate side effects were 0, recovery success was 100%, and provider/schema failures were 0.
- The licensed, commit-pinned real-repository suite and the 100-task, three-repetition cloud evaluation are not yet complete.
- Local Windows execution is degraded isolation. Docker/Linux is the strong command boundary described by the project.
