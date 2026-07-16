from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

from .benchmark import run_replay_benchmark
from .cases import CaseCard, CaseStore
from .doctor import doctor
from .execution import DockerExecutionBackend, ExecutionPolicy, LocalExecutionBackend, ResourceLimits
from .models import FakeModelClient
from .live_eval import run_live_eval
from .performance import benchmark_workspace
from .providers import create_model_client
from .replay import ReplayDiffReport, ReplayResult, ReplayRunner, ReplaySpec
from .runtime import Tifa, build_agent


def _approver(name: str, arguments: dict[str, Any]) -> bool:
    answer = input(f"Approve {name} {json.dumps(arguments, ensure_ascii=False)}? [y/N] ")
    return answer.strip().lower() in {"y", "yes"}


def command_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="tifa")
    commands = root.add_subparsers(dest="command")
    replay = commands.add_parser("replay"); replay.add_argument("bundle", type=Path); replay.add_argument("--mode", choices=["offline", "forked", "counterfactual"], default="offline"); replay.add_argument("--output", type=Path)
    replay.add_argument("--spec", type=Path); replay.add_argument("--workspace", type=Path)
    replay.add_argument("--provider", choices=["fake", "openai", "anthropic", "ollama"], default="fake"); replay.add_argument("--model")
    diff = commands.add_parser("replay-diff"); diff.add_argument("original", type=Path); diff.add_argument("replayed", type=Path)
    resume = commands.add_parser("resume-run"); resume.add_argument("run_id"); resume.add_argument("prompt", nargs="?"); resume.add_argument("--checkpoint"); resume.add_argument("--cwd", default="."); resume.add_argument("--provider", choices=["fake", "openai", "anthropic", "ollama"], default="fake"); resume.add_argument("--model"); resume.add_argument("--approval", choices=["never", "on-risk", "always"], default="on-risk")
    cases = commands.add_parser("cases"); csub = cases.add_subparsers(dest="cases_command")
    promote = csub.add_parser("promote"); promote.add_argument("card", type=Path); promote.add_argument("replay", type=Path); promote.add_argument("--cwd", default=".")
    search = csub.add_parser("search"); search.add_argument("category"); search.add_argument("--top-k", type=int, default=3); search.add_argument("--cwd", default=".")
    listing = csub.add_parser("list"); listing.add_argument("--cwd", default=".")
    reject = csub.add_parser("reject"); reject.add_argument("case_id"); reject.add_argument("--cwd", default=".")
    benchmark = commands.add_parser("benchmark"); bsub = benchmark.add_subparsers(dest="benchmark_command"); breplay = bsub.add_parser("replay"); breplay.add_argument("--mode", choices=["smoke", "full"], default="smoke"); breplay.add_argument("--output", type=Path)
    bworkspace = bsub.add_parser("workspace"); bworkspace.add_argument("--output", type=Path); bworkspace.add_argument("--repeats", type=int, default=3)
    doctor_command = commands.add_parser("doctor"); doctor_command.add_argument("--cwd", default="."); doctor_command.add_argument("--model", default="qwen2.5-coder:3b"); doctor_command.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    evaluation = commands.add_parser("eval"); esub = evaluation.add_subparsers(dest="eval_command"); live = esub.add_parser("live"); live.add_argument("--provider", choices=["ollama"], default="ollama"); live.add_argument("--model", default="qwen2.5-coder:3b"); live.add_argument("--repetitions", type=int, default=1); live.add_argument("--output", type=Path, required=True); live.add_argument("--code-version")
    return root


def agent_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="tifa")
    root.add_argument("prompt", nargs="?"); root.add_argument("--cwd", default="."); root.add_argument("--provider", choices=["fake", "openai", "anthropic", "ollama"], default="fake")
    root.add_argument("--model"); root.add_argument("--max-steps", type=int, default=8); root.add_argument("--approval", choices=["never", "on-risk", "always"], default="on-risk"); root.add_argument("--resume")
    root.add_argument("--execution-backend", choices=["docker", "local"], default="local"); root.add_argument("--network-policy", choices=["deny", "allow"], default="deny"); root.add_argument("--cpu-limit", type=float, default=1.0); root.add_argument("--memory-limit", type=int, default=512); root.add_argument("--timeout", type=int, default=30)
    return root


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    is_command = bool(raw and raw[0] in {"replay", "replay-diff", "resume-run", "cases", "benchmark", "doctor", "eval"})
    args = (command_parser() if is_command else agent_parser()).parse_args(raw)
    if getattr(args, "command", None) == "replay":
        spec = ReplaySpec(**json.loads(args.spec.read_text(encoding="utf-8"))) if args.spec else None
        replay_client = None if args.mode == "offline" else (FakeModelClient() if args.provider == "fake" else create_model_client(args.provider, args.model))
        result = ReplayRunner().replay(args.bundle, args.mode, spec=spec, workspace=args.workspace, model_client=replay_client)
        payload = asdict(result) if isinstance(result, (ReplayDiffReport, ReplayResult)) else result
        if args.output: args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        consistent = payload.get("replay_consistent", payload.get("report", {}).get("replay_consistent", False))
        print(json.dumps(payload, ensure_ascii=False, indent=2)); return 0 if consistent else 1
    if getattr(args, "command", None) == "replay-diff":
        original = json.loads(args.original.read_text(encoding="utf-8")); replayed = json.loads(args.replayed.read_text(encoding="utf-8")); print(json.dumps(ReplayRunner.diff(original, replayed), ensure_ascii=False, indent=2)); return 0
    if getattr(args, "command", None) == "resume-run":
        client = FakeModelClient() if args.provider == "fake" else create_model_client(args.provider, args.model)
        agent = Tifa.resume_run(args.cwd, client, args.run_id, args.checkpoint, approval_policy=args.approval, approver=_approver)
        request = args.prompt or (agent.history[-1].get("content", "Continue the interrupted task") if agent.history else "Continue the interrupted task")
        print(json.dumps(asdict(agent.ask(request)), ensure_ascii=False, indent=2)); return 0
    if getattr(args, "command", None) == "cases":
        store = CaseStore(Path(args.cwd) / ".tifa" / "cases")
        if args.cases_command == "promote":
            card = CaseCard(**json.loads(args.card.read_text(encoding="utf-8"))); replay = json.loads(args.replay.read_text(encoding="utf-8")); payload = asdict(store.promote(card, replay))
        elif args.cases_command == "search": payload = [asdict(c) for c in store.search(args.category, top_k=args.top_k)]
        elif args.cases_command == "list": payload = [asdict(c) for c in store.list()]
        elif args.cases_command == "reject": payload = asdict(store.reject(args.case_id))
        else: command_parser().error("cases requires a command")
        print(json.dumps(payload, ensure_ascii=False, indent=2)); return 0
    if getattr(args, "command", None) == "benchmark":
        payload = run_replay_benchmark(args.mode, args.output) if args.benchmark_command == "replay" else benchmark_workspace(args.output, repeats=args.repeats) if args.benchmark_command == "workspace" else None
        if payload is None: command_parser().error("benchmark requires replay or workspace")
        print(json.dumps(payload, ensure_ascii=False, indent=2)); return 0 if payload.get("status", "passed") == "passed" else 1
    if getattr(args, "command", None) == "doctor":
        payload = doctor(args.cwd, args.model, args.ollama_url); print(json.dumps(payload, ensure_ascii=False, indent=2)); return 0 if payload["status"] == "healthy" else 1
    if getattr(args, "command", None) == "eval":
        if args.eval_command != "live": command_parser().error("eval requires live")
        payload = run_live_eval(args.output, args.provider, args.model, args.repetitions, args.code_version); print(json.dumps(payload, ensure_ascii=False, indent=2)); return 0
    client = FakeModelClient() if args.provider == "fake" else create_model_client(args.provider, args.model)
    backend = DockerExecutionBackend() if args.execution_backend == "docker" else LocalExecutionBackend()
    kwargs = {"max_steps": args.max_steps, "approval_policy": args.approval, "approver": _approver, "execution_backend": backend, "execution_policy": ExecutionPolicy(network=args.network_policy), "resource_limits": ResourceLimits(cpus=args.cpu_limit, memory_mb=args.memory_limit, timeout_seconds=args.timeout)}
    agent = Tifa.from_session(args.cwd, client, args.resume, **kwargs) if args.resume else build_agent(args.cwd, client, **kwargs)
    if args.prompt:
        print(json.dumps(asdict(agent.ask(args.prompt)), ensure_ascii=False, indent=2)); return 0
    print("Tifa interactive mode. Use /exit to quit.")
    while True:
        try: request = input("tifa> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if request in {"/exit", "/quit"}: break
        if request: print(agent.ask(request).answer)
    return 0
