from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

from .benchmark import run_replay_benchmark
from .models import FakeModelClient
from .providers import create_model_client
from .replay import ReplayDiffReport, ReplayRunner
from .runtime import Tifa, build_agent


def _approver(name: str, arguments: dict[str, Any]) -> bool:
    answer = input(f"Approve {name} {json.dumps(arguments, ensure_ascii=False)}? [y/N] ")
    return answer.strip().lower() in {"y", "yes"}


def command_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="tifa")
    commands = root.add_subparsers(dest="command")
    replay = commands.add_parser("replay"); replay.add_argument("bundle", type=Path); replay.add_argument("--mode", choices=["offline", "forked", "counterfactual"], default="offline"); replay.add_argument("--output", type=Path)
    benchmark = commands.add_parser("benchmark"); bsub = benchmark.add_subparsers(dest="benchmark_command"); breplay = bsub.add_parser("replay"); breplay.add_argument("--mode", choices=["smoke", "full"], default="smoke"); breplay.add_argument("--output", type=Path)
    return root


def agent_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="tifa")
    root.add_argument("prompt", nargs="?"); root.add_argument("--cwd", default="."); root.add_argument("--provider", choices=["fake", "openai", "anthropic", "ollama"], default="fake")
    root.add_argument("--model"); root.add_argument("--max-steps", type=int, default=8); root.add_argument("--approval", choices=["never", "on-risk", "always"], default="on-risk"); root.add_argument("--resume")
    return root


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    is_command = bool(raw and raw[0] in {"replay", "benchmark"})
    args = (command_parser() if is_command else agent_parser()).parse_args(raw)
    if getattr(args, "command", None) == "replay":
        result = ReplayRunner().replay(args.bundle, args.mode)
        payload = asdict(result) if isinstance(result, ReplayDiffReport) else result
        if args.output: args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2)); return 0 if payload.get("replay_consistent", payload.get("status") == "NOT_IMPLEMENTED") else 1
    if getattr(args, "command", None) == "benchmark":
        if args.benchmark_command != "replay": command_parser().error("benchmark requires replay")
        print(json.dumps(run_replay_benchmark(args.mode, args.output), ensure_ascii=False, indent=2)); return 0
    client = FakeModelClient() if args.provider == "fake" else create_model_client(args.provider, args.model)
    kwargs = {"max_steps": args.max_steps, "approval_policy": args.approval, "approver": _approver}
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
