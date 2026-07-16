# Configuration reference

## CLI execution

- `--execution-backend docker|local`: command and verifier backend; local is degraded.
- `--network-policy deny|allow`: Docker network mode. Default: `deny`.
- `--cpu-limit`: Docker CPU quota. Default: `1.0`.
- `--memory-limit`: memory MiB. Default: `512`.
- `--timeout`: command timeout seconds. Default: `30`.

## Provider environment

- OpenAI-compatible: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`.
- Anthropic-compatible: `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`.
- Ollama: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`.
- Optional live smoke: `TIFA_LIVE_TEST_PROVIDER`.

## Workspace indexing

Tifa combines `.gitignore`, `.tifaignore`, and built-in cache/build-directory exclusions. The index caches path, size, modification time, and content digest in `.tifa/cache/`; only changed files are rehashed.
