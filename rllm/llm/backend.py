"""Pluggable LLM backends — the actor and reviewer are each just an LLMBackend.

- `MockBackend`   : deterministic canned responses (for tests / dry-run of the control loop).
- `CLIBackend`    : shells out to a CLI agent (Claude Code `claude -p`, or `codex exec`) headlessly,
                    feeding the user prompt on stdin and the system prompt via a flag, and reading the
                    model's text back from stdout. The exact argv is CONFIGURABLE (a template list with
                    a {system} placeholder) so the same class drives claude or codex, and so flags can
                    be corrected for your CLI version without touching code.

Backends return raw text; `parse_json` robustly extracts the JSON object the prompts ask the model for.
NOTE: real CLI invocation can only be validated on your machine (it needs the logged-in subscription).
The default flag sets below are starting points — verify against your `claude`/`codex` version.
"""
from __future__ import annotations

import json
import re
import subprocess
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    name: str

    @abstractmethod
    def ask(self, system: str, prompt: str) -> str:
        """Return the model's text response to (system, prompt)."""


class MockBackend(LLMBackend):
    name = "mock"

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def ask(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        return self._responses.pop(0) if self._responses else "{}"


class CLIBackend(LLMBackend):
    """Drive a headless CLI agent. `argv_template` items may contain '{system}' and '{prompt}',
    substituted before exec. `text_from`: 'raw' (stdout IS the text) or a JSON field ('result' for the
    Claude `--output-format json` envelope). If no template item contains '{system}', the system prompt
    is folded into the prompt (for CLIs without a system-prompt flag, e.g. codex).

    Claude preset uses NON-bare mode on purpose -> it authenticates via the logged-in subscription
    session (no ANTHROPIC_API_KEY needed). `--permission-mode dontAsk` keeps it text/JSON only (no
    blocking prompts, no actions). Optionally add `--model ...` / `--max-budget-usd ...` to the template."""

    # Claude Code headless (verified against the docs): text lands in the JSON envelope's "result".
    CLAUDE = ["claude", "-p", "{prompt}", "--output-format", "json",
              "--append-system-prompt", "{system}", "--permission-mode", "dontAsk"]
    CLAUDE_TEXT_FROM = "result"
    # Codex non-interactive; no system flag -> system gets folded into the prompt. Adjust to your codex.
    CODEX = ["codex", "exec", "{prompt}"]

    def __init__(self, name: str, argv_template: list[str], text_from: str = "raw", timeout: float = 300.0):
        self.name = name
        self.argv_template = argv_template
        self.text_from = text_from
        self.timeout = timeout

    def ask(self, system: str, prompt: str) -> str:
        has_system_slot = any("{system}" in a for a in self.argv_template)
        if not has_system_slot and system:
            prompt = f"[SYSTEM INSTRUCTIONS]\n{system}\n\n[TASK]\n{prompt}"
        argv = [a.replace("{system}", system).replace("{prompt}", prompt) for a in self.argv_template]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"{self.name} CLI failed ({proc.returncode}): {proc.stderr[:500]}")
        out = proc.stdout
        if self.text_from == "raw":
            return out
        return str(json.loads(out).get(self.text_from, out))

    @classmethod
    def claude(cls, timeout: float = 300.0) -> "CLIBackend":
        return cls("claude", cls.CLAUDE, text_from=cls.CLAUDE_TEXT_FROM, timeout=timeout)

    @classmethod
    def codex(cls, timeout: float = 300.0) -> "CLIBackend":
        return cls("codex", cls.CODEX, text_from="raw", timeout=timeout)


def parse_json(text: str) -> dict:
    """Extract the first top-level JSON object from a model response (tolerates prose/code fences)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        depth += 1 if text[i] == "{" else -1 if text[i] == "}" else 0
        if depth == 0:
            return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object in response")
