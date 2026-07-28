"""Pluggable LLM backends — the actor and reviewer are each just an LLMBackend.

- `MockBackend`   : deterministic canned responses (for tests / dry-run of the control loop).
- `CLIBackend`    : shells out to a CLI agent (Claude Code `claude -p`, or `codex exec`) headlessly,
                    feeding the user prompt on stdin and the system prompt via a flag, and reading the
                    model's text back from stdout or a last-message file. The exact argv is CONFIGURABLE
                    (a template list with {system}/{prompt}/{outfile} placeholders) so the same class
                    drives claude or codex, and so flags can be corrected for your CLI version without
                    touching code.

Backends return raw text; `parse_json` robustly extracts the JSON object the prompts ask the model for.

Both presets below are VERIFIED against claude 2.1.220 / codex-cli 0.145.0 with the logged-in
subscriptions (no API keys). Both follow implementations.md §7.4 (model execution safety): the prompt
goes on stdin (not argv, so size is unbounded), and the agents get no ability to act — claude with
`--tools ""`, codex with `--sandbox read-only --ephemeral`.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path


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
    """Drive a headless CLI agent. `argv_template` items may contain the placeholders:

      {system}  -> the system prompt (if absent from the template it is folded into the prompt text,
                   for CLIs with no system-prompt flag)
      {prompt}  -> the user prompt (omit it and set stdin_prompt=True to feed the prompt on stdin,
                   which is what both presets do: argv has a length limit, prompts do not)
      {outfile} -> a temp file the CLI is told to write its final message to (codex `-o`)

    `text_from` selects where the answer is read from: 'raw' (stdout IS the text), 'file' (read
    {outfile}), or a JSON field name ('result' for Claude's `--output-format json` envelope).

    Both presets run in non-bare mode on purpose -> they authenticate via the logged-in subscription
    session (no ANTHROPIC_API_KEY / OPENAI_API_KEY needed)."""

    # Claude Code headless. Prompt on stdin; answer in the JSON envelope's "result" field.
    # `--tools ""` disables the entire built-in tool set -> the actor can only emit text (§7.4).
    CLAUDE = ["claude", "-p", "--output-format", "json", "--append-system-prompt", "{system}", "--tools", ""]
    CLAUDE_TEXT_FROM = "result"
    # Codex non-interactive. Trailing "-" = read the prompt from stdin; no system flag, so the system
    # prompt is folded in. read-only sandbox + --ephemeral (no session files) per §7.4; stdout is an
    # event log, so the actual answer is read from the -o last-message file.
    CODEX = ["codex", "exec", "--sandbox", "read-only", "--ephemeral", "--skip-git-repo-check",
             "--color", "never", "-o", "{outfile}", "-"]

    def __init__(self, name: str, argv_template: list[str], text_from: str = "raw",
                 timeout: float = 600.0, stdin_prompt: bool = False, model: str | None = None):
        self.name = name
        self.argv_template = argv_template
        self.text_from = text_from
        self.timeout = timeout
        self.stdin_prompt = stdin_prompt
        self.model = model

    def ask(self, system: str, prompt: str) -> str:
        if not any("{system}" in a for a in self.argv_template) and system:
            prompt = f"[SYSTEM INSTRUCTIONS]\n{system}\n\n[TASK]\n{prompt}"

        outfile = None
        if any("{outfile}" in a for a in self.argv_template):
            fd, outfile = tempfile.mkstemp(prefix="rllm_llm_", suffix=".txt")
            os.close(fd)
        try:
            argv = [a.replace("{system}", system).replace("{outfile}", outfile or "")
                     .replace("{prompt}", "" if self.stdin_prompt else prompt)
                    for a in self.argv_template]
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout,
                                  input=prompt if self.stdin_prompt else None,
                                  stdin=None if self.stdin_prompt else subprocess.DEVNULL)
            if proc.returncode != 0:
                raise RuntimeError(f"{self.name} CLI failed ({proc.returncode}): {proc.stderr[:500]}")
            if self.text_from == "file":
                return Path(outfile).read_text()
            if self.text_from == "raw":
                return proc.stdout
            return str(json.loads(proc.stdout).get(self.text_from, proc.stdout))
        finally:
            if outfile:
                Path(outfile).unlink(missing_ok=True)

    @classmethod
    def claude(cls, timeout: float = 600.0, model: str | None = None) -> "CLIBackend":
        argv = list(cls.CLAUDE) + (["--model", model] if model else [])
        return cls("claude", argv, text_from=cls.CLAUDE_TEXT_FROM, timeout=timeout,
                   stdin_prompt=True, model=model)

    @classmethod
    def codex(cls, timeout: float = 600.0, model: str | None = None) -> "CLIBackend":
        argv = list(cls.CODEX)
        if model:  # insert before the trailing "-" (stdin marker)
            argv = argv[:-1] + ["-m", model, "-"]
        return cls("codex", argv, text_from="file", timeout=timeout, stdin_prompt=True, model=model)


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
