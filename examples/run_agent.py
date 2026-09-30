#!/usr/bin/env python3
"""OpenHands SDK ReAct loop wired to an ENFORCED Jev policy gate.

Every tool call is routed through a ``GatedExecutor`` that consults
``JevPolicyGate.evaluate()`` before delegating to the real tool executor.
ALLOW delegates to the real executor; DENY/ESCALATE return a blocked
observation so the agent can adapt (no undocumented SDK middleware).

Run modes:
    uv run python examples/run_agent.py --dry-run           # keyless, no network,
                                                              # exercises both gated
                                                              # executors directly
    op run --env-file=env.example.op -- uv run python examples/run_agent.py --fake-jev
    op run --env-file=env.example.op -- uv run python examples/run_agent.py

See docs/CONTRACT.md for the full interface contract.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

# --- Make src/ importable without requiring installation --------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from jev_policy_gate import (
    DeterministicFakeJevClient,
    GateAction,
    JevPolicyGate,
)

DEFAULT_MODEL = "openrouter/anthropic/claude-sonnet-4.5"
WORKSPACE_DIR = _REPO_ROOT / "examples" / "sample_workspace"
AUDIT_DIR = WORKSPACE_DIR / "audit"
AUDIT_PATH = AUDIT_DIR / "gate_audit.jsonl"


def _build_jev_client(fake: bool):
    if fake:
        return DeterministicFakeJevClient()

    # Lazy import: the real client needs an API key and hits the network.
    from jev_policy_gate import JevDecisionClient

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Either export it (e.g. via "
            "`op run --env-file=env.example.op -- ...`) or pass --fake-jev "
            "to run keyless with the deterministic fake Jev client, or use "
            "--dry-run to exercise the gate with zero keys/network."
        )
    return JevDecisionClient(api_key=api_key)


def _build_llm():
    """Construct the OpenHands SDK LLM. Deferred import so keyless modes
    (--dry-run) never need openhands-sdk's LLM class or a key at all."""
    from openhands.sdk import LLM

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Export it (e.g. via "
            "`op run --env-file=env.example.op -- ...`) before running "
            "without --dry-run."
        )

    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    return LLM(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


class AuditLog:
    """Appends every gate decision (and other events) as JSONL."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        record = {"ts": time.time(), **record}
        line = json.dumps(record, default=str)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return line


def _gated_executor_base():
    """ToolExecutor is generic and needs openhands.sdk; deferred import so
    keyless code paths never require the SDK to be importable at module
    load time (only when a tool is actually being built)."""
    from openhands.sdk.tool import ToolExecutor

    return ToolExecutor


class GatedExecutor(_gated_executor_base()):
    """Wraps a real tool executor with a mandatory Jev policy check.

    ALLOW -> delegates to the wrapped executor.
    DENY / ESCALATE -> returns a blocked observation (never runs the tool).
    """

    def __init__(
        self,
        tool_name: str,
        real_executor,
        gate: JevPolicyGate,
        audit: AuditLog,
        observation_cls,
    ):
        self.tool_name = tool_name
        self.real_executor = real_executor
        self.gate = gate
        self.audit = audit
        self.observation_cls = observation_cls

    def __call__(self, action, conversation=None):
        action_args = _action_to_dict(action)
        state = {"tool": self.tool_name, "action_args": action_args}

        decision = self.gate.evaluate(
            tool_name=self.tool_name, action_args=action_args, state=state
        )

        line = self.audit.write(
            {
                "event": "gate_decision",
                "tool": self.tool_name,
                "action_args": action_args,
                "gate_action": decision.action.name,
                "confidence": decision.confidence,
                "reason": decision.reason,
            }
        )
        print(f"[gate] {self.tool_name}: {decision.action.name} -> {line}")

        if decision.action == GateAction.ALLOW:
            result = self.real_executor(action, conversation=conversation)
            self.audit.write(
                {
                    "event": "tool_executed",
                    "tool": self.tool_name,
                    "action_args": action_args,
                }
            )
            return result

        blocked_text = (
            f"[POLICY GATE] Action on tool '{self.tool_name}' was "
            f"{decision.action.name} by Jev (confidence={decision.confidence:.2f}, "
            f"reason={decision.reason!r}). "
        )
        if decision.action == GateAction.ESCALATE:
            blocked_text += "This action requires human approval before it can run."
        else:
            blocked_text += "This action was not executed. Choose a safer alternative."

        return self.observation_cls.from_text(text=blocked_text, is_error=True)


def _action_to_dict(action) -> dict[str, Any]:
    """Best-effort conversion of an SDK Action into a plain dict for Jev."""
    if hasattr(action, "model_dump"):
        return action.model_dump()
    if hasattr(action, "__dict__"):
        return dict(action.__dict__)
    return {"repr": repr(action)}


def _seed_sample_workspace() -> None:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    notes = WORKSPACE_DIR / "notes.txt"
    if not notes.exists():
        notes.write_text(
            "ops runbook: check disk usage, rotate logs, do not delete prod backups\n",
            encoding="utf-8",
        )
    log = WORKSPACE_DIR / "log.txt"
    if not log.exists():
        log.write_text("2024-01-01 ok: nightly backup completed\n", encoding="utf-8")
    file9 = WORKSPACE_DIR / "file9.txt"
    if not file9.exists():
        file9.write_text("safe demo file: hello from file9\n", encoding="utf-8")
    file4 = WORKSPACE_DIR / "file4.txt"
    if not file4.exists():
        file4.write_text("safe demo file: hello from file4\n", encoding="utf-8")


# --- Custom tools, per docs.openhands.dev/sdk/guides/custom-tools ----------


def _build_tool_classes(gate: JevPolicyGate, audit: AuditLog):
    """Builds and registers the ReadFileTool/DeleteFileTool ToolDefinition
    subclasses whose ``create()`` wraps each real executor in a
    GatedExecutor. Returns (ReadFileTool, DeleteFileTool) classes.

    Deferred import: this needs openhands.sdk installed, but not a key or
    network access, so --dry-run can still call it.
    """
    from openhands.sdk import Action, Observation
    from openhands.sdk.tool import ToolDefinition, ToolExecutor, register_tool
    from pydantic import Field

    class ReadFileAction(Action):
        """Read a text file from the sample workspace (safe, read-only)."""

        path: str = Field(description="Path relative to the sample workspace.")

    class ReadFileObservation(Observation):
        """Observation from reading a file (or from a gate block)."""

    class _RealReadFileExecutor(ToolExecutor[ReadFileAction, ReadFileObservation]):
        def __call__(self, action: ReadFileAction, conversation=None):
            target = (WORKSPACE_DIR / action.path).resolve()
            try:
                content = target.read_text(encoding="utf-8")
                return ReadFileObservation.from_text(text=content)
            except OSError as exc:
                return ReadFileObservation.from_text(
                    text=f"error reading {action.path}: {exc}", is_error=True
                )

    class ReadFileTool(ToolDefinition[ReadFileAction, ReadFileObservation]):
        @classmethod
        def create(cls, conv_state=None, **kwargs) -> Sequence[ReadFileTool]:
            gated = GatedExecutor(
                "read_file", _RealReadFileExecutor(), gate, audit, ReadFileObservation
            )
            return [
                cls(
                    description=ReadFileAction.__doc__ or "Read a file.",
                    action_type=ReadFileAction,
                    observation_type=ReadFileObservation,
                    executor=gated,
                )
            ]

    class DeleteFileAction(Action):
        """Delete a file from the sample workspace (destructive, risky)."""

        path: str = Field(description="Path relative to the sample workspace.")

    class DeleteFileObservation(Observation):
        """Observation from deleting a file (or from a gate block)."""

    class _RealDeleteFileExecutor(ToolExecutor[DeleteFileAction, DeleteFileObservation]):
        def __call__(self, action: DeleteFileAction, conversation=None):
            target = (WORKSPACE_DIR / action.path).resolve()
            try:
                target.unlink()
                return DeleteFileObservation.from_text(text=f"deleted {action.path}")
            except OSError as exc:
                return DeleteFileObservation.from_text(
                    text=f"error deleting {action.path}: {exc}", is_error=True
                )

    class DeleteFileTool(ToolDefinition[DeleteFileAction, DeleteFileObservation]):
        @classmethod
        def create(cls, conv_state=None, **kwargs) -> Sequence[DeleteFileTool]:
            gated = GatedExecutor(
                "delete_file",
                _RealDeleteFileExecutor(),
                gate,
                audit,
                DeleteFileObservation,
            )
            return [
                cls(
                    description=DeleteFileAction.__doc__ or "Delete a file.",
                    action_type=DeleteFileAction,
                    observation_type=DeleteFileObservation,
                    executor=gated,
                )
            ]

    register_tool("read_file", ReadFileTool)
    register_tool("delete_file", DeleteFileTool)

    return ReadFileTool, DeleteFileTool


def run_dry_run(gate: JevPolicyGate, audit: AuditLog) -> None:
    """Keyless, network-free proof: builds both GatedExecutors via
    ToolDefinition.create() (no Conversation/LLM involved) and exercises
    each with one representative action -- a safe read and a risky
    delete -- printing the gate decision and audit line for each."""
    ReadFileTool, DeleteFileTool = _build_tool_classes(gate, audit)

    read_defs = ReadFileTool.create()
    delete_defs = DeleteFileTool.create()
    read_tool_def = read_defs[0]
    delete_tool_def = delete_defs[0]

    read_action = read_tool_def.action_type(path="file4.txt")
    print("\n--- Exercising read_file (expected: ALLOW) ---")
    read_obs = read_tool_def.executor(read_action)
    print(f"[observation] {read_obs.to_llm_content}")

    delete_action = delete_tool_def.action_type(path="notes.txt")
    print("\n--- Exercising delete_file (expected: DENY/ESCALATE) ---")
    delete_obs = delete_tool_def.executor(delete_action)
    print(f"[observation] {delete_obs.to_llm_content}")

    print(f"\nAudit log written to: {AUDIT_PATH}")


def run_live(gate: JevPolicyGate, audit: AuditLog, task: str) -> None:
    """Runs the real OpenHands SDK ReAct loop against a live chat LLM."""
    llm = _build_llm()

    _build_tool_classes(gate, audit)

    from openhands.sdk import Agent, Conversation, Tool

    tools = [Tool(name="read_file"), Tool(name="delete_file")]
    agent = Agent(llm=llm, tools=tools)

    def on_event(event) -> None:
        audit.write({"event": "conversation_event", "detail": repr(event)})
        print(f"[event] {event!r}")

    conversation = Conversation(
        agent=agent, callbacks=[on_event], workspace=str(WORKSPACE_DIR)
    )
    conversation.send_message(task)
    conversation.run()

    print(f"\nAudit log written to: {AUDIT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fake-jev",
        action="store_true",
        help="Use DeterministicFakeJevClient instead of the real Jev decision API.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Keyless, network-free: directly exercise both GatedExecutors "
            "(one safe read, one risky delete) instead of running the SDK "
            "Conversation loop. Implies --fake-jev; needs no LLM key."
        ),
    )
    parser.add_argument(
        "--task",
        default=(
            "You manage a small ops workspace. First read notes.txt to understand "
            "the runbook, then attempt to delete notes.txt as a cleanup step."
        ),
        help="Task prompt handed to the agent (demo has one safe + one risky action).",
    )
    args = parser.parse_args()

    _seed_sample_workspace()
    audit = AuditLog(AUDIT_PATH)

    jev_client = _build_jev_client(fake=args.fake_jev or args.dry_run)
    gate = JevPolicyGate(client=jev_client)

    if args.dry_run:
        run_dry_run(gate, audit)
    else:
        run_live(gate, audit, args.task)


if __name__ == "__main__":
    main()
