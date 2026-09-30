# CONTRACT — jev-openhands-react-policy-gate

Single production-expandable example: **OpenHands SDK ReAct loop + enforced Jev policy layer**.
Jev is a DECISION model only (never a chat/brain model). Chat LLM (via OpenRouter) drives the loop.

## Package layout

```
src/jev_policy_gate/          # Jev client + policy gate (NO OpenHands imports here)
  __init__.py
  jev_client.py               # JevDecisionClient (+ JevDecisionError)
  fake.py                     # DeterministicFakeJevClient
  policy.py                   # JevPolicyGate, GateDecision, GateAction
examples/
  run_agent.py                # OpenHands SDK wiring: gated executors + conversation + audit
tests/                        # unittest, stdlib only, KEYLESS (use fake client)
docs/CONTRACT.md              # this file
env.example.op                # OPENROUTER_API_KEY="op://hermes1/OpenRouter/credential"
pyproject.toml                # uv; deps: openhands-sdk, openhands-tools (pinned same version)
README.md
```

## Jev API (verified contract — do not deviate)

```
POST https://openrouter.ai/api/alpha/decisions
Authorization: Bearer $OPENROUTER_API_KEY        # from env var ONLY; never hardcode/log/persist
{"model": "typesafe/jev-1.13", "state": {...}, "questions": {...}}
```

- Every question object needs EXACTLY: `type` (`noul`|`choice`|`score`), `instructions` (string — never `text`),
  `criteria`.
- noul: criteria = record with `true`/`false` entries, each `{what, not_for?, examples?}`.
- choice: criteria = RECORD mapping option-name -> description (never an array, never key `options`).
- score: criteria = ARRAY of level objects `{what, signals?}`.
- Answers at `response.answers.<name>`: noul = 0-1 float; choice = chosen option; score = NORMALIZED 0-1 value
  plus `legend` dict mapping "0".."4" -> level objects. To get a level label, bucket = min(4, floor(score*5)),
  then take that legend level's `what` — label-match only, NEVER arithmetic like score*4+k.
- A 400 with zod path `questions.<name>.instructions` means text was sent under the wrong key — follow the error's
  `path` field. A missing answer field fails that one question, not the whole run.
- Reference state fields in question text with backtick dot paths: `` `flight.delay_minutes` ``.

## Interfaces (both implementers code against EXACTLY this)

### src/jev_policy_gate/jev_client.py
```python
class JevDecisionClient(Protocol):
    def decide(self, state: dict, questions: dict[str, dict]) -> dict[str, JevAnswer]: ...

@dataclass(frozen=True)
class JevAnswer:
    name: str
    value: float | str        # noul/score -> float 0-1; choice -> str
    confidence: float | None
    raw: dict
```
`JevDecisionClient(api_key: str, model: str = "typesafe/jev-1.13", base_url: str = "https://openrouter.ai/api/alpha/decisions")`.
Uses urllib (stdlib) only. Raises `JevDecisionError` on HTTP error / malformed response / missing answer
(per-question failure tolerated: omit that answer, do not raise for the whole batch).

### src/jev_policy_gate/policy.py
```python
class GateAction(Enum): ALLOW, DENY, ESCALATE
@dataclass(frozen=True)
class GateDecision:
    action: GateAction; confidence: float; reason: str; jev_raw: dict

class JevPolicyGate:
    def __init__(self, client: JevDecisionClient, allow_threshold: float = 0.70, ...): ...
    def evaluate(self, tool_name: str, action_args: dict, state: dict) -> GateDecision: ...
    def audit_log(self) -> list[dict]   # every decision, JSONL-serializable
```
Rules: gate sends ONE batched Jev request per evaluate() call (noul go/no-go + score risk). Decisions:
confidence >= allow_threshold and go -> ALLOW; go but low confidence -> ESCALATE; no-go -> DENY.
Thresholds are constructor args, never magic numbers in logic.

### src/jev_policy_gate/fake.py
`DeterministicFakeJevClient` — hash-based deterministic answers with CONTINUOUS confidence spread
(e.g. 0.05..0.95 derived from a stable hash of tool_name+args — never a few discrete values). Same
noul/score/legend response shape as the real API so tests match the live contract.

### examples/run_agent.py (OpenHands SDK side)
- `LLM(model=<litllm openrouter id>, api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")`
  — read from env ONLY; exit with a clear message if unset. The runner wraps execution in `op run --env-file`.
- Wrap EVERY tool executor in a `GatedExecutor` (in examples/, may import jev_policy_gate): its `__call__`
  runs `policy.evaluate(...)` first — ALLOW -> delegate to real executor; DENY -> return an observation whose
  `to_llm_content` says the action was blocked by policy (so the agent can adapt); ESCALATE -> blocked observation
  saying it needs human approval. This is the enforcement layer; do NOT rely on undocumented SDK middleware.
- Conversation callbacks: log every event; write audit JSONL to `<workspace>/audit/gate_audit.jsonl`.
- One concrete demo task in a sample workspace (a small ops scenario with clearly risky + clearly safe actions,
  so the gate visibly allows some and blocks others).
- Keyless first: a `--fake-jev` flag using DeterministicFakeJevClient so the whole loop runs without a key.

## Hard rules for both implementers
- Python >= 3.10; stdlib + openhands deps only; no pandas/pytest — unittest.
- Never print/log/persist the API key; source from env var only.
- Tests must run keyless and network-free.
- Do not fabricate live-API results; if unsure of an OpenHands SDK symbol, check
  https://docs.openhands.dev/sdk/guides/custom-tools and the hello-world guide — the confirmed imports are:
  `from openhands.sdk import LLM, Agent, Conversation, Tool`; tools register via `Tool(name=...)` and
  `ToolDefinition[Action, Observation]` subclasses with `.create(conv_state, ...)`, executors subclass
  `ToolExecutor[Action, Observation]` with `__call__(self, action, conversation=None) -> Observation`,
  observations implement `to_llm_content`.
