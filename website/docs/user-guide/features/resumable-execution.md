---
sidebar_position: 14
title: "Resumable Execution"
description: "Durable automatic continuation across bounded agent invocations"
---

# Resumable Kanban execution

Hermes treats an agent iteration limit as a **yield point**, not a task outcome. Dispatcher-owned work is re-run from durable project and board state until it reaches a governed terminal state.

## Lifecycle

```text
authorized task
  -> bounded worker run
  -> complete? yes: validate / audit / close
  -> complete? no: immutable checkpoint + atomic continuation requeue
  -> clean worker verifies checkpoint bindings and resumes
```

The checkpoint and queue transition share one SQLite transaction. A crash cannot leave a checkpoint without an eligible task or an eligible continuation without its checkpoint. Replaying the same yield is idempotent on task, source run, and canonical checkpoint hash.

Continuations are selected before unrelated fresh work. Only one pending continuation can exist for an exclusive task. Process restart recovery promotes a rich phase checkpoint left by an interrupted worker without consuming the task failure breaker.

## Checkpoint contents

Workers use `kanban_phase_checkpoint` after material phase progress and before long operations. The JSON contract records:

- task identity, version, objective, and authorized scope;
- execution role and worker identity/model/reasoning effort;
- phase and phase status;
- base commit/tree, live Git state, an exact synthetic candidate tree, and a durable content-addressed snapshot manifest/object set;
- dossier, contract, and evidence hashes when governed work supplies them;
- completed and remaining steps;
- artifacts, validation, audit state, unresolved findings, decisions, and external actions;
- exact continuation instruction, retries, timestamps, and source-run provenance.

If no rich phase checkpoint exists when the cap is reached, Hermes writes a conservative emergency checkpoint. It never guesses completed work: the next worker must reconcile durable artifacts before proceeding.

## Restoration and divergence

Before spawning a continuation, Hermes validates the checkpoint and verifies the bound repository state. If bound tracked or untracked bytes drift, Hermes restores them from the content-addressed checkpoint before releasing the worker launch gate, then verifies the complete binding again. The gate is an inherited anonymous-pipe capability and is released only after the worker PID is durably registered to the exact run/claim. A new unbound untracked path—or a directory collision containing ignored/unbound data—is never deleted automatically: it fails closed for reconciliation.

The default inclusion policy binds tracked files and ordinary untracked files. Ignored files are excluded unless the controller supplies a JSON string array through `HERMES_RESUMABLE_IGNORED_PATHS`; those paths are recorded in `repository_inclusion_policy.explicit_ignored_paths` and included in both the synthetic Git tree and the restorable object manifest. Verification and continuation reuse the frozen list from the checkpoint rather than rereading mutable configuration. Empty directories and special files are outside the contract.

## Authority separation

Checkpoint validation preserves role boundaries across runs:

- independent audit continuations require reviewer identity `codex-high-independent-auditor`, model `gpt-5.6-sol`, reasoning effort `high`, a fresh invocation, and read-only/non-producer provenance. Their tool schema and dispatch boundary allow only repository reads plus narrow checkpoint/heartbeat channels—no shell, writes, network, task completion, or general database mutation;
- a closer requires a CLEAN audit and may close only when the candidate tree equals the audited candidate tree. Closer model processes are also read-only; the external controller applies signed authority;
- tasks marked `requires_independent_audit` reject ordinary `kanban_complete`. Completion requires a schema-v2 detached Ed25519 verdict that binds the authority task, audit task/checkpoint, candidate tree, task contract, evidence manifest, dossier, reviewer/model/effort, producer identity, freshness window, and signer identity;
- the signer must exist in an operator-pinned public trust store. Configure `kanban.audit_trust_store_path` as an absolute regular-file path and `kanban.audit_trust_store_sha256` as the expected digest. On POSIX the file and every parent directory must be operator-owned and non-writable by the worker identity; a worker-owned file cannot repin authority by replacing both configuration values. Revoked, stale, mis-scoped, replay-conflicting, unsigned, or misbound verdicts fail closed;
- legacy rows whose audit requirement predates the classification column remain `UNKNOWN` and cannot complete until explicitly classified;
- implementation workers cannot mark their own governed candidate complete through either the resumable checkpoint path or the legacy completion tool.

Controllers can label created cards `fresh`, `remediation`, or `audit`; dispatcher-owned yields change the same exclusive card to `continuation`. Dependency-blocked work remains distinguishable through its typed block/status state. Continuations outrank unrelated fresh work, and retry-safe creation plus the pending-continuation uniqueness constraint prevent duplicate cards and active successors.

A cap is never approval, rejection, completion, or an operator exception.

## Terminal states

Supported governed terminal states are:

- `COMPLETED_CLEAN`
- `KILLED_BY_EVIDENCE`
- `BLOCKED_EXTERNAL_DEPENDENCY`
- `BLOCKED_OPERATOR_EXCEPTION`
- `FAILED_RECOVERABLE` (normally requeued for autonomous remediation)

`ITERATION_CAP_REACHED` is intentionally not a terminal state. Spending, secrets, consequential external actions, human-only release authority, and irreversible important-data mutation remain operator exceptions; phase boundaries, retries, validation, audits, and remediation do not become new manual gates.

## Observability

Events record `phase_checkpoint_saved`, `continuation_enqueued`, `continuation_resumed`, and the final `terminal_state`. Each continuation event includes its checkpoint identifier/hash, reason, number, previous phase, completed work, and remaining work. Accepted signed verdict identities and canonical hashes are stored in a uniqueness-constrained replay ledger before terminal intent is written.
