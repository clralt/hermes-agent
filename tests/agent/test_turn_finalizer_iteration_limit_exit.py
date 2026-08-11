"""Regression tests for iteration-limit exit normalization (#61631)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.turn_finalizer import finalize_turn


class _LimitAgent:
    def __init__(
        self,
        *,
        max_iterations=60,
        budget_remaining=0,
        completion_explainer=False,
    ):
        self.max_iterations = max_iterations
        self.iteration_budget = SimpleNamespace(
            remaining=budget_remaining, used=max_iterations, max_total=max_iterations
        )
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = "sess-test"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.persisted_messages = None
        self._handle_max_iterations_called = False
        self._completion_explainer = completion_explainer
        self.extension_calls = []

    def _handle_max_iterations(self, messages, api_call_count):
        self._handle_max_iterations_called = True
        return "summary from extra call"

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = list(messages)

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return self._completion_explainer

    def _format_turn_completion_explanation(self, _reason):
        return "iteration-limit explanation"

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        self.extension_calls.append("external_memory")

    def _spawn_background_review(self, **_kwargs):
        self.extension_calls.append("background_review")


def _finalize(
    agent,
    *,
    final_response,
    exit_reason,
    api_call_count=60,
    pending_verification_response=None,
):
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason=exit_reason,
        _pending_verification_response=pending_verification_response,
    )


@pytest.mark.parametrize("role", ["auditor", "closer"])
def test_locked_execution_role_suppresses_entire_extension_lifecycle(
    monkeypatch, role
):
    agent = _LimitAgent(max_iterations=60, budget_remaining=1)
    agent.context_compressor = SimpleNamespace(
        last_prompt_tokens=0,
        _micro_compact_enabled=True,
        _micro_compact=lambda messages: (
            agent.extension_calls.append("micro_compact") or messages
        ),
    )
    agent._skill_nudge_interval = 1
    agent._iters_since_skill = 1
    agent.valid_tool_names = ["skill_manage"]
    monkeypatch.setenv("HERMES_EXECUTION_ROLE", role)

    def invoke_hook(name, **_kwargs):
        agent.extension_calls.append(name)
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke_hook)
    monkeypatch.setattr(
        "agent.conversation_loop._notify_context_engine_turn_complete",
        lambda *_args, **_kwargs: agent.extension_calls.append("context_engine"),
    )

    _finalize(
        agent,
        final_response="done",
        exit_reason="text_response(4 chars)",
        api_call_count=1,
    )

    assert agent.extension_calls == []


def test_normal_execution_role_runs_expected_extension_lifecycle(monkeypatch):
    agent = _LimitAgent(max_iterations=60, budget_remaining=1)
    agent.context_compressor = SimpleNamespace(
        last_prompt_tokens=0,
        _micro_compact_enabled=True,
        _micro_compact=lambda messages: (
            agent.extension_calls.append("micro_compact") or messages
        ),
    )
    agent._skill_nudge_interval = 1
    agent._iters_since_skill = 1
    agent.valid_tool_names = ["skill_manage"]
    monkeypatch.delenv("HERMES_EXECUTION_ROLE", raising=False)

    def invoke_hook(name, **_kwargs):
        agent.extension_calls.append(name)
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke_hook)
    monkeypatch.setattr(
        "agent.conversation_loop._notify_context_engine_turn_complete",
        lambda *_args, **_kwargs: agent.extension_calls.append("context_engine"),
    )

    _finalize(
        agent,
        final_response="done",
        exit_reason="text_response(4 chars)",
        api_call_count=1,
    )

    assert agent.extension_calls == [
        "micro_compact",
        "transform_llm_output",
        "post_llm_call",
        "context_engine",
        "external_memory",
        "background_review",
        "on_session_end",
    ]
















@pytest.mark.parametrize(
    ("exit_reason", "interrupted", "failed"),
    [
        ("interrupted_by_user", True, False),
        ("all_retries_exhausted_no_response", False, False),
        ("provider_failure", False, True),
    ],
)
def test_pending_response_does_not_mask_later_terminal_exit(
    monkeypatch, exit_reason, interrupted, failed
):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()

    result = finalize_turn(
        agent,
        final_response=None,
        api_call_count=60,
        interrupted=interrupted,
        failed=failed,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason=exit_reason,
        _pending_verification_response="stale premature report",
    )

    assert result["final_response"] is None
    assert result["turn_exit_reason"] == exit_reason
    assert result["completed"] is False
    assert agent._handle_max_iterations_called is False


def test_pending_response_records_kanban_timeout(monkeypatch):
    """B-004: iteration cap must enqueue a durable continuation checkpoint.

    An ordinary iteration cap is a normal task timeslice, not a failure. The
    dispatcher-owned worker must persist a complete machine-readable checkpoint
    and atomically requeue the task for continuation instead of recording a
    failure or retrying as a protocol violation.

    Failure mechanism on pre-B-004 code: no checkpoint was built; the task was
    retried as a protocol violation or had _record_task_failure called on it
    rather than yield_task_for_continuation.
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "99")
    conn = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("hermes_cli.kanban_db.connect", lambda: conn)

    from hermes_cli.kanban_db import Task
    fake_task = Task(
        id="task-123",
        title="test task",
        body=None,
        assignee="coder",
        status="running",
        priority=0,
        created_by=None,
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path="/tmp",
        claim_lock=None,
        claim_expires=None,
        tenant=None,
        current_run_id=99,
    )
    monkeypatch.setattr("hermes_cli.kanban_db.get_task", lambda _conn, _tid: fake_task)
    fake_cp = {"task_id": "task-123", "schema_version": 1, "worker": {}}
    monkeypatch.setattr(
        "hermes_cli.kanban_db.get_saved_phase_checkpoint",
        lambda *_a, **_kw: fake_cp,
    )
    yield_mock = MagicMock(name="yield_task_for_continuation")
    yield_mock.return_value = SimpleNamespace(continuation_number=1, checkpoint_sha256="a" * 64)
    monkeypatch.setattr("hermes_cli.kanban_db.yield_task_for_continuation", yield_mock)

    agent = _LimitAgent()
    result = _finalize(
        agent,
        final_response=None,
        exit_reason="unknown",
        pending_verification_response="composed report",
    )

    assert result["turn_exit_reason"] == "max_iterations_reached(60/60)"
    yield_mock.assert_called_once_with(
        conn,
        "task-123",
        checkpoint=fake_cp,
        expected_run_id=99,
        reason="ITERATION_CAP_REACHED",
    )


def test_published_pending_candidate_is_not_duplicated_by_finalizer(monkeypatch):
    """When budget exhaustion preserves a verification candidate that is
    already the tail assistant message, the finalizer must NOT append a
    duplicate. The content-comparison guard prevents this. (#65919 §7)
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()
    report = "the composed report"

    result = finalize_turn(
        agent,
        final_response=report,
        api_call_count=60,
        interrupted=False,
        failed=False,
        # The candidate is already in messages as the tail assistant.
        messages=[
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": report},
        ],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="unknown",
        _pending_verification_response=report,
    )

    # The tail assistant already matches final_response — no duplicate appended.
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["user", "assistant"]
    # Persisted messages should also have no duplicate.
    assert agent.persisted_messages is not None
    persisted_roles = [m["role"] for m in agent.persisted_messages]
    assert persisted_roles == ["user", "assistant"]


