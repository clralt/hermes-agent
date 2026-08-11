"""Deterministic model-call cap regressions.

All provider objects in this module are mocks.  The assertions count attempted
outbound calls, rather than relying on provider responses or live credentials.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.iteration_budget import IterationBudget
from run_agent import AIAgent


def _response(content="done", *, tool_calls=None, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        model="test/model",
        usage=None,
    )


def _tool_call():
    return SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(name="web_search", arguments="{}"),
    )


def _make_agent(tmp_path, *, max_iterations=1):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            session_id="token-cap-regression",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._cached_system_prompt = "stable test prompt"
    agent._session_db = None
    agent._session_json_enabled = False
    agent.save_trajectories = False
    agent.compression_enabled = False
    agent._cleanup_task_resources = lambda *_a, **_kw: None
    agent._save_trajectory = lambda *_a, **_kw: None
    return agent


def test_case_1_normal_model_calls_share_the_hard_cap():
    budget = IterationBudget(3)
    assert [budget.consume_model_call() for _ in range(3)] == [True, True, True]
    assert budget.model_calls == 3
    assert budget.model_call_remaining == 0


def test_case_2_exact_exhaustion_denies_the_next_model_call():
    budget = IterationBudget(2)
    assert budget.consume_model_call() is True
    assert budget.consume_model_call() is True
    assert budget.consume_model_call() is False
    assert budget.model_calls == 2


def test_case_3_exhaustion_does_not_add_a_full_context_summary_call(tmp_path):
    agent = _make_agent(tmp_path, max_iterations=1)
    agent.valid_tool_names = ["web_search"]
    responses = iter([
        _response("", tool_calls=[_tool_call()], finish_reason="tool_calls"),
        _response("done"),
    ])
    main_calls = []
    agent._interruptible_api_call = lambda _kwargs: (
        main_calls.append(1) or next(responses)
    )

    def fake_execute_tool_calls(_assistant_message, messages, *_args):
        messages.append({"role": "tool", "tool_call_id": "call-1", "content": "ok"})
        # The pre-fix loop admitted a grace call after max_iterations.
        setattr(agent, "_budget_grace_call", True)

    setattr(agent, "_execute_tool_calls", fake_execute_tool_calls)

    with (
        patch("hermes_cli.plugins.has_hook", return_value=False),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("inspect /tmp/project")

    assert len(main_calls) + agent.client.chat.completions.create.call_count == 1
    assert agent.model_call_budget.model_calls == 1
    assert result["completed"] is False


def test_case_4_grace_flag_cannot_bypass_the_model_call_cap(tmp_path):
    agent = _make_agent(tmp_path, max_iterations=1)
    agent._budget_grace_call = True
    calls = []
    agent._interruptible_api_call = lambda _kwargs: (
        calls.append(1) or _response("done")
    )

    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = agent.run_conversation("finish")

    assert calls == [1]
    assert agent.model_call_budget.model_calls == 1
    assert result["final_response"] == "done"


def test_case_5_finalizer_overflow_is_denied_before_provider_call(tmp_path):
    agent = _make_agent(tmp_path, max_iterations=1)
    assert agent.model_call_budget.consume_model_call() is True
    agent.client.chat.completions.create.return_value = _response("must not be used")

    result = agent._handle_max_iterations([{"role": "user", "content": "task"}], 1)

    assert agent.client.chat.completions.create.call_count == 0
    assert agent.model_call_budget.model_calls == 1
    assert "before another model call" in result


def test_case_6_summary_helper_consumes_one_call_at_its_boundary(tmp_path):
    agent = _make_agent(tmp_path, max_iterations=2)
    agent.client.chat.completions.create.return_value = _response("summary")

    result = agent._handle_max_iterations([{"role": "user", "content": "task"}], 2)

    assert result == "summary"
    assert agent.client.chat.completions.create.call_count == 1
    assert agent.model_call_budget.model_calls == 1


def test_case_7_empty_summary_cannot_retry_past_the_cap(tmp_path):
    agent = _make_agent(tmp_path, max_iterations=1)
    agent.client.chat.completions.create.return_value = _response("")

    result = agent._handle_max_iterations([{"role": "user", "content": "task"}], 1)

    assert agent.client.chat.completions.create.call_count == 1
    assert agent.model_call_budget.model_calls == 1
    assert "before another model call" in result


def test_case_8_verification_candidate_does_not_trigger_summary_call(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, max_iterations=1)
    agent._interruptible_api_call = lambda _kwargs: _response("composed report")
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    agent._turn_file_mutation_paths = {"changed.py"}
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")

    with (
        patch("agent.verification_stop.build_verify_on_stop_nudge", return_value="verify it"),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("edit changed.py")

    assert result["final_response"] == "composed report"
    assert agent.model_call_budget.model_calls == 1
    agent._handle_max_iterations.assert_not_called()


def test_case_9_verification_continuation_stops_at_exact_call_cap(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, max_iterations=2)
    responses = iter([_response("candidate one"), _response("candidate two")])
    agent._interruptible_api_call = lambda _kwargs: next(responses)
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")

    with (
        patch(
            "agent.verification_stop.build_verify_on_stop_nudge",
            side_effect=["verify one", "verify two"],
        ),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("edit changed.py")

    assert agent.model_call_budget.model_calls == 2
    assert agent._handle_max_iterations.call_count == 0
    assert result["completed"] is False
