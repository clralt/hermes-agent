"""Per-agent iteration budget — thread-safe consume/refund counter.

Extracted from ``run_agent.py``.  Each ``AIAgent`` instance (parent or
subagent) holds an :class:`IterationBudget`; the parent's cap comes from
``max_iterations`` (default 500), each subagent's cap comes from
``delegation.max_iterations`` (default 50).

``run_agent`` re-exports ``IterationBudget`` so existing
``from run_agent import IterationBudget`` imports keep working unchanged.
"""

from __future__ import annotations

import threading


class ModelCallBudgetExhausted(RuntimeError):
    """Raised when an outbound model request would exceed the hard cap."""


class ModelCallBudgetStateInvalid(ModelCallBudgetExhausted):
    """Raised when the provider boundary cannot validate the budget state."""


class IterationBudget:
    """Thread-safe iteration counter for an agent.

    Each agent (parent or subagent) gets its own ``IterationBudget``.
    The parent's budget is capped at ``max_iterations`` (default 500).
    Each subagent gets an independent budget capped at
    ``delegation.max_iterations`` (default 50) — this means total
    iterations across parent + subagents can exceed the parent's cap.
    Users control the per-subagent limit via ``delegation.max_iterations``
    in config.yaml.

    ``execute_code`` (programmatic tool calling) iterations are refunded via
    :meth:`refund` so they don't eat into the budget.
    """

    def __init__(self, max_total: int):
        self.max_total = max_total
        self._used = 0
        # Iteration accounting is intentionally separate from model-call
        # accounting.  Logical iterations may be refunded when a request is
        # rebuilt (or when execute_code is the only tool), but an outbound
        # model request has already spent its turn and must remain counted.
        self._model_calls = 0
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """Try to consume one logical iteration.  Returns True if allowed."""
        with self._lock:
            if self._used >= self.max_total:
                return False
            self._used += 1
            return True

    def consume_model_call(self) -> bool:
        """Reserve one outbound model call before it reaches a provider.

        Unlike :meth:`consume`, this reservation is never refunded: retries,
        summaries, continuations, and fallback calls are all real outbound
        requests and must share the same hard cap.
        """
        with self._lock:
            if self._model_calls >= self.max_total:
                return False
            self._model_calls += 1
            return True

    @property
    def model_calls(self) -> int:
        with self._lock:
            return self._model_calls

    @property
    def model_call_remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self._model_calls)

    def refund(self) -> None:
        """Give back one iteration (e.g. for execute_code turns)."""
        with self._lock:
            if self._used > 0:
                self._used -= 1

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self._used)


def consume_model_call_budget(owner) -> bool:
    """Consume one outbound call at the provider boundary.

    The provider boundary is fail-closed: a missing, incomplete, or broken
    budget object must never be treated as an unlimited budget.
    """
    budget = getattr(owner, "model_call_budget", None)
    if budget is None:
        budget = getattr(owner, "iteration_budget", None)
    consume = getattr(budget, "consume_model_call", None)
    remaining = getattr(budget, "model_call_remaining", None)
    if not callable(consume) or not isinstance(remaining, int):
        raise ModelCallBudgetStateInvalid(
            "model-call budget state is missing or invalid"
        )
    try:
        return bool(consume())
    except Exception:
        try:
            owner._model_call_budget_state_invalid = True
        except Exception:
            pass
        return False


def model_call_budget_is_valid(owner) -> bool:
    """Return whether the owner has the complete boundary budget interface."""
    try:
        budget = getattr(owner, "model_call_budget", None)
        if budget is None:
            budget = getattr(owner, "iteration_budget", None)
        return callable(getattr(budget, "consume_model_call", None)) and isinstance(
            getattr(budget, "model_call_remaining", None), int
        )
    except Exception:
        return False


__all__ = [
    "IterationBudget",
    "ModelCallBudgetExhausted",
    "ModelCallBudgetStateInvalid",
    "consume_model_call_budget",
    "model_call_budget_is_valid",
]
