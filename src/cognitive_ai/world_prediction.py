"""Forecast & compare over simulated states (Phase M4-F).

Orchestration only: action simulation comes from world_dynamics,
per-state reasoning from world_reasoning. No duplicated search, no
unification, no IO, no persistence, no learning. Nothing here claims
reality: forecasts describe what an action model derives, comparisons
describe agreement with an independently supplied reference state.
"""
from __future__ import annotations

from typing import Any, Optional

from . import world_dynamics as _dynamics
from . import world_reasoning as _reasoning
from .models import (WorldActionSchema, WorldForecast,
                     WorldLiteralPattern, WorldPredictionComparison,
                     WorldPredictionComparisonItem, WorldReasoningResult,
                     WorldState, WorldTransitionResult,
                     WorldComparisonOutcome, WorldInferenceRule,
                     WorldRuleView)
from .policy import MAX_CHAIN_DEPTH

CONSISTENT = WorldComparisonOutcome.CONSISTENT.value
MISMATCH = WorldComparisonOutcome.MISMATCH.value
REFERENCE_UNKNOWN = WorldComparisonOutcome.REFERENCE_UNKNOWN.value
CONFLICTED = WorldComparisonOutcome.CONFLICTED.value
LIMITED = WorldComparisonOutcome.LIMITED.value
INCONCLUSIVE = WorldComparisonOutcome.INCONCLUSIVE.value
INVALID = WorldComparisonOutcome.INVALID.value

# Fixed deterministic aggregation priority (strongest signal first).
_OVERALL_PRIORITY = (INVALID, CONFLICTED, MISMATCH, LIMITED,
                     REFERENCE_UNKNOWN, INCONCLUSIVE, CONSISTENT)


def _valid_rules(rules) -> Optional[str]:
    if rules is None:
        return "rules missing"
    try:
        items = list(rules)
    except TypeError:
        return "rules not iterable"
    for r in items:
        if not isinstance(r, (WorldInferenceRule, WorldRuleView, dict)):
            return f"bad rule entry {r!r}"
    return None


def forecast_action(start_state: WorldState, action: WorldActionSchema,
                    bindings: dict, queries,
                    rules, max_depth: Optional[int] = None,
                    new_state_id: str = "",
                    event_id: str = "") -> WorldForecast:
    """Simulate, then reason about each query in the predicted state."""
    if max_depth is None:
        max_depth = MAX_CHAIN_DEPTH
    action_id = action.action_id if isinstance(action,
                                               WorldActionSchema) else ""
    start_id = start_state.state_id if isinstance(start_state,
                                                  WorldState) else ""
    bound = tuple(sorted(bindings.items(), key=lambda kv: kv[0])) \
        if isinstance(bindings, dict) else ()
    if not isinstance(start_state, WorldState) or \
            not isinstance(action, WorldActionSchema) or \
            not isinstance(bindings, dict):
        return WorldForecast(
            start_state_id=start_id, action_id=action_id, bindings=bound,
            transition=WorldTransitionResult(
                outcome="invalid",
                reason="start state, action or bindings invalid"),
            complete=False,
            diagnostics=("invalid forecast input",))
    if not isinstance(queries, (list, tuple)) or not queries or \
            any(not isinstance(q, WorldLiteralPattern) for q in queries):
        return WorldForecast(
            start_state_id=start_id, action_id=action_id, bindings=bound,
            transition=WorldTransitionResult(
                outcome="invalid", reason="queries invalid"),
            complete=False,
            diagnostics=("invalid forecast input",))
    rules_problem = _valid_rules(rules)
    if rules_problem is not None:
        return WorldForecast(
            start_state_id=start_id, action_id=action_id, bindings=bound,
            transition=WorldTransitionResult(
                outcome="invalid", reason=rules_problem),
            complete=False,
            diagnostics=("invalid forecast input",))
    transition = _dynamics.simulate_action(start_state, action, bindings,
                                             new_state_id, event_id)
    if transition.outcome != "applied" or transition.new_state is None:
        return WorldForecast(
            start_state_id=start_id, action_id=action_id, bindings=bound,
            transition=transition, predicted_state=None,
            queries=tuple(queries), results=(), complete=False,
            diagnostics=(f"no forecast: {transition.outcome}: "
                         f"{transition.reason}",))
    predicted = transition.new_state
    results = tuple(_reasoning.reason_about_state(predicted, q, rules,
                                                  max_depth)
                    for q in queries)
    complete = all(r.complete for r in results)
    diagnostics = [transition.reason] if transition.reason else []
    for i, r in enumerate(results):
        if not r.complete:
            diagnostics.append(
                f"query[{i}] {r.verdict} (complete=False)")
    return WorldForecast(
        start_state_id=start_id, action_id=action_id, bindings=bound,
        transition=transition, predicted_state=predicted,
        queries=tuple(queries), results=results, complete=complete,
        diagnostics=tuple(diagnostics))


def _classify(forecast_verdict: str, reference_verdict: str) -> tuple:
    if forecast_verdict == "invalid" or reference_verdict == "invalid":
        return INVALID, "invalid input or query on some side"
    if forecast_verdict == "conflicted" or \
            reference_verdict == "conflicted":
        return CONFLICTED, "a side derives both polarities"
    if {forecast_verdict, reference_verdict} == {"entailed", "refuted"}:
        return MISMATCH, \
            f"forecast {forecast_verdict} vs reference {reference_verdict}"
    if forecast_verdict == "limited" or reference_verdict == "limited":
        return LIMITED, "a side hit a reasoning limit"
    if forecast_verdict == "unknown":
        return INCONCLUSIVE, "forecast derives nothing to compare"
    if reference_verdict == "unknown":
        return REFERENCE_UNKNOWN, \
            "reference state derives nothing for this query"
    return CONSISTENT, f"both {forecast_verdict}"


def compare_forecast(forecast: WorldForecast,
                     reference_state: WorldState, rules,
                     max_depth: Optional[int] = None) -> Any:
    """Compare each forecast query against an independent reference."""
    if max_depth is None:
        max_depth = MAX_CHAIN_DEPTH
    if not isinstance(forecast, WorldForecast) or \
            not isinstance(reference_state, WorldState):
        return WorldPredictionComparison(
            overall=INVALID, complete=False,
            diagnostics=("forecast or reference state invalid",))
    if forecast.predicted_state is None or not forecast.queries:
        return WorldPredictionComparison(
            start_state_id=forecast.start_state_id,
            forecast_state_id="",
            reference_state_id=reference_state.state_id,
            overall=INVALID, complete=False,
            diagnostics=("forecast has no predicted state or queries",))
    if reference_state.state_id == forecast.predicted_state.state_id:
        return WorldPredictionComparison(
            start_state_id=forecast.start_state_id,
            forecast_state_id=forecast.predicted_state.state_id,
            reference_state_id=reference_state.state_id,
            overall=INVALID, complete=False,
            diagnostics=("reference must differ from the predicted state",
                        ))
    rules_problem = _valid_rules(rules)
    if rules_problem is not None:
        return WorldPredictionComparison(
            start_state_id=forecast.start_state_id,
            forecast_state_id=forecast.predicted_state.state_id,
            reference_state_id=reference_state.state_id,
            overall=INVALID, complete=False,
            diagnostics=(rules_problem,))
    items = []
    for q, fres in zip(forecast.queries, forecast.results):
        rres = _reasoning.reason_about_state(reference_state, q, rules,
                                             max_depth)
        category, detail = _classify(fres.verdict, rres.verdict)
        items.append(WorldPredictionComparisonItem(
            query=q, forecast_verdict=fres.verdict,
            reference_verdict=rres.verdict, category=category,
            detail=detail,
            forecast_proofs=fres.positive_proofs + fres.negative_proofs,
            reference_proofs=rres.positive_proofs + rres.negative_proofs))
    present = {it.category for it in items}
    overall = next(c for c in _OVERALL_PRIORITY if c in present)
    complete = overall == CONSISTENT and forecast.complete and all(
        it.category == CONSISTENT for it in items)
    return WorldPredictionComparison(
        start_state_id=forecast.start_state_id,
        forecast_state_id=forecast.predicted_state.state_id,
        reference_state_id=reference_state.state_id,
        items=tuple(items), overall=overall, complete=complete,
        diagnostics=tuple(forecast.diagnostics))
