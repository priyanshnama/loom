"""Tests for the should_continue conditional edge."""

from loom.edges import ROUTE_END, ROUTE_REFINE, should_continue
from loom.models import LoomResponse
from loom.state import LoomState


def _state(**kwargs) -> LoomState:  # type: ignore[no-untyped-def]
    return LoomState(query="test", **kwargs)


def test_no_response_routes_to_refine() -> None:
    assert should_continue(_state()) == ROUTE_REFINE


def test_low_confidence_routes_to_refine(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("loom.config.settings.loom_confidence_threshold", 0.75)
    response = LoomResponse(answer="idk", confidence_score=0.4, sources=[])
    state = _state(response=response, iteration_count=1, max_iterations=3)
    assert should_continue(state) == ROUTE_REFINE


def test_high_confidence_routes_to_end(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("loom.config.settings.loom_confidence_threshold", 0.75)
    response = LoomResponse(answer="sure", confidence_score=0.95, sources=[])
    state = _state(response=response, iteration_count=1, max_iterations=3)
    assert should_continue(state) == ROUTE_END


def test_max_iterations_forces_end(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("loom.config.settings.loom_confidence_threshold", 0.75)
    response = LoomResponse(answer="meh", confidence_score=0.3, sources=[])
    state = _state(response=response, iteration_count=3, max_iterations=3)
    assert should_continue(state) == ROUTE_END
