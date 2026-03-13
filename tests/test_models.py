"""Tests for LoomResponse model validation."""

import pytest

from loom.models import LoomResponse


def test_valid_response() -> None:
    r = LoomResponse(answer="42", confidence_score=0.9, sources=["wiki"])
    assert r.answer == "42"
    assert r.confidence_score == 0.9
    assert r.sources == ["wiki"]


def test_confidence_clamped_above_one() -> None:
    r = LoomResponse(answer="hi", confidence_score=1.5, sources=[])
    assert r.confidence_score == 1.0


def test_confidence_clamped_below_zero() -> None:
    r = LoomResponse(answer="hi", confidence_score=-0.3, sources=[])
    assert r.confidence_score == 0.0


def test_needs_refinement_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loom.config.settings.loom_confidence_threshold", 0.75)
    r = LoomResponse(answer="unsure", confidence_score=0.5, sources=[])
    assert r.needs_refinement is True


def test_needs_refinement_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loom.config.settings.loom_confidence_threshold", 0.75)
    r = LoomResponse(answer="confident", confidence_score=0.9, sources=[])
    assert r.needs_refinement is False
