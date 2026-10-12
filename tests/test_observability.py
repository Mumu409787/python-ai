from contextlib import contextmanager

import pytest

from python_ai import observability as obs


def test_exporter_setup_failure_does_not_break_business(monkeypatch):
    def fail():
        raise RuntimeError("exporter unavailable")

    monkeypatch.setattr(obs, "tracing_client", fail)
    with obs.observation("test") as span:
        assert span.trace_id is None
    with pytest.raises(ValueError, match="business error"), obs.observation("test"):
        raise ValueError("business error")


async def test_traced_function_preserves_business_exception(monkeypatch):
    events = []

    @contextmanager
    def context(*a, **kw):
        events.append("start")
        try:
            yield
        finally:
            events.append("end")

    monkeypatch.setattr(obs, "observation", context)

    @obs.traced("test")
    async def fail():
        raise ValueError("business error")

    with pytest.raises(ValueError):
        await fail()
    assert events == ["start", "end"]
