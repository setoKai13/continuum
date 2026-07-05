"""Trace stream: the debug console's data source must be dependable."""

from __future__ import annotations

from trace import NullTracer, Tracer


def test_tracer_writes_tagged_single_line_events(tmp_path) -> None:
    path = tmp_path / "trace.log"
    tracer = Tracer(path)
    tracer.event("HEARD", "ouvre  Notes\net écris bonjour")  # whitespace + newline
    tracer.event("THINK", "x" * 2000)  # oversized reasoning blob
    tracer.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, "one event = exactly one line, always"
    assert "| HEARD" in lines[0] and "ouvre Notes et écris bonjour" in lines[0]
    assert len(lines[1]) < 700, "reasoning blobs are capped"


def test_tracer_truncates_previous_session(tmp_path) -> None:
    path = tmp_path / "trace.log"
    path.write_text("old session noise\n", encoding="utf-8")
    tracer = Tracer(path)
    tracer.event("BOOT", "new run")
    tracer.close()

    content = path.read_text(encoding="utf-8")
    assert "old session" not in content, "the debug window narrates THIS run only"
    assert "new run" in content


def test_null_tracer_is_inert() -> None:
    tracer = NullTracer()
    tracer.event("ANY", "thing")  # must not raise
    tracer.close()
