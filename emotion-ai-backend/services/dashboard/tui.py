"""
services/dashboard/tui.py
─────────────────────────
Module 7 — Local Observability & Demo Dashboard (Rich Terminal UI)

A full-screen terminal UI that shows:
  • Active session ID and VAD state (Silence / Speaking)
  • Latest fast-path emotion with confidence bar
  • Groq API usage counter and daily quota
  • Last utterance transcript and judge verdict with reasoning
  • System log tail (last 10 entries)

Usage:
  python -m services.dashboard.tui
  # Or via Makefile:  make dashboard

The TUI polls the local Orchestrator REST API every 500ms.
It works in MOCK_APIS=true mode (uses the same polling endpoint).
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.config import get_settings

cfg = get_settings()

# ── Rich imports ──────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn
    from rich.table import Table
    from rich.text import Text
    from rich import box
except ImportError:
    print("Rich not installed. Run: pip install rich")
    sys.exit(1)

console = Console()

# ── Colour Palette ────────────────────────────────────────────────────────────
EMOTION_COLORS = {
    "Neutral": "grey70",
    "Happy": "bright_yellow",
    "Sad": "dodger_blue1",
    "Angry": "bright_red",
    "Surprised": "bright_magenta",
    "Unknown": "grey42",
}

VAD_COLORS = {
    "silence": "grey50",
    "speech_start": "yellow",
    "speaking": "bright_green",
    "speech_end": "orange3",
}

# ── Dashboard State ───────────────────────────────────────────────────────────

class DashboardState:
    session_id: str = "—"
    vad_state: str = "silence"
    fast_path_emotion: str = "Unknown"
    fast_path_confidence: float = 0.0
    fast_path_window: int = 0
    groq_used: int = 0
    groq_limit: int = 14_400
    mock_mode: bool = False
    last_transcript: str = ""
    last_verdict_emotion: str = "—"
    last_verdict_confidence: float = 0.0
    last_verdict_reasoning: str = ""
    active_sessions: int = 0
    log_lines: list[str] = []
    last_updated: float = 0.0

    def push_log(self, line: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log_lines.append(f"[grey50]{ts}[/grey50] {line}")
        if len(self.log_lines) > 15:
            self.log_lines.pop(0)


_state = DashboardState()

# ── Custom Log Handler ────────────────────────────────────────────────────────

class _DashboardLogHandler(logging.Handler):
    """Captures log records and injects them into the dashboard state."""

    def emit(self, record: logging.LogRecord) -> None:
        level = record.levelname
        color_map = {"WARNING": "yellow", "ERROR": "red", "INFO": "white", "DEBUG": "grey50"}
        color = color_map.get(level, "white")
        msg = self.format(record)
        _state.push_log(f"[{color}]{msg[:90]}[/{color}]")


# ── Layout Builder ─────────────────────────────────────────────────────────────

def _build_header() -> Panel:
    mock_badge = " [bold red]⚠ MOCK MODE[/bold red]" if _state.mock_mode else ""
    title = Text(f"🧠 Real-Time Emotion Intelligence{mock_badge}", style="bold cyan", justify="center")
    subtitle = Text(f"Session: {_state.session_id}", style="grey70", justify="center")
    return Panel(Text.assemble(title, "\n", subtitle), box=box.DOUBLE_EDGE, style="bright_cyan")


def _build_vad_panel() -> Panel:
    state_str = _state.vad_state.upper()
    color = VAD_COLORS.get(_state.vad_state, "white")
    icon = "🔴" if _state.vad_state == "silence" else "🟢"
    content = Text(f"{icon}  {state_str}", style=f"bold {color}", justify="center")
    return Panel(content, title="[bold]VAD State[/bold]", border_style=color)


def _build_fast_path_panel() -> Panel:
    emotion = _state.fast_path_emotion
    color = EMOTION_COLORS.get(emotion, "white")
    conf_pct = int(_state.fast_path_confidence * 100)
    bar_filled = "█" * (conf_pct // 5)
    bar_empty = "░" * (20 - len(bar_filled))

    lines = [
        Text(f"  {emotion}", style=f"bold {color}"),
        Text(f"  [{color}]{bar_filled}{bar_empty}[/{color}] {conf_pct}%  (window #{_state.fast_path_window})"),
    ]
    content = Text.assemble(*lines)
    return Panel(content, title="[bold]Fast-Path Emotion (500ms)[/bold]", border_style="bright_blue")


def _build_groq_panel() -> Panel:
    used = _state.groq_used
    limit = _state.groq_limit
    pct = (used / limit * 100) if limit > 0 else 0
    bar_len = 30
    filled = int(bar_len * pct / 100)
    empty = bar_len - filled
    color = "green" if pct < 70 else ("yellow" if pct < 90 else "red")
    bar = f"[{color}]{'█' * filled}[/{color}]{'░' * empty}"
    content = Text.from_markup(
        f"  Requests: {used:,} / {limit:,}\n"
        f"  {bar}  {pct:.1f}%\n"
        f"  {'⚠ HIGH USAGE' if pct >= 80 else 'OK'}"
    )
    return Panel(content, title="[bold]Groq Daily Quota[/bold]", border_style="yellow")


def _build_verdict_panel() -> Panel:
    color = EMOTION_COLORS.get(_state.last_verdict_emotion, "white")
    reasoning = (_state.last_verdict_reasoning[:120] + "…") if len(_state.last_verdict_reasoning) > 120 else _state.last_verdict_reasoning
    transcript = (_state.last_transcript[:100] + "…") if len(_state.last_transcript) > 100 else _state.last_transcript

    content = Text.from_markup(
        f"  [grey70]Transcript:[/grey70]  [italic]{transcript or '(waiting…)'}[/italic]\n\n"
        f"  [grey70]Verdict:   [/grey70]  [{color}][bold]{_state.last_verdict_emotion}[/bold][/{color}]"
        f"  ({int(_state.last_verdict_confidence * 100)}%)\n\n"
        f"  [grey70]Reasoning: [/grey70]  {reasoning or '(waiting…)'}"
    )
    return Panel(content, title="[bold]Last Utterance — Final Verdict[/bold]", border_style="bright_magenta")


def _build_log_panel() -> Panel:
    lines = _state.log_lines[-12:]
    content = Text.from_markup("\n".join(lines) if lines else "  [grey50](no log entries yet)[/grey50]")
    return Panel(content, title="[bold]System Log[/bold]", border_style="grey50")


def _build_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="middle", ratio=1),
        Layout(name="verdict", size=8),
        Layout(name="logs", size=14),
    )
    layout["middle"].split_row(
        Layout(name="vad", ratio=1),
        Layout(name="fast_path", ratio=2),
        Layout(name="groq", ratio=2),
    )
    layout["header"].update(_build_header())
    layout["middle"]["vad"].update(_build_vad_panel())
    layout["middle"]["fast_path"].update(_build_fast_path_panel())
    layout["middle"]["groq"].update(_build_groq_panel())
    layout["verdict"].update(_build_verdict_panel())
    layout["logs"].update(_build_log_panel())
    return layout


# ── Polling ───────────────────────────────────────────────────────────────────

async def _poll_orchestrator(session_id: Optional[str] = None) -> None:
    """Poll the Orchestrator API and update dashboard state."""
    import aiohttp

    base = f"http://localhost:{cfg.ORCHESTRATOR_PORT}"

    while True:
        try:
            async with aiohttp.ClientSession() as client:
                # Health check
                async with client.get(f"{base}/health", timeout=aiohttp.ClientTimeout(total=1.0)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        _state.groq_used = data.get("groq_requests_today", 0)
                        _state.groq_limit = data.get("groq_daily_limit", 14_400)
                        _state.mock_mode = data.get("mock_mode", False)
                        _state.active_sessions = data.get("active_sessions", 0)

                # Session status
                if session_id:
                    async with client.get(
                        f"{base}/session/{session_id}/status",
                        timeout=aiohttp.ClientTimeout(total=1.0),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            _state.vad_state = data.get("vad_state", "silence")
                            last = data.get("latest_emotion")
                            if last:
                                _state.fast_path_emotion = last.get("emotion", "Unknown")
                                _state.fast_path_confidence = last.get("confidence", 0.0)
                                _state.fast_path_window = last.get("window_index", 0)
                            _state.last_updated = time.time()

        except Exception as exc:
            _state.push_log(f"[red]Poll error: {exc}[/red]")

        await asyncio.sleep(0.5)


# ── Main Entry Point ──────────────────────────────────────────────────────────

async def _run_dashboard(session_id: Optional[str] = None) -> None:
    _state.session_id = session_id or "—"

    # Attach log handler
    handler = _DashboardLogHandler()
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logging.root.addHandler(handler)

    # Start polling in background
    poll_task = asyncio.create_task(_poll_orchestrator(session_id))

    try:
        with Live(console=console, refresh_per_second=4, screen=True) as live:
            while True:
                live.update(_build_layout())
                await asyncio.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        poll_task.cancel()
        console.print("\n[bold cyan]Dashboard closed.[/bold cyan]")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Emotion AI Dashboard")
    parser.add_argument("--session", help="Session ID to monitor", default=None)
    args = parser.parse_args()

    asyncio.run(_run_dashboard(session_id=args.session))


if __name__ == "__main__":
    main()
