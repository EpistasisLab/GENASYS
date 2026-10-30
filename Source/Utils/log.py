"""Tiny shared logging and formatting helpers.

Kept in its own module so ``model_config``, ``credentials``, the
topology checks and the EA's reporting can share one warning and one duration
format without importing one another.
"""

from __future__ import annotations


def warn(message: str, source: str = "model_config") -> None:
    print(f"    [{source}] WARNING: {message}", flush=True)


def format_duration(seconds: float) -> str:
    """``'1523.4s (25m 23s)'``: raw seconds plus a readable breakdown."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(round(seconds)), 60)
    hours, mins = divmod(mins, 60)
    human = f"{hours}h {mins}m {secs}s" if hours else f"{mins}m {secs}s"
    return f"{seconds:.1f}s ({human})"
