"""Times of day, and durations in the units people say them in.

The config stores a recording length in seconds and intervals in minutes.
These helpers turn those into "2 h" or "once a day at 10:00" for display, and
back from a number and a unit typed into a form. No GUI code here.
"""

from __future__ import annotations

import re
from typing import Any, Tuple

MINUTES_PER_DAY = 24 * 60
DEFAULT_START_TIME = "00:00"

# Seconds per unit, for a recording length.
DURATION_UNITS = (("s", 1.0), ("min", 60.0), ("h", 3600.0))
# Minutes per unit, for an interval between recordings.
INTERVAL_UNITS = (("min", 1.0), ("h", 60.0), ("day", float(MINUTES_PER_DAY)))

_TIME_OF_DAY = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")


def parse_time_of_day(value: Any) -> int:
    """'10:30' -> 630 (minutes after midnight). Raises ``ValueError`` if it is not a time."""
    match = _TIME_OF_DAY.match(str(value if value is not None else ""))
    if not match:
        raise ValueError(f"expected a 24-hour time such as 09:30, got: {value!r}")
    hours, minutes = int(match.group(1)), int(match.group(2))
    if hours > 23 or minutes > 59:
        raise ValueError(f"expected a 24-hour time such as 09:30, got: {value!r}")
    return hours * 60 + minutes


def format_time_of_day(minutes: int) -> str:
    minutes = int(minutes) % MINUTES_PER_DAY
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def start_minutes(config: dict) -> int:
    """Minutes after midnight that a config's schedule is counted from; 0 if unset or unreadable."""
    scheduling = config.get("scheduling", {}) if isinstance(config.get("scheduling"), dict) else {}
    try:
        return parse_time_of_day(scheduling.get("start_time", DEFAULT_START_TIME))
    except ValueError:
        return 0


def _number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def best_unit(amount: Any, units: Tuple[Tuple[str, float], ...]) -> Tuple[float, str]:
    """The largest unit that shows ``amount`` (in the first unit) as a whole number.

    7200 seconds -> (2, 'h'); 90 seconds -> (90, 's'); 600 seconds -> (10, 'min').
    """
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return 0.0, units[0][0]
    for name, size in reversed(units):
        if amount >= size and abs(amount / size - round(amount / size)) < 1e-9:
            return float(round(amount / size)), name
    return amount, units[0][0]


def to_base(value: float, unit: str, units: Tuple[Tuple[str, float], ...]) -> float:
    """A number and its unit, in the first unit of ``units`` (seconds or minutes)."""
    return float(value) * dict(units)[unit]


def describe_seconds(seconds: Any) -> str:
    """A recording length the way it would be said: '20 s', '10 min', '2 h', '1 h 30 min'."""
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return "?"
    if seconds < 60 or seconds % 60:
        return f"{_number(seconds)} s"
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"{minutes} min"
    hours, rest = divmod(minutes, 60)
    return f"{hours} h {rest} min" if rest else f"{hours} h"


def describe_interval(interval_minutes: Any, start: int = 0) -> str:
    """When recordings fall: 'every 30 min', 'every 2 h', 'once a day at 10:00'.

    ``start`` is minutes after midnight that the slots are counted from. It is
    always stated for a daily recording, and otherwise only when it is not
    midnight.
    """
    try:
        interval = int(interval_minutes)
    except (TypeError, ValueError):
        return "at an unknown interval"
    if interval >= MINUTES_PER_DAY:
        return f"once a day at {format_time_of_day(start)}"
    hours, rest = divmod(interval, 60)
    every = f"every {hours} h" if hours and not rest else f"every {interval} min"
    return f"{every}, counted from {format_time_of_day(start)}" if start % MINUTES_PER_DAY else every
