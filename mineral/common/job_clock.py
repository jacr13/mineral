import os
import time

import re
from datetime import timedelta


_DURATION_RE = re.compile(r"(\d+)([smhd])")


def parse_runtime(runtime: str) -> int:
    runtime = runtime.strip().lower().replace(" ", "")

    if not runtime:
        raise ValueError("Runtime value cannot be empty.")

    # Format: HH:MM:SS or D-HH:MM:SS
    if ":" in runtime:
        if "-" in runtime:
            days_part, hms = runtime.split("-", 1)
            days = int(days_part)
        else:
            days = 0
            hms = runtime

        parts = hms.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid clock format: {runtime}")

        hours, minutes, seconds = map(int, parts)
        td = timedelta(
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
        )
        return int(td.total_seconds())

    # Format: composite suffixes (e.g. 2h30m)
    matches = _DURATION_RE.findall(runtime)
    if not matches:
        raise ValueError(f"Invalid runtime format: {runtime}")
    parsed = "".join(f"{n}{u}" for n, u in matches)
    if parsed != runtime:
        raise ValueError(f"Invalid runtime format: {runtime}")

    kwargs = {"days": 0, "hours": 0, "minutes": 0, "seconds": 0}

    for amount, suffix in matches:
        amount = int(amount)
        if suffix == "s":
            kwargs["seconds"] += amount
        elif suffix == "m":
            kwargs["minutes"] += amount
        elif suffix == "h":
            kwargs["hours"] += amount
        elif suffix == "d":
            kwargs["days"] += amount

    td = timedelta(**kwargs)
    return int(td.total_seconds())


class JobClock:
    def __init__(self, max_runtime, factor=3.0):
        if factor <= 0.0:
            raise ValueError("factor must be > 0.")

        self.max_runtime_seconds = (
            parse_runtime(max_runtime) if max_runtime is not None else None
        )
        self.factor = factor

        self.start_time = None
        self._step_start = None
        self._step_durations = []

    def arm(self):
        if self.start_time is not None:
            return

        now = time.perf_counter()

        slurm_start = os.environ.get("SLURM_JOB_START_TIME")
        if slurm_start is not None:
            try:
                wall_start = float(slurm_start)
                wall_now = time.time()
                offset = wall_now - wall_start
                self.start_time = now - offset
                return
            except ValueError:
                pass

        self.start_time = now

    def elapsed(self):
        if self.start_time is None:
            return None
        return time.perf_counter() - self.start_time

    def remaining_time(self):
        if self.max_runtime_seconds is None:
            return None
        elapsed = self.elapsed()
        if elapsed is None:
            return None
        return max(0.0, self.max_runtime_seconds - elapsed)

    def mean_step_duration(self):
        if not self._step_durations:
            return None
        return sum(self._step_durations) / len(self._step_durations)

    def should_safe_stop(self):
        """Stop if we likely cannot finish another step safely."""
        remaining = self.remaining_time()
        mean_step = self.mean_step_duration()

        if remaining is None or mean_step is None:
            return False

        return remaining <= self.factor * mean_step

    def step(self, check_safe_stop=False):
        """Record step duration.

        If check_safe_stop=True, also return safe_stop.
        """
        now = time.perf_counter()

        if self._step_start is None:
            self._step_start = now
            if check_safe_stop:
                return None, False
            return None

        duration = now - self._step_start
        self._step_durations.append(duration)
        self._step_start = now

        if check_safe_stop:
            return duration, self.should_safe_stop()
        return duration
