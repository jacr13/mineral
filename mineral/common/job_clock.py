import os
import time


def parse_runtime(value):
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    value = str(value).strip()
    if not value:
        return None
    if value.isdigit():
        return int(value) * 60

    days = 0
    if "-" in value:
        days_part, value = value.split("-", 1)
        if days_part:
            days = int(days_part)

    parts = value.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = (int(p) for p in parts)
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = (int(p) for p in parts)
    elif len(parts) == 1:
        hours = 0
        minutes = 0
        seconds = int(parts[0])
    else:
        return None

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


class JobClock:
    def __init__(self, max_runtime=None, factor=3.0):
        if factor <= 0.0:
            raise ValueError("factor must be > 0.")

        self.max_runtime_seconds = parse_runtime(max_runtime)
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
