"""Prometheus metrics without a client library.

A handful of counters and histograms rendered in the Prometheus text
exposition format (version 0.0.4). Small enough to read in one sitting,
and keeps the dependency list short.
"""

from __future__ import annotations

import math
import threading
from bisect import bisect_left
from collections import defaultdict

# Milliseconds: covers an in-memory cache hit (~10 ms) up to a slow LLM call.
LATENCY_BUCKETS_MS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000)


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(names: tuple[str, ...], values: tuple[str, ...], extra: str = "") -> str:
    parts = [f'{n}="{_escape(v)}"' for n, v in zip(names, values)]
    if extra:
        parts.append(extra)
    return "{" + ",".join(parts) + "}" if parts else ""


class Counter:
    def __init__(self, name: str, help_text: str, labels: tuple[str, ...] = ()):
        self.name, self.help, self.label_names = name, help_text, labels
        self._values: dict[tuple[str, ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(str(labels.get(n, "")) for n in self.label_names)
        with self._lock:
            self._values[key] += amount

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        with self._lock:
            for key, value in sorted(self._values.items()):
                lines.append(f"{self.name}{_labels(self.label_names, key)} {value:g}")
        return lines


class Histogram:
    def __init__(self, name: str, help_text: str, labels: tuple[str, ...] = (), buckets=LATENCY_BUCKETS_MS):
        self.name, self.help, self.label_names = name, help_text, labels
        self.buckets = tuple(sorted(buckets))
        self._series: dict[tuple[str, ...], list] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: str) -> None:
        if value is None or math.isnan(value):
            return
        key = tuple(str(labels.get(n, "")) for n in self.label_names)
        with self._lock:
            counts, total = self._series.setdefault(key, [[0] * (len(self.buckets) + 1), [0.0, 0]])[0:2]
            counts[bisect_left(self.buckets, value)] += 1
            total[0] += value
            total[1] += 1

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        with self._lock:
            for key, (counts, (total, count)) in sorted(self._series.items()):
                cumulative = 0
                for bound, n in zip(self.buckets, counts):
                    cumulative += n
                    le = 'le="%g"' % bound
                    lines.append(f"{self.name}_bucket{_labels(self.label_names, key, le)} {cumulative}")
                inf = 'le="+Inf"'
                lines.append(f"{self.name}_bucket{_labels(self.label_names, key, inf)} {count}")
                lines.append(f"{self.name}_sum{_labels(self.label_names, key)} {total:g}")
                lines.append(f"{self.name}_count{_labels(self.label_names, key)} {count}")
        return lines


class Metrics:
    """All metrics CacheGuard exports. One instance per app."""

    def __init__(self):
        self.http_requests = Counter("cacheguard_http_requests_total", "HTTP requests by route and status.", ("route", "method", "status"))
        self.http_latency = Histogram("cacheguard_http_request_ms", "HTTP request latency in milliseconds.", ("route",))
        self.lookups = Counter("cacheguard_lookups_total", "Cache lookups by outcome and deciding stage.", ("outcome", "stage", "reason"))
        self.query_latency = Histogram("cacheguard_query_ms", "End-to-end query latency by outcome, in milliseconds.", ("outcome",))
        self.decision_latency = Histogram("cacheguard_decision_ms", "Time to decide hit or miss (includes the verifier), in milliseconds.", ("stage",))
        self.backend_latency = Histogram("cacheguard_backend_ms", "Backend (LLM) call latency in milliseconds.", ("backend",))
        self.backend_errors = Counter("cacheguard_backend_errors_total", "Failed backend calls.", ("backend", "retryable"))
        self.rejections = Counter("cacheguard_limit_rejections_total", "Requests refused by protection limits.", ("limit",))

    def render(self, gauges: dict[str, tuple[str, float]] | None = None) -> str:
        lines: list[str] = []
        for metric in (self.http_requests, self.http_latency, self.lookups, self.query_latency,
                       self.decision_latency, self.backend_latency, self.backend_errors, self.rejections):
            lines += metric.render()
        for name, (help_text, value) in (gauges or {}).items():
            lines += [f"# HELP {name} {help_text}", f"# TYPE {name} gauge", f"{name} {value:g}"]
        return "\n".join(lines) + "\n"
