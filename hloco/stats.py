"""Small statistics helpers."""

from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_ci(xs: list[float], z: float = 1.959963984540054) -> tuple[float, float, float]:
    """Mean and normal-approximation 95% CI of the mean."""
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    m = sum(xs) / n
    if n == 1:
        return (m, float("nan"), float("nan"))
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    h = z * math.sqrt(var / n)
    return (m, m - h, m + h)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)
