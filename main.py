"""GTA VI trophy earn-rate forecast for make.fun competition
gta6-trophy-rarity-forecast-01 (wallet w8).

Forecasts, for each of the game's PS5 trophies, five quantiles of its Exophase
earn rate (percent) at the case's target time.

Approach
--------
Two regimes, chosen per trophy from the spread of its own history:

* Flat history (every read identical -- Exophase has not refreshed its counts
  yet).  A refresh can only dilute the rates: the same earned counts divided
  by a larger tracked-player base.  The median is the last rate scaled by a
  dilution factor of 0.30, with a wide 1.35-logit spread, because the size of
  the refresh is unknown.  The one exception is a trophy at 100%: every
  tracked player has earned it, and a refresh leaves it at 100%, so its
  median stays at 100% with a narrow spread.

* Trending history (rates have moved).  An ensemble of three sub-models,
  combined as a weighted mean in logit space:
    (a) flat -- the last observed rate (weight 0.25);
    (b) logit-linear trend -- least-squares line of logit(rate) against
        ln(1 + hours since 2026-11-19T00:00:00Z), fitted on the last 24h of
        reads, read at the target (weight 0.45);
    (c) saturation -- rate(t) = A * (1 - exp(-k*t)) with k chosen from
        {0.01, 0.04, 0.16} per hour by least squares on the rate scale, A
        fitted by least squares (weight 0.30).
  The spread is the disagreement between the three sub-models (population
  standard deviation of their logits, floored at 0.2) plus a random-walk term
  0.5 * sqrt(horizon in days) plus 0.25.

Quantiles are the median logit shifted by the standard-normal z-scores
(-1.645, -0.674, 0, 0.674, 1.645) times the spread, mapped back through the
logistic function, clamped to [0.01, 99.99], rounded to 4 decimals (half up),
and forced non-decreasing.

Standard library only, no network, no randomness: two runs produce identical
bytes.  Reads every *.in in $MAKEFUN_INPUT_DIR (else /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Time origin for the trend and saturation fits: GTA VI launch day, UTC.
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Flat-history branch: dilution factor and logit spread.
DILUTION_FACTOR = 0.30
FLAT_SPREAD = 1.35
# Spread kept for a trophy every tracked player has earned (stays at 100%).
FULL_SPREAD = 0.5

# Trending-history ensemble weights: flat / trend / saturation.
W_FLAT, W_TREND, W_SAT = 0.25, 0.45, 0.30
# Saturation growth rates tried, per hour.
K_GRID = (0.01, 0.04, 0.16)
# Trend fit window, in hours before the last read.
TREND_WINDOW_HOURS = 24.0
# Random-walk spread, in logit units per square-root day.
RANDOM_WALK_SIGMA = 0.5
# Floor on the ensemble-disagreement part of the spread, and extra padding.
SPREAD_FLOOR = 0.2
SPREAD_EXTRA = 0.25
# A rate range (percentage points) across reads that still counts as flat.
FLAT_EPS = 0.01


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(logit_value: float) -> float:
    """Logistic function, in percent, computed without overflow."""
    if logit_value >= 0:
        return 100.0 / (1.0 + math.exp(-logit_value))
    e = math.exp(logit_value)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    value = float(Decimal(repr(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
    return int(value) if value == int(value) else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_epoch(moment: datetime) -> float:
    return (moment - EPOCH).total_seconds() / 3600.0


def least_squares(xs, ys):
    """Ordinary least squares y = a + b*x; None when x has no variance."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return mean_y - slope * mean_x, slope


def std_pop(values) -> float:
    n = len(values)
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def quantiles_from(median_logit: float, spread: float) -> list:
    qs = [round4(clamp_pct(sigmoid(median_logit + z * spread))) for z in Z]
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_trophy(reads, target_hours: float, horizon_days: float) -> list:
    """Forecast one trophy.

    reads: [(hours since epoch, rate or None)], oldest first.
    """
    points = [(h, r) for h, r in reads if r is not None]
    if not points:
        # No usable read at all: a wide, centred forecast.
        return quantiles_from(0.0, 2.0)
    rates = [r for _, r in points]
    last_rate = rates[-1]
    flat = (max(rates) - min(rates)) <= FLAT_EPS

    if flat:
        if last_rate >= 99.99:
            # Everyone tracked has earned it; a refresh leaves it at 100%.
            return quantiles_from(logit(100.0), FULL_SPREAD)
        # Anticipate the refresh: same earned counts, more players tracked.
        return quantiles_from(logit(last_rate * DILUTION_FACTOR), FLAT_SPREAD)

    # --- trending history: three-model ensemble in logit space ---

    # (a) Flat: the last observed rate.
    logit_flat = logit(last_rate)

    # (b) Logit-linear trend against ln(1 + hours), last 24h of reads.
    last_h = points[-1][0]
    window = [(h, r) for h, r in points if h >= last_h - TREND_WINDOW_HOURS]
    logit_trend = logit_flat
    if len(window) >= 2:
        xs = [math.log1p(max(h, -0.999999)) for h, _ in window]
        ys = [logit(r) for _, r in window]
        fit = least_squares(xs, ys)
        if fit is not None:
            intercept, slope = fit
            logit_trend = intercept + slope * math.log1p(max(target_hours, -0.999999))

    # (c) Saturation curve A * (1 - exp(-k*t)) on the rate scale.
    logit_sat = logit_flat
    positive = [(h, r) for h, r in points if h > 0]
    if len(positive) >= 2:
        best = None
        for k in K_GRID:
            factors = [1.0 - math.exp(-k * h) for h, _ in positive]
            denom = sum(f * f for f in factors)
            if denom <= 0:
                continue
            a_fit = sum(r * f for (_, r), f in zip(positive, factors)) / denom
            sse = sum((r - a_fit * f) ** 2 for (_, r), f in zip(positive, factors))
            if best is None or sse < best[0]:
                best = (sse, a_fit, k)
        if best is not None:
            _, a_fit, k = best
            a_fit = min(100.0, max(0.0, a_fit))
            predicted = a_fit * (1.0 - math.exp(-k * max(target_hours, 0.0)))
            logit_sat = logit(predicted)

    median_logit = W_FLAT * logit_flat + W_TREND * logit_trend + W_SAT * logit_sat
    disagreement = std_pop([logit_flat, logit_trend, logit_sat])
    spread = (
        max(disagreement, SPREAD_FLOOR)
        + RANDOM_WALK_SIGMA * math.sqrt(horizon_days)
        + SPREAD_EXTRA
    )
    return quantiles_from(median_logit, spread)


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    target_hours = hours_since_epoch(parse_time(case["target"]))
    horizon_days = float(case["horizon_minutes"]) / 1440.0
    read_hours = [hours_since_epoch(parse_time(read["at"])) for read in history]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        reads = []
        for read, h in zip(history, read_hours):
            rates = read.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            reads.append((h, rate))
        rows.append({"id": trophy["id"], "q": forecast_trophy(reads, target_hours, horizon_days)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
