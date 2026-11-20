[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/binasalama12/makefun-gta6-trophy-rarity-forecast-01-w8/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/binasalama12/makefun-gta6-trophy-rarity-forecast-01-w8/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast — wallet w8

Forecasts, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, five
quantiles (5%, 25%, 50%, 75%, 95%) of its Exophase earn rate (percent) at the
case's target time. Built for the make.fun competition
`gta6-trophy-rarity-forecast-01`.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads every `*.in`
in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (else `/output`).

Per trophy it picks one of two regimes from the spread of that trophy's own
history:

**Flat history — the dilution branch.** When every read so far is identical,
Exophase has not refreshed its counts yet, and a refresh can only dilute the
rates: the same earned counts divided by a larger tracked-player base. The
median is the last rate scaled by 0.30, with a wide 1.35-logit spread, since
the size of the refresh is unknown. The one exception is a trophy at 100%:
every tracked player has earned it, and a refresh leaves it at 100%, so its
median stays at 100% with a narrow spread.

**Trending history — a three-model ensemble in logit space.**

- (a) Flat: the last observed rate (weight 0.25).
- (b) Logit-linear trend: least-squares line of logit(rate) against
  ln(1 + hours since 2026-11-19T00:00:00Z), fitted on the last 24 hours of
  reads and read at the target (weight 0.45).
- (c) Saturation: rate(t) = A·(1 − exp(−k·t)) with k chosen from
  {0.01, 0.04, 0.16} per hour by least squares on the rate scale, A fitted by
  least squares (weight 0.30).

The final median logit is the weighted mean of the three sub-model logits.
The spread is the disagreement between the sub-models (population standard
deviation of their logits, floored at 0.2) plus a random-walk term
0.5·√(horizon in days) plus 0.25.

Quantiles are the median logit shifted by the standard-normal z-scores
(−1.645, −0.674, 0, 0.674, 1.645) times the spread, mapped back through the
logistic function, clamped to [0.01, 99.99], rounded to 4 decimals (half up),
and forced non-decreasing.

The program is deterministic: no randomness, no wall-clock reads, files
processed in name order, so two runs produce identical bytes.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Files

- `main.py` — the whole entry.
- `README.md` — this file.
- `LICENSE` — MIT.
