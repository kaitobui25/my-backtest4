# Screening gates

Default gates:

- TRAIN: expectancy at least +0.225R and at least 300 closed trades.
- VALIDATION: expectancy at least +0.225R and at least 50 closed trades.
- FINAL OOS: expectancy at least +0.225R and at least 50 closed trades.

The runner stops before FINAL OOS if an earlier stage fails. This prevents consuming the untouched OOS period for strategy families that have already failed screening.
