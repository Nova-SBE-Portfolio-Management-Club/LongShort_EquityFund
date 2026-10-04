# Shared backtesting conventions

The implementation lives in [`src/utils/backtest.py`](../src/utils/backtest.py). The earnings-surprise `backtest.py` re-exports these functions for existing notebooks; it contains no separate calculations.

## Capital and exposure

Inputs to `backtest` are simple returns of the underlying long and short books. A positive return in the short input is a loss for the portfolio. At gross leverage `L`, target long exposure is `L / 2` times equity and target short exposure is `-L / 2` times equity.

For each observed period:

```text
portfolio_return = leverage × (long_return − short_return) / 2
equity_t = equity_(t−1) × (1 + portfolio_return_t)
```

At leverage 1, 10% long-book appreciation and 10% short-book depreciation produce a 10% portfolio return. At leverage 2, that same pair produces a 20% portfolio return.

The books are rebalanced to these target exposures each period. `long_value` and `short_value` are positive target notionals after that period's return; they are not separate equity accounts and their sum is gross leverage times equity. A zero-leverage portfolio stays flat. Returns below a 100% equity loss are rejected; a complete loss leaves equity at zero.

The shared simulator does not model trading costs, interest on cash, short borrow, financing, margin requirements, or liquidation. Strategy-specific returns that already include portfolio weights should go directly to `summary_stats`, rather than being weighted again through `backtest`.

## Dates and missing data

- Both inputs must be numeric pandas Series with unique, non-missing period labels.
- Inputs are sorted into period order. Returns use the intersection of their labels, then drop periods where either book has a missing return.
- Missing observations are never converted into flat returns. Input series are not mutated.
- An empty sample, no valid overlap, infinite returns, or an underlying simple return below -100% produces an explicit error.
- `result.attrs` records the initial capital, gross leverage, and number of discarded periods from each input.

There is no calendar inference. When dates are dropped, annualized metrics apply to the remaining observed sample. Callers must use comparable periods and supply the correct `periods_per_year` (default 252 for trading-day returns).

## Performance metrics

- **Annualized return:** geometric CAGR, based on compounded simple returns and the count of observed periods.
- **Annualized volatility:** sample standard deviation (`ddof=1`) multiplied by the square root of periods per year.
- **Sharpe ratio:** mean per-period excess return divided by sample standard deviation, multiplied by the square root of periods per year. The constant annual cash rate is converted to a compounded per-period rate. Square-root scaling assumes comparable periods without serial dependence. This follows the differential-return definition in [William Sharpe's original explanation](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm).
- **Drawdown:** compounded wealth divided by its running peak, with initial wealth of one included in the peak. An initial loss is counted, and a complete loss has a drawdown of -100%.
- **Undefined statistics:** one observation has no sample volatility; a constant-return sample has no finite Sharpe ratio. Those metrics are NaN, rather than being reported as zero. Empty samples are rejected.
- `summary_stats` uses decimal percentage units. `compare_strategies` converts percentage columns to percent units. The legacy `Total Days` key counts observed periods even for non-daily inputs.

## Changes from the previous implementation

The old portfolio return used the full long-minus-short spread while the displayed allocations used half the capital per book. The corrected implementation uses half the spread at leverage 1. Previously saved demo and momentum outputs must therefore be regenerated.

The Sharpe calculation now uses arithmetic excess returns rather than subtracting a cash rate from CAGR. Drawdowns include the starting capital, and incomplete input observations are discarded rather than filled with zero.

The bundled earnings strategy constructs weighted daily portfolio returns itself. Its portfolio weights are unaffected by the shared exposure correction; the published report uses the corrected shared statistics to assess those returns.
