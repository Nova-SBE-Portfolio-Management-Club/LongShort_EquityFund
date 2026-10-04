# Basic Ranking (Momentum over Volatility)

**What**  
Cross-sectional ranking using momentum (rolling mean of returns) over volatility (rolling standard deviation) on a fixed lookback window.

**How**  
- Data source: Yahoo Finance (adjusted close)  
- Score formula: `momentum / volatility`  
- Cross-sectional ranking computed per date (1 = best asset)

**Run**

Install the environment described in the [root README](../../../README.md). From the repository root:

```bash
python -m jupyter lab src/playground/basic_ranking/basic_ranking.ipynb
```

Run the cells in order. The notebook imports `Ranking` from `src/utils/ranking.py`, downloads Yahoo Finance prices, and writes `ranking_<date>.csv` in the notebook's working directory.

Edit the tickers, sample dates, and lookback settings in the notebook before running a new experiment. Internet access is required for the price download.
