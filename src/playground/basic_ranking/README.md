# Basic Ranking (Momentum over Volatility)

**What**  
Cross-sectional ranking using momentum (rolling mean of returns) over volatility (rolling standard deviation) on a fixed lookback window.

**How**  
- Data source: Yahoo Finance (adjusted close)  
- Score formula: `momentum / volatility`  
- Cross-sectional ranking computed per date (1 = best asset)

**Run**
```bash
python src/playground/basic_ranking/basic_ranking.py
