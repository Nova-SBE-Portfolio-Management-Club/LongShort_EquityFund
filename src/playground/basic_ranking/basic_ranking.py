# src/playground/basic_ranking/basic_ranking.py
# Loads prices via yfinance, ranks assets using Ranking, prints latest table, saves CSV.

from __future__ import annotations
import pandas as pd
import yfinance as yf
from models import Ranking 
import warnings
warnings.filterwarnings("ignore")

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
START = "2022-01-01"
END = "2024-12-31" 


def load_prices(tickers, start, end) -> pd.DataFrame:
    """
    Download adjusted prices; return DataFrame (dates x tickers).
    """
    print(f"[INFO] Downloading {tickers} from {start} to {end}")
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   
        progress=True,
        threads=False    
    )
    if data.empty:
        raise ValueError("No data returned. Check tickers or date range.")
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].dropna(how="all")
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})
    return prices.dropna()


def main():
    prices = load_prices(TICKERS, START, END)
    model = Ranking(lookback=60, min_obs=40, winsor_pct=0.01)
    print("[INFO] Computing scores and ranks...")
    scores, ranks = model.rank(prices, ascending=False)

    last_dt = ranks.index[-1]
    latest = pd.DataFrame({
        "score": scores.loc[last_dt].sort_values(ascending=False),
        "rank": ranks.loc[last_dt].sort_values()
    })
    print(f"\nRanking at {last_dt.date()}:\n")
    print(latest.to_string(float_format=lambda x: f"{x:,.4f}"))

    out = f"ranking_{last_dt.date()}.csv"
    latest.to_csv(out)
    print(f"\n[INFO] Saved latest ranking to {out}")


if __name__ == "__main__":
    main()
