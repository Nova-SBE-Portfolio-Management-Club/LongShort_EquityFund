from pathlib import Path

try:
    from .backtest import run_backtest_pipeline
    from .paths import reports_dir
except ImportError:  # pragma: no cover - supports direct script execution
    from backtest import run_backtest_pipeline
    from paths import reports_dir

def main():
    _ = Path(__file__).resolve().parent
    run_backtest_pipeline(output_dir=reports_dir())

if __name__ == "__main__":
    main()
