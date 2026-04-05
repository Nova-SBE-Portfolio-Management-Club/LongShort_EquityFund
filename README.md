# 🎯 Long/Short Equity Fund - Sector Rotation Strategy

**AI-powered sector rotation model using ensemble machine learning for systematic long/short equity strategies**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Documentation](#documentation)
- [Contributing](#contributing)

---

## 🎯 Overview

This repository contains a **production-ready sector rotation strategy** that uses machine learning to predict S&P 500 sector performance and construct optimal long/short portfolios.

### Key Features

✅ **Ensemble ML Model** - Combines Random Forest, Gradient Boosting, and Ridge Regression  
✅ **Walk-Forward Validation** - No lookahead bias, truly out-of-sample testing  
✅ **Stress Tested** - Validated from 2007-2025, including 6 major market crises  
✅ **Transaction Cost Modeling** - Realistic 30bps round-trip costs + short borrow  
✅ **Regime Detection** - Adapts to Risk-On vs Risk-Off market conditions  
✅ **Comprehensive Backtesting** - Full performance metrics and visualizations  

### Strategy Summary

- **Universe**: 11 S&P 500 sector ETFs (XLK, XLF, XLE, etc.)
- **Rebalancing**: Quarterly
- **Position Structure**: Long top 3 sectors, Short bottom 3 sectors
- **Features**: 15+ macroeconomic indicators from FRED
- **Holding Period**: 1 quarter (3 months)

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher required
python --version

# Install dependencies
pip install -r requirements.txt
```

### Run the Model

```bash
# 1. Generate current sector predictions
python models/sector_rotation/random_forest_v3.py

# 2. Run stress test (2007-2025 backtest)
python models/sector_rotation/stress_test.py

# 3. Open Jupyter dashboard for analysis
jupyter notebook notebooks/sector_dashboard.ipynb
```

---

## 📁 Project Structure

```
LongShort_EquityFund/
│
├── models/                          # Production models
│   ├── sector_rotation/
│   │   ├── random_forest_v3.py     # Main prediction model ⭐
│   │   ├── stress_test.py          # Comprehensive backtest
│   │   └── config.py               # Model configuration
│   └── data/
│       └── sector_model_full_data.csv  # Cached macro + sector data
│
├── notebooks/                       # Analysis & visualization
│   ├── sector_dashboard.ipynb      # Main dashboard ⭐
│   ├── stress_test_analysis.ipynb  # Stress test deep dive
│   └── exploratory/                # Experimental notebooks
│
├── results/                         # Model outputs
│   ├── predictions/                # Current sector recommendations
│   ├── stress_tests/               # Historical backtest results
│   └── backtests/                  # Walk-forward validation
│
├── docs/                           # Documentation
│   ├── QUICKSTART.md              # Getting started guide
│   ├── IMPROVEMENTS.md            # Model enhancement docs
│   └── STRESS_TEST_README.md      # Stress test guide
│
├── archive/                        # Historical/experimental work
│   ├── 2025_work/                 # Previous year experiments
│   └── 2026_work/                 # Current year experiments
│
├── src/                            # Original infrastructure
│   ├── database/                  # Data persistence layer
│   ├── bloomberg/                 # Bloomberg API integration
│   └── utils/                     # Utility functions
│
├── README.md                       # This file
├── requirements.txt                # Python dependencies
└── .gitignore                      # Git ignore rules
```

---

## ✨ Features

### 1. **Ensemble Machine Learning Model**
- **Random Forest** (40% weight): Captures non-linear relationships
- **Gradient Boosting** (35% weight): Sequential error correction
- **Ridge Regression** (25% weight): Linear baseline + regularization

### 2. **Robust Data Pipeline**
- Automatic data fetching from FRED & Yahoo Finance
- Exponential backoff retry logic for API rate limits
- Local caching to avoid repeated downloads
- Data validation and quality checks

### 3. **Advanced Features**
- **Macroeconomic Indicators**: GDP growth, unemployment, inflation, yield curve, credit spreads, VIX, etc.
- **Momentum Features**: 1Q, 3Q, 6Q trailing returns
- **Relative Strength**: Sector performance vs S&P 500
- **Volatility Features**: Rolling sector volatility

### 4. **Risk Management**
- **Transaction Costs**: 10bps execution + 5bps slippage
- **Short Borrow Costs**: 1.5% annual borrow rate
- **Position Limits**: Max 20% per sector
- **Correlation Control**: Avoid highly correlated positions (>0.7)

### 5. **Regime Detection**
- **Risk-Off**: High VIX (>25), inverted yield curve, wide credit spreads
- **Risk-On**: Normal volatility, positive yield curve
- **Dynamic Adjustments**: +2% boost to defensive in Risk-Off, +1% to cyclicals in Risk-On

### 6. **Comprehensive Testing**
- **Walk-Forward Validation**: Out-of-sample R² scores per sector
- **Stress Testing**: Performance in 6 major crises (2008, 2020, 2022, etc.)
- **Regime Analysis**: Risk-On vs Risk-Off performance breakdown

---

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/LongShort_EquityFund.git
cd LongShort_EquityFund
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `models/sector_rotation/config.py` file:

```python
# FRED API Key (get free key from https://fred.stlouisfed.org/docs/api/api_key.html)
FRED_API_KEY = "your_fred_api_key_here"
```

---

## 🎮 Usage

### Generate Current Predictions

```bash
cd models/sector_rotation
python random_forest_v3.py
```

**Output:**
- `sector_predictions_v3.csv` - Current sector rankings
- `validation_metrics_v3.csv` - Out-of-sample performance
- Console output with portfolio recommendations

### Run Comprehensive Stress Test

```bash
python stress_test.py
```

**Output:**
- `results/stress_tests/stress_test_quarterly_results.csv` - Quarterly performance
- `results/stress_tests/stress_test_crisis_analysis.csv` - Crisis performance
- `results/stress_tests/stress_test_regime_analysis.csv` - Regime breakdown
- Performance visualizations (PNG files)

### Analyze Results in Jupyter

```bash
jupyter notebook notebooks/sector_dashboard.ipynb
```

**Features:**
- Current sector recommendations visualization
- Historical performance charts
- Crisis period analysis
- Prediction accuracy by sector
- Automated diagnostics and recommendations

---

## 📊 Results

### Stress Test Performance (2011-2025)

| Metric | Value |
|--------|-------|
| **Total Return** | -52.26% ⚠️ |
| **Annualized Return** | -4.97% |
| **Sharpe Ratio** | -0.33 |
| **Max Drawdown** | -64.58% |
| **Win Rate** | 43.1% |

⚠️ **Status**: Model needs improvements - predictions appear inverted. See `docs/IMPROVEMENTS_DOCUMENTATION.md` for action items.

### Crisis Performance

| Crisis | Return | Status |
|--------|--------|--------|
| 2011 Debt Crisis | +1.12% | ✅ |
| 2015 Oil Collapse | +4.92% | ✅ |
| 2018 Q4 Selloff | -0.05% | ⚠️ |
| 2020 COVID Crash | -9.11% | ❌ |
| 2022 Bear Market | -6.11% | ❌ |

---

## 📚 Documentation

- **[Quick Start Guide](docs/QUICKSTART_GUIDE.md)** - Get up and running fast
- **[Stress Test Documentation](docs/STRESS_TEST_README.md)** - Understanding the backtest
- **[Improvement Documentation](docs/IMPROVEMENTS_DOCUMENTATION.md)** - Model enhancement roadmap
- **[API Documentation](docs/API.md)** - Code reference (coming soon)

---

## 🔧 Development

### Project Status

🚧 **Current Status**: Model in development - stress test identified prediction inversion issues

### Key Findings from Stress Test

1. ❌ **Predictions Inverted**: Negative correlation between scores and actual returns
2. ❌ **LONG positions losing money**: Picking wrong sectors to buy
3. ❌ **SHORT positions gaining**: Picking wrong sectors to short
4. ❌ **Risk-Off catastrophic losses**: -49% in defensive periods

### Next Steps

1. Debug prediction logic in `random_forest_v3.py`
2. Verify ensemble model training
3. Check feature scaling and regime adjustments
4. Test with inverted scores (multiply by -1)
5. Re-run stress test to validate fixes

See `docs/IMPROVEMENTS_DOCUMENTATION.md` for detailed action plan.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **Data Sources**: FRED (Federal Reserve Economic Data), Yahoo Finance
- **Machine Learning**: scikit-learn, XGBoost
- **Analysis**: pandas, numpy, matplotlib, seaborn

---

## 📧 Contact

For questions or collaboration: [your-email@example.com]

**Repository**: [https://github.com/yourusername/LongShort_EquityFund](https://github.com/yourusername/LongShort_EquityFund)

---

*Last Updated: February 2026*
