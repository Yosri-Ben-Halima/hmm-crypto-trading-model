
# Hidden Markov Model (HMM) Bitcoin Trading Strategy

A quantitative trading framework that applies **Hidden Markov Models (HMMs)** to Bitcoin.  
The goal is to identify hidden market regimes (e.g., bullish or bearish) and adapt trading decisions accordingly — instead of passively holding Bitcoin.  

---

## Overview

This project combines **statistical modeling**, **machine learning**, and **backtesting** to evaluate whether regime-switching strategies can outperform a simple Buy & Hold (HODL) benchmark.  

The pipeline includes:

- Data loading and feature engineering  
- State calibration and HMM training  
- Signal generation based on inferred regimes  
- Backtesting with transaction costs  
- Benchmarking against Buy & Hold  
- Monte Carlo simulations for robustness  

---

## Average Results (Monte-Carlo on Test Period)

| Metric              | HMM Strategy  | Buy & Hold |
|----------------------|--------------|------------|
| Annualized Sharpe    | **1.71**     | 1.02       |
| P&L (%)              | **+51%**     | +23%       |
| Max Drawdown (%)     | **-21%**     | -28%       |

The HMM strategy delivered **higher risk-adjusted returns**, while **reducing drawdowns** compared to simply holding Bitcoin.  

---

## Quick Start

Clone the repo and install dependencies:

```bash
git clone https://github.com/Yosri-Ben-Halima/hmm-crypto-trading-model.git
cd hmm-crypto-trading-model
pip install -r requirements.txt
````

Run the Jupyter notebook for a full example:

```bash
jupyter notebook main.ipynb
```

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.
It is **not investment advice** and should not be used for live trading.

---

## Connect with Me

Thank you for visiting my GitHub profile! Feel free to reach out if you have any questions or opportunities to collaborate. Let's connect and explore new possibilities together:

[![GitHub](https://img.shields.io/badge/GitHub-Yosri--Ben--Halima-black?logo=github)](https://github.com/Yosri-Ben-Halima)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Yosri%20Ben%20Halima-blue?logo=linkedin)](https://www.linkedin.com/in/yosri-benhalima/)
[![Facebook](https://img.shields.io/badge/Facebook-@Yosry%20Ben%20Hlima-navy?logo=facebook)](https://www.facebook.com/NottherealYxsry)
[![Instagram](https://img.shields.io/badge/Instagram-@yosrybh-orange?logo=instagram)](https://www.instagram.com/yosrybh/)
[![Email](https://img.shields.io/badge/Email-yosri.benhalima@ept.ucar.tn-white?logo=gmail)](mailto:yosri.benhalima@ept.ucar.tn)
[![Personal Web Page](https://img.shields.io/badge/Personal%20Web%20Page-Visit%20Now-green?logo=googlechrome)](https://personal-web-page-yosribenhlima.streamlit.app/)
[![Google Drive](https://img.shields.io/badge/My%20Resume-Click%20Here-red?logo=googledrive&logoColor=white)](https://drive.google.com/file/d/18xB1tlZUBWz5URSli_9kewEFZwZPz235/view?usp=sharing)
[![PyPI](https://img.shields.io/badge/PyPI-yosri--ben--halima-pink?logo=pypi)](https://pypi.org/user/yosri-ben-halima/)
