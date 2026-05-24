# Discrete Regimes vs Conditional Volatility for Market Risk Forecasting

Comparing HMM and GARCH-style models for SPY density forecasts, VaR calibration, Expected Shortfall severity, and stress-period behaviour.

## Executive Summary

This project evaluates whether equity market risk is better forecast using discrete volatility regimes or continuous conditional volatility dynamics.

Using daily SPY log returns from 2000–2025, models are trained on 2000–2019 and tested on 2020–2025 under a fixed-origin forecasting design. The main comparison is between:

- **Hidden Markov Models (HMMs)**: volatility evolves through persistent latent regimes;
- **GARCH / t-GARCH models**: volatility evolves continuously through recent shocks and past variance.

Static Gaussian and Student-t models are included as diagnostic baselines.

### Main Findings

1. **The 4-state HMM gives the strongest average density forecasts.**  
   It achieves the best out-of-sample predictive log-score and significantly outperforms Gaussian, Student-t, GARCH, and t-GARCH benchmarks under Diebold-Mariano tests.

2. **t-GARCH gives the strongest VaR calibration.**  
   It passes Kupiec, independence, and conditional coverage tests at both 95% and 99% VaR levels.

3. **Density forecasting and tail-risk calibration are not the same objective.**  
   The HMM forecasts the overall predictive distribution well, but t-GARCH is better calibrated for extreme one-day VaR.

4. **Static models reveal useful failure modes.**  
   Gaussian models underestimate 99% tail risk. Student-t models improve unconditional tail thickness, but breach clustering remains.

5. **The HMM’s VaR undercoverage appears more like tail-shape misspecification than timing failure.**  
   HMM breaches are not strongly clustered, but breach counts are too high. This suggests the Gaussian emissions may be too thin-tailed even though regime timing is informative.

---

## Aims

> Can discrete volatility regimes provide useful market-risk forecasts compared with continuous conditional volatility models?

In particular:

- Do SPY returns exhibit persistent volatility regimes?
- Does a Gaussian HMM improve full-density forecasting relative to standard benchmarks?
- Do HMM and GARCH-style models produce reliable one-day VaR forecasts?
- Does heavy-tailed conditional volatility improve tail-risk calibration?
- How do models behave during stress periods such as COVID-19?

---

## Data and Forecasting Design

| Item | Description |
|---|---|
| Asset | SPY ETF |
| Frequency | Daily |
| Return definition | Percentage log returns, $100 \log(P_t / P_{t-1})$ |
| Full sample | 2000–2025 |
| Training period | 2000–2019 |
| Test period | 2020–2025 |
| Forecast horizon | One trading day |
| Design | Fixed-origin forecasting |

All model parameters are estimated using only the training period. During the test period, parameters remain fixed, but recursive state variables are updated:

- HMM filtered regime probabilities update through the forward filter;
- GARCH conditional volatility updates through the variance recursion.

This avoids parameter leakage and creates a conservative out-of-sample validation design.

---

## Models Compared

### Main Models

| Model | Purpose |
|---|---|
| **4-state Gaussian HMM** | Captures persistent discrete volatility regimes |
| **GARCH(1,1)** | Captures continuous conditional volatility dynamics |
| **t-GARCH(1,1)** | Combines conditional volatility with heavy-tailed shocks |

### Baseline Models

| Model | Purpose |
|---|---|
| **i.i.d. Gaussian** | Constant volatility, thin-tailed benchmark |
| **i.i.d. Student-t** | Constant volatility, heavy-tailed benchmark |

The baselines are included to distinguish between failures due to constant volatility, thin tails, and lack of dynamic risk updating.

---

## HMM Model Selection

Gaussian HMMs with 2–5 states are compared using BIC and regime persistence.

| HMM States | BIC | Minimum Expected Duration |
|---:|---:|---:|
| 2 | 14044.44 | 36.41 |
| 3 | 13639.10 | 27.03 |
| 4 | 13598.68 | 16.18 |
| 5 | 13614.24 | 1.07 |

The **4-state HMM** is selected because it has the lowest BIC among non-degenerate HMM specifications. The 5-state model has short-lived regimes, suggesting overfitting or weak state identification.

The selected HMM identifies persistent volatility states with approximately negligible daily mean returns, indicating that regime classification is primarily driven by variance rather than drift.

---

## Density Forecasting Results

Predictive log-score evaluates the full one-step-ahead return distribution. Higher log-score is better.

| Model | AIC | BIC | Log-Score |
|---|---:|---:|---:|
| 4-state HMM | 13448.653 | 13598.681 | **-1.412** |
| Gaussian | 16063.557 | 16076.603 | -1.698 |
| Student-t | 14663.968 | 14683.537 | -1.523 |
| GARCH(1,1) | 13671.563 | 13691.132 | -1.446 |
| t-GARCH(1,1) | 13459.764 | **13485.856** | -1.424 |

The HMM produces the strongest average predictive log-score, while t-GARCH is the closest benchmark and is more parsimonious under BIC.

### Diebold-Mariano Tests

Diebold-Mariano tests are performed using negative log predictive density losses. Negative mean differences indicate that the HMM has lower average loss.

| Comparison | DM Statistic | p-value | Mean Difference |
|---|---:|---:|---:|
| HMM vs Gaussian | -5.716 | 0.000 | -0.286 |
| HMM vs Student-t | -8.376 | 0.000 | -0.111 |
| HMM vs GARCH(1,1) | -3.163 | 0.0016 | -0.034 |
| HMM vs t-GARCH(1,1) | -2.280 | 0.0227 | -0.012 |

The HMM’s log-score improvement is statistically significant against all benchmarks, including t-GARCH.

---

## Market-Risk Forecasting

Each model’s one-day predictive return distribution is converted into:

- 95% VaR
- 99% VaR
- 99% Expected Shortfall

For a long SPY position, losses are defined as:

$$
L_t = -r_t
$$

A VaR breach occurs when:

$$
L_t > \mathrm{VaR}_t
$$

VaR is formally evaluated using:

- breach rates;
- Kupiec unconditional coverage test;
- Christoffersen independence test;
- conditional coverage test.

Expected Shortfall is reported as a complementary tail-severity diagnostic rather than formally backtested.

---

## VaR Backtesting Results

| Model | VaR Level | Breaches | Breach Rate | Expected Rate | Kupiec p | Independence p | Conditional Coverage p | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 4-state HMM | 95% | 99 | 6.57% | 5.00% | 0.007 | 0.840 | 0.027 | Reject coverage |
| 4-state HMM | 99% | 24 | 1.59% | 1.00% | 0.033 | 0.378 | 0.070 | Mild undercoverage |
| Gaussian | 95% | 71 | 4.71% | 5.00% | 0.608 | 0.167 | 0.337 | Pass |
| Gaussian | 99% | 36 | 2.39% | 1.00% | 0.000 | 0.010 | 0.000 | Reject |
| Student-t | 95% | 96 | 6.37% | 5.00% | 0.019 | 0.023 | 0.005 | Reject |
| Student-t | 99% | 18 | 1.20% | 1.00% | 0.460 | 0.017 | 0.044 | Clustering issue |
| GARCH(1,1) | 95% | 88 | 5.84% | 5.00% | 0.143 | 0.945 | 0.342 | Pass |
| GARCH(1,1) | 99% | 32 | 2.12% | 1.00% | 0.000 | 0.711 | 0.001 | Reject |
| t-GARCH(1,1) | 95% | 92 | 6.11% | 5.00% | 0.056 | 0.867 | 0.159 | Pass |
| t-GARCH(1,1) | 99% | 23 | 1.53% | 1.00% | 0.056 | 0.362 | 0.107 | Pass |

### Interpretation

The t-GARCH model gives the strongest VaR backtesting performance. It is the only model that passes unconditional coverage, independence, and conditional coverage tests at both 95% and 99%.

The HMM produces informative risk forecasts, but its breach frequency is too high. Since the independence test does not reject, the issue appears less related to breach clustering and more related to tail-level calibration. This is consistent with Gaussian emissions being too thin-tailed within regimes.

The Student-t model performs reasonably at the 99% breach count level, but its breaches are clustered. This suggests that heavy tails alone can improve unconditional tail size, but cannot fully capture volatility clustering.

Gaussian GARCH adapts to volatility, but still fails at the 99% level, indicating that conditional volatility alone is not enough when innovations remain thin-tailed.

---

## Expected Shortfall Severity

Expected Shortfall measures the average model-implied loss inside the extreme tail. It is used here to compare tail severity, especially during COVID-era stress.

| Model | Avg VaR 99 | Avg ES 99 | Max ES 99 | COVID Avg ES 99 |
|---|---:|---:|---:|---:|
| 4-state HMM | 2.756 | 3.399 | 9.614 | 8.489 |
| Gaussian | 2.756 | 3.160 | 3.160 | 3.160 |
| Student-t | 3.494 | 5.882 | 5.882 | 5.882 |
| GARCH(1,1) | 2.566 | 2.939 | 16.954 | 9.377 |
| t-GARCH(1,1) | 2.875 | 3.647 | 20.898 | 11.944 |

### Interpretation

The Gaussian model has constant ES, so it cannot respond to stress periods.

The Student-t model has materially larger average ES because of heavier unconditional tails, but it is still time-invariant.

GARCH and t-GARCH produce sharply elevated stress-period ES because their conditional volatility rises during COVID. t-GARCH produces the largest stress ES because it combines volatility updating with heavy-tailed innovations.

The HMM also produces elevated COVID ES, indicating that regime probabilities shift toward high-volatility states. However, its stress response is less extreme than t-GARCH, consistent with smoother regime-based updating.

---

## Key Plots

The main report includes:

1. **Realised returns/losses versus VaR forecasts**  
   Shows where VaR breaches occur and whether forecasts rise during stress periods.

2. **HMM mixture volatility versus GARCH volatility**  
   Compares discrete regime-based volatility with continuous conditional volatility.

Empirically, the HMM and GARCH volatility forecasts are broadly similar outside major stress periods. During COVID, GARCH-type models spike more aggressively, while the HMM response is smoother. This helps explain why t-GARCH performs better for VaR calibration, while the HMM remains strong for average density forecasting.

---

## Main Conclusion

The empirical results suggest that discrete-regime and conditional-volatility models capture different aspects of equity return risk.

The **4-state HMM** provides the strongest full-density forecasts and captures persistent volatility-state structure in SPY returns. However, the **t-GARCH(1,1)** model delivers the strongest one-day VaR calibration, passing formal coverage and independence tests at both 95% and 99%.

This distinction is important from a model-validation perspective:

> A model that forecasts the average predictive distribution well is not necessarily the best model for extreme quantile calibration.

The HMM’s VaR undercoverage, despite relatively non-clustered breaches, suggests that its Gaussian emissions may understate extreme tail severity. By contrast, t-GARCH performs well because it combines both key ingredients for short-horizon market risk forecasting: time-varying volatility and heavy-tailed innovations.

---

## Practical Takeaways

- **HMMs are useful for regime-aware risk monitoring and density forecasting.**
- **t-GARCH is stronger as an operational one-day VaR benchmark.**
- **Static heavy tails help with unconditional tail size, but not with volatility timing.**
- **Conditional volatility helps with timing, but Gaussian innovations can still be too thin-tailed.**
- **Market-risk validation should evaluate both full-distribution accuracy and tail-specific calibration.**

---

## Limitations

This project is intentionally focused and should not be interpreted as a production market-risk engine.

Main limitations:

1. **Single asset**  
   The analysis is restricted to SPY and does not consider multi-asset portfolio effects.

2. **One-day horizon**  
   Results may differ for longer risk horizons.

3. **Fixed-origin design**  
   Parameters are estimated once using 2000–2019 data and held fixed during 2020–2025. This is useful for validation but may be conservative relative to rolling re-estimation.

4. **Gaussian HMM emissions**  
   The HMM may conflate regime changes with tail risk because each regime has Gaussian emissions. Student-t or skewed emissions could improve tail calibration.

5. **VaR-focused formal testing**  
   VaR is formally backtested, while ES is reported as a severity diagnostic. Formal ES backtesting is left outside scope.

---

## Repository Structure

```text
src/
  models/           # Gaussian, Student-t, GARCH, t-GARCH, HMM model classes
  plots/            # Plotting HMM states, smoothed probabilities, and VaR breaches
  utils/            # Information critera, Kupiec & Christoffersen tests, HMM fitting           

analysis.ipynb      # Main analysis 

figures/            # Key figures

tables/             # Key tables

README.md
requirements.txt
```

---

## How to Run

```bash
pip install -r requirements.txt
```

Then run the analysis notebook:

```text
analysis.ipynb
```

The key output tables and figures are saved to the `tables/` and `figures/` folders.

---

## Tools Used

- Python
- NumPy
- pandas
- SciPy
- matplotlib
- arch
- hmmlearn

---