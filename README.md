#  OMEGA — Ontario Multi-Objective Energy Grid Analyzer

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-LSTM-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Array-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-Vertex_AI-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-Version_Control-181717?style=for-the-badge&logo=github&logoColor=white)

![Optimization](https://img.shields.io/badge/MILP-Optimization-green?style=for-the-badge)
![Ensemble](https://img.shields.io/badge/Stacking-Ensemble-blueviolet?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-GBM-orange?style=for-the-badge)
![PuLP](https://img.shields.io/badge/PuLP-CBC_Solver-red?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

> **A Carbon-Priced Electricity Dispatch System for Ontario's Grid**  
> 🎓 DAMO 699 — Capstone Project | Master of Data Analytics | University of Niagara Falls, Canada | Spring 2026  
> 👨‍🏫 Supervisor: Prof. Hany Osman

---

## Table of Contents

1. [🔍 Project Overview](#1--project-overview)
2. [👥 Team](#2--team)
3. [🔬 Research Question & Hypotheses](#3--research-question--hypotheses)
4. [📁 Repository Structure](#4--repository-structure)
5. [📊 Dataset](#5--dataset)
6. [🤖 Module 1 — Stacking Ensemble Forecaster](#6--module-1--stacking-ensemble-forecaster)
7. [⚙️ Module 2 — Carbon-Priced MILP Optimizer](#7-️-module-2--carbon-priced-milp-optimizer)
8. [📈 Results & Findings](#8--results--findings)
9. [✅ Hypothesis Testing Outcomes](#9--hypothesis-testing-outcomes)
10. [🏛️ Key Policy Findings](#10-️-key-policy-findings)
11. [🛠️ Technology Stack](#11-️-technology-stack)
12. [🚀 Setup & Usage](#12--setup--usage)
13. [⚠️ Limitations](#13-️-limitations)
14. [📚 References](#14--references)

---

## 1. 🔍 Project Overview

Ontario's electricity grid faces a dual challenge: **meeting rising electricity demand reliably while reducing carbon emissions** in line with Canada's federal climate commitments. According to IESO (2025), provincial electricity demand is projected to grow by **75% by 2050**, driven by industrial electrification, EV manufacturing, data centres, and population growth.

OMEGA (Ontario Multi-Objective Energy Grid Analyzer) is a **two-module end-to-end analytics pipeline** that directly addresses this gap. It extends the ensemble forecasting framework of Osman et al. (2025) to provincial-level electricity dispatch and adds a **carbon-priced MILP optimization layer** — the first of its kind calibrated to Ontario's specific generation fleet and federal policy benchmarks.

### Why OMEGA Matters

- Ontario's generation fleet: Nuclear (~60%), Hydro (~22%), Gas (~9%), Wind (~5%), Solar (~1.5%), Biofuel (~0.5%)
- Gas is the **primary flexible balancing resource** but also the **dominant source of grid carbon emissions**
- Over the 10-year study period (2015–2025), the grid generated approximately **67.24 million tonnes of CO₂**
- **2,291 hours (2.4% of all hours)** exceeded 100 g CO₂/kWh — many preventable with smarter dispatch
- No predictive, data-driven system currently exists that simultaneously optimises for **cost AND carbon emissions**

### System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          OMEGA PIPELINE                          │
│           Ontario Multi-Objective Energy Grid Analyzer           │
│                                                                  │
│  ┌──────────────────────────┐                                    │
│  │ HISTORICAL DATA          │                                    │
│  │ Ontario IESO, 2015-2025  │                                    │
│  │ 96,264 hourly rows       │                                    │
│  │ 46 input features        │                                    │
│  └──────────────────────────┘                                    │
│                │                                                 │
│                ▼                                                 │
│  ┌──────────────────────────┐                                    │
│  │ MODULE 1                 │   ──►  24-hour forecasts:          │
│  │ Stacking Ensemble        │         • Electricity demand       │
│  │ Base: GBM · RF · LSTM    │         • Solar generation         │
│  │ Meta: Ridge regressor    │         • Wind generation          │
│  └──────────────────────────┘                                    │
│                │                                                 │
│                ▼                                                 │
│  ┌──────────────────────────┐                                    │
│  │ MODULE 2                 │   ──►  Optimal dispatch:           │
│  │ Carbon-Priced LP / MILP  │         • Cost vs. CO₂ Pareto      │
│  │ 6 techs · 10 constraints │         • Threshold ~ $80/t        │
│  │ $0-$250/t · 6 scenarios  │         • CO₂ -13%, gas 13->9h     │
│  └──────────────────────────┘                                    │
│                │                                                 │
│                ▼                                                 │
│  ┌──────────────────────────┐                                    │
│  │ MODULE 3                 │   ──►  Visual outputs:             │
│  │ Interactive Dashboard    │         • Forecast & dispatch      │
│  │ Streamlit · live controls│         • Cost-CO₂ Pareto          │
│  │ Date + carbon selector   │         • Scenario comparison      │
│  └──────────────────────────┘                                    │
│                                                                  │
│      │
└──────────────────────────────────────────────────────────────────┘


## 2. 👥 Team

| Name | Institution |
|------|-------------|
| Akash Kumar | University of Niagara Falls, Canada |
| Nidhi Jaiswal | University of Niagara Falls, Canada |
| Aena Parekh | University of Niagara Falls, Canada |
| Rushil Manish Jani | University of Niagara Falls, Canada |
| Shraddha Ginoya | University of Niagara Falls, Canada |

**Group:** Group 1  
**Supervisor:** Professor Hany Osman  
**Program:** Master of Data Analytics — DAMO 699 Capstone  
**Submission:** June 2026

---

## 3. 🔬 Research Question & Hypotheses

### 🔬 Research Question

| # | Research Question |
|---|-------------------|
| **RQ1** | 💡 At what carbon price does Ontario's electricity dispatch shift toward lower emissions? |
| **RQ2** | 🤖 Can a stacking ensemble outperform individual models for Ontario electricity forecasting? |
| **RQ3** | 💰 How much does carbon pricing actually cost Ontario households? |

---

### Hypothesis 1 — Ensemble Superiority

| | Statement |
|--|-----------|
| **H₀ (Null)** | The stacking ensemble (GBM + RF + LSTM → Ridge) will **not** outperform any individual base model on demand, solar, and wind forecasting on the held-out test set. |
| **H₁ (Alternative)** | The stacking ensemble will **match or outperform every individual base model** across all three targets. The Ridge meta-learner will automatically assign different weights to different models for different targets — leaning on LSTM for solar and GBM for demand — without any manual tuning. |

**Rationale:** Ensemble methods exploit complementary model strengths. LSTM captures temporal sequences; GBM captures non-linear feature interactions; RF provides variance reduction through bagging. The Ridge meta-learner learns the optimal combination without overfitting.

**Outcome: Partially Supported**  
Ridge R²=0.832 marginally outperforms GBM=0.830 for demand. More importantly, Ridge meta-learner coefficients reveal meaningful **automatic specialisation**: LSTM receives 0.905 weight for solar; GBM receives 0.608 for demand — achieved without any manual configuration. This automatic specialisation is the primary contribution of the ensemble architecture.

---

### Hypothesis 2 — Carbon Price Threshold Effect on Dispatch

*This hypothesis tests whether Ontario's grid responds to carbon pricing gradually or all at once — and identifies the exact price point where meaningful change occurs.*

| | Statement |
|--|-----------|
| **H₀ (Null)** | Increasing the carbon price will produce **gradual, continuous** reductions in gas dispatch and CO₂ emissions. Each dollar increase will result in a proportional decrease in emissions. |
| **H₁ (Alternative)** | Ontario's electricity grid will respond to carbon pricing in a **non-linear, step-change** way — below a critical price point, dispatch will remain completely unchanged, and above it a sudden reduction in gas dispatch and CO₂ emissions will occur. The federal OBPS rate ($80/tonne) will be sufficient to reduce daily CO₂ emissions by **at least 10%** compared to the zero-carbon-price baseline. |

**Rationale:** In Ontario's generation fleet, carbon-adjusted costs are non-linear. Once gas carbon-adjusted cost ($85 + ρ×0.490) crosses biofuel's cost ($105 + ρ×0.230), a discrete merit-order switch occurs. The crossover arithmetic predicts this at exactly $80/tonne.

**Threshold Arithmetic:**
```
At ρ = $80/tonne:
  Gas effective cost    = $85 + (0.490 × $80) = $124.2/MWh
  Biofuel effective cost = $105 + (0.230 × $80) = $123.4/MWh

Gas becomes marginally more expensive than biofuel at exactly $80/tonne
→ triggering the dispatch switch
```

**Outcome: Fully Supported**  
The $80/tonne threshold triggers a **13% CO₂ reduction** and **31% gas reduction** (38,730 → 26,850 MWh/day). Scenarios $0, $50, and $72 are completely identical; scenarios $80, $170, and $250 are completely identical — confirming the discrete step-change response.

---

### Hypothesis 3 — Household Affordability of Carbon Pricing

*This hypothesis addresses one of the most common public concerns about carbon policy — whether aggressive carbon pricing creates a meaningful financial burden for ordinary Ontario households.*

| | Statement |
|--|-----------|
| **H₀ (Null)** | Even at moderate carbon price levels, the additional monthly electricity cost per Ontario household will **exceed $5.00**, making aggressive carbon pricing financially burdensome for residents. |
| **H₁ (Alternative)** | Even at the highest carbon price scenario tested ($250/tonne), the additional monthly electricity cost per Ontario household will remain **under $1.00**, demonstrating that carbon pricing in Ontario's low-carbon grid is affordable for ordinary residents. |

**Rationale:** Ontario's grid is dominated by nuclear (zero emission, $29/MWh) and hydro (near-zero emission). Carbon pricing primarily affects gas dispatch (9% of mix), so the incremental cost passed to households is expected to be small relative to total electricity spend.

**Household Cost Calculation:**
```
Extra cost per MWh generated = $0.617/MWh (weighted across all technologies)
Average Ontario household monthly consumption = 900 kWh = 0.9 MWh
Additional monthly cost = $0.617 × 0.9 = $0.556 ≈ $0.56/month
This applies at ANY carbon price from $80 to $250/tonne
```

**Outcome: Fully Supported**  
At any carbon price from $80 to $250/tonne, the additional household cost is **only $0.56/month** — well below the $1.00 threshold. This is less than a coffee per month for the maximum achievable emissions reduction through pricing alone.

---

## 4. 📁 Repository Structure

```
OMEGA-Ontario-Multi-Objective-Energy-Grid-Analyzer/
│
├── final_forcast.ipynb                    # Module 1: 14-step stacking ensemble forecasting pipeline
├── optimizer.ipynb                        # Module 2: 10-step carbon-priced MILP dispatch optimizer
├── merged_ontario_energy_dataset.csv      # Master dataset: 96,264 hourly records (2015–2025), 46 features
│
├── omega_models/                          # Saved model files (Git LFS)
│   ├── gbm_demand.pkl                     # GBM model for demand forecasting
│   ├── gbm_solar.pkl                      # GBM model for solar forecasting
│   ├── gbm_wind.pkl                       # GBM model for wind forecasting
│   ├── rf_demand.pkl                      # Random Forest model for demand
│   ├── rf_solar.pkl                       # Random Forest model for solar
│   ├── rf_wind.pkl                        # Random Forest model for wind
│   ├── lstm_demand.keras                  # LSTM model for demand forecasting
│   ├── lstm_solar.keras                   # LSTM model for solar forecasting
│   ├── lstm_wind.keras                    # LSTM model for wind forecasting
│   └── ridge_meta.pkl                     # Ridge meta-learner (stacking combiner)
│
├── outputs/
│   ├── figures/                           # All generated visualizations
│   │   ├── 01_demand_distribution.png
│   │   ├── 02_demand_heatmap.png
│   │   ├── 03_demand_season_weekend.png
│   │   ├── 04_energy_mix.png
│   │   ├── 05_energy_mix_by_year.png
│   │   ├── 06_renewable_share_trend.png
│   │   ├── 07_gas_dispatch_patterns.png
│   │   ├── 08_gas_heatmap.png
│   │   ├── 09_gas_vs_demand.png
│   │   ├── 10_carbon_analysis.png
│   │   ├── 11_high_carbon_patterns.png
│   │   ├── 12_cost_analysis.png
│   │   ├── 13_weather_renewable_scatter.png
│   │   ├── 14_correlation_matrix.png
│   │   └── 15_lag_validation.png
│   └── optimization/                      # MILP optimizer outputs
│
├── forecast_for_milp_2026-06-10.csv       # Sample forecast output (input to optimizer)
├── forecast_2026-06-10.png                # Forecast visualization
└── README.md                              # This file
```

---

## 5. 📊 Dataset

### `merged_ontario_energy_dataset.csv`

**96,264 hourly records | January 2015 – December 2025 | 46 features | Zero missing values**

### Data Sources

| Source | Key Variables | Coverage | Records |
|--------|--------------|----------|---------|
| **IESO Open Data** (ieso.ca) | Hourly demand (MW), generation by fuel type (nuclear, hydro, solar, wind, gas, biofuel) | 2015–2025, hourly | 96,264 |
| **NASA POWER** | Temperature (°C), wind speed (10m/50m), solar irradiance (kWh/m²), relative humidity | Ontario lat/lon, hourly | 96,264 |
| **Environment Canada (ECCC)** | CO₂ emission factors by generation technology (kg/MWh) | Annual averages | Derived |
| **Our World in Data** | Long-term Canada energy trends for contextual benchmarking | Yearly, 2000–2025 | Reference |

*All sources are publicly available open government or academic datasets. No proprietary or personal data is used.*

### Summary Statistics

| Variable | Mean | Std Dev | Min | Max | Unit |
|----------|------|---------|-----|-----|------|
| Ontario Demand | 14,836 | 2,104 | 9,891 | 24,588 | MW |
| Nuclear Generation | 8,854 | 1,041 | 3,200 | 11,400 | MW |
| Hydro Generation | 3,412 | 946 | 712 | 9,024 | MW |
| Gas Generation | 750 | 934 | 0 | 5,189 | MW |
| Wind Generation | 741 | 714 | 0 | 4,302 | MW |
| Solar Generation | 209 | 364 | 0 | 2,487 | MW |
| Temperature | 8.4 | 11.2 | -32.1 | 38.7 | °C |

> Gas generation mean of 750 MW vs peak of 5,189 MW illustrates the high variability that motivates proactive dispatch optimisation.

### Feature Engineering (46 Features)

**Temporal Features**
- `hour` (0–23), `day_of_week` (0–6), `month` (1–12), `season_num` (0–3), `year`
- `is_weekend` (binary), `is_holiday` (Ontario statutory holidays binary)
- `fiscal_quarter` (Q1–Q4), `is_peak_hour` (binary)

**Lag Features**
- `demand_lag_1h`, `demand_lag_2h`, `demand_lag_24h`, `demand_lag_48h`, `demand_lag_168h`
- `solar_lag_24h`, `wind_lag_24h` (same hour yesterday)

**Rolling Statistics**
- `demand_rolling_mean_24h`, `demand_rolling_std_24h`
- `demand_rolling_mean_168h`, `demand_rolling_std_168h`

**Weather Features**
- `temperature_c`, `solar_irradiance_kwh_m2`, `wind_speed_10m_ms`, `wind_speed_50m_ms`
- `relative_humidity_pct`, `cloud_cover` (derived from irradiance ratio)
- `clearsky_solar_kwh_m2`

**Interaction Features**
- `temperature_x_weekend`, `solar_x_hour` (solar irradiance × hour of day)
- `temp_squared` (non-linear heating/cooling effect)

### Data Splits

A strict **chronological 90/5/5 split** was applied to preserve temporal order and prevent future data leakage:

| Split | Period | Rows | Purpose |
|-------|--------|------|---------|
| Training (90%) | Jan 2015 – Aug 2024 | ~86,400 | Model training and cross-validation |
| Validation (5%) | Sep 2024 – Jun 2025 | ~4,800 | Hyperparameter selection only |
| Test (5%) | Jul 2025 – Dec 2025 | ~4,800 | Final evaluation (held out, used once) |

> The test set was **never inspected** during model development — it was used only once to report final metrics. Shuffling would be inappropriate for time series data.

### Three-Layer Analog Proxy System (Future Date Inference)

For forecasting dates beyond the dataset (e.g., any 2026 date), a weighted analog system was developed:

| Layer | Weight | Match Criteria | Rationale |
|-------|--------|----------------|-----------|
| 1 | 50% | Same day-of-week + same month + same hour | Most similar calendar and seasonal context |
| 2 | 30% | Same month + same hour | Relaxed match when Layer 1 has <10 samples |
| 3 | 20% | Same season + same hour | Broadest fallback for robust estimation |

---

## 6. 🤖 Module 1 — Stacking Ensemble Forecaster

**File:** `final_forcast.ipynb`

### Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Imports & environment setup (TensorFlow, scikit-learn, pandas, numpy) |
| 2 | Load dataset — 96,264 rows, 46 features, datetime indexing |
| 3 | Feature engineering + chronological 90/5/5 split |
| 4 | GBM cross-validation (5-fold chronological) |
| 5 | RF cross-validation (5-fold chronological) |
| 6 | GBM hyperparameter experiments (4 configurations on validation set) |
| 7 | RF hyperparameter experiments (4 configurations on validation set) |
| 8 | Final GBM + RF training with best parameters |
| 9 | LSTM cross-validation (3-fold — limited by training cost ~10 min/fold) |
| 10 | LSTM final training with early stopping (patience=10) |
| 11 | Cross-validation summary table (GBM + RF + LSTM) |
| 12 | Ridge meta-learner training on validation out-of-fold predictions |
| 13 | Final metrics + evaluation on held-out test set |
| 14 | Predict any target date + export 24-hour forecast CSV to MILP |

### Base Model Architecture

Three base learners trained **independently** for each of three forecast targets (demand, solar, wind) = 9 base models total:

| Model | Best For | Key Strength | Final Parameters |
|-------|----------|-------------|-----------------|
| **GBM** (Gradient Boosting) | Demand, tabular features | Non-linear feature interactions, weather × calendar | n_est=200, depth=7, lr=0.1 |
| **RF** (Random Forest) | Robust baseline all targets | Variance reduction via bagging, outlier robustness | n_est=300, depth=10, n_jobs=-1 |
| **LSTM** | Solar, temporal sequences | Long-range temporal memory, daily/weekly cycles | LSTM(64)→Drop(0.2)→LSTM(32)→Drop(0.1)→Dense(1) |

### LSTM Implementation Details

- **Sequence length:** 24 hours (each prediction uses previous 24 hours as context)
- **Architecture:** `LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.1) → Dense(1)`
- **Optimizer:** Adam with learning rate 0.001
- **Loss function:** Mean Squared Error
- **Early stopping:** patience=10 on validation loss
- **Critical detail:** Target variable (y) scaled independently with its own `MinMaxScaler` to [0,1] before training — without this, raw MW values (9,000–24,000 MW) saturate the sigmoid activation in LSTM gates, producing R² near zero

**LSTM Early Stopping Results:**

| Target | Stopped Epoch | Val Loss at Stop | Interpretation |
|--------|--------------|------------------|----------------|
| Demand | Epoch 37 | 0.002639 | Stable convergence, no overfitting |
| Solar | Epoch 44 | 0.001228 | Smooth daily curve well captured |
| Wind | Epoch 9 | 0.018565 | Early stop — wind is inherently noisy |

### Hyperparameter Selection Protocol

The correct academic protocol was followed:
1. 5-fold chronological cross-validation to establish baseline
2. 4-configuration validation experiments to select optimal hyperparameters
3. Final retraining on full training set with winning parameters
4. Test set used only once at the very end

**GBM Hyperparameter Experiments:**

| Exp | n_est | Depth | D-MAPE | S-MAPE | W-MAPE | Selected? |
|-----|-------|-------|--------|--------|--------|-----------|
| 1 | 150 | 5 | 1.51% | 75.26% | 79.00% | |
| 2 | 100 | 3 | 1.66% | 97.70% | 79.54% | ❌ Underfits solar |
| **3** | **200** | **7** | **1.51%** | **73.87%** | **79.65%** | **✅ Selected** |
| 4 | 300 | 5 | 1.50% | 77.24% | 79.01% | |

**RF Hyperparameter Experiments:**

| Exp | n_est | Depth | D-MAPE | S-MAPE | W-MAPE | Selected? |
|-----|-------|-------|--------|--------|--------|-----------|
| 1 | 150 | 10 | 1.60% | 75.37% | 79.37% | |
| 2 | 100 | 5 | 1.85% | 87.26% | 79.37% | |
| 3 | 200 | 5 | 1.54% | 76.87% | 80.49% | |
| **4** | **300** | **10** | **1.59%** | **75.42%** | **79.28%** | **✅ Selected** |

### Cross-Validation Results

5-fold chronological CV for GBM and RF; 3-fold for LSTM (training cost ~10 min/fold/target):

| Model | Target | R² (±std) | MAE (±std) | MAPE (±std) |
|-------|--------|-----------|------------|-------------|
| GBM (5-fold) | Demand | 0.891 ± 0.000 | 242.3 ± 0.4 MW | 1.54 ± 0.00% |
| GBM (5-fold) | Solar | 0.979 ± 0.000 | 67.9 ± 0.4 MW | 62.40 ± 1.80% |
| GBM (5-fold) | Wind | 0.274 ± 0.007 | 398.2 ± 2.1 MW | 86.02 ± 3.31% |
| RF (5-fold) | Demand | 0.877 ± 0.001 | 256.8 ± 0.7 MW | 1.63 ± 0.00% |
| RF (5-fold) | Solar | 0.978 ± 0.000 | 69.2 ± 0.5 MW | 61.80 ± 2.11% |
| RF (5-fold) | Wind | 0.273 ± 0.007 | 398.6 ± 2.2 MW | 86.11 ± 3.47% |
| LSTM (3-fold) | Demand | 0.889 ± 0.001 | 244.6 ± 1.0 MW | 1.56 ± 0.01% |
| LSTM (3-fold) | Solar | 0.980 ± 0.001 | 68.8 ± 1.0 MW | 69.09 ± 5.47% |
| LSTM (3-fold) | Wind | 0.279 ± 0.007 | 396.8 ± 2.6 MW | 85.93 ± 4.89% |

> Near-zero std across folds confirms stable generalisation. High solar/wind MAPE is expected — MAPE is inflated by near-zero nighttime values. R² is the primary accuracy metric.

### Ridge Meta-Learner

A **Ridge regression** meta-learner (`alpha=1.0`, L2 regularisation) combines base model predictions. It is trained on **validation set out-of-fold predictions only** to prevent data leakage. The L2 penalty prevents overfitting when base model predictions are correlated.

**Ridge Coefficients — Automatic Model Specialisation:**

| Target | GBM coef | RF coef | LSTM coef |
|--------|---------|---------|-----------|
| Demand | **0.6082** | -0.1217 | 0.5149 |
| Solar | -0.0043 | 0.0897 | **0.9053** |
| Wind | 0.2270 | 0.2967 | 0.4421 |

**Interpretation:**
- **Solar → LSTM dominates (0.905):** The smooth daily generation curve (zero at dawn, peak at noon, zero at dusk) has temporal structure that LSTM's recurrent memory captures directly. Tree-based models without memory treat each hour as independent and miss intra-day continuity.
- **Demand → GBM leads (0.608) + LSTM contributes (0.515):** GBM captures complex non-linear weather × calendar × lag interactions. LSTM adds temporal continuity. RF is suppressed (-0.122) because GBM already covers the same signal.
- **Wind → Distributed across all three:** Even distribution reflects wind's inherent unpredictability — no single architecture consistently outperforms, so Ridge hedges by combining all three.

---

## 7. ⚙️ Module 2 — Carbon-Priced MILP Optimizer

**File:** `optimizer.ipynb`

### Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Imports & configuration — auto-extracts date and season from filename |
| 2 | Load & validate 24-hour forecast CSV (24 rows × 4 columns: hour, demand, solar, wind) |
| 3 | Forecast visualization — demand curve vs renewable availability caps |
| 4 | Fuel parameters & auto-calibrated seasonal constraints |
| 5 | MILP solver formulation (10 constraints, C1–C10) |
| 6 | Run all 6 carbon price scenarios sequentially |
| 7 | Operator-style dispatch readout per scenario |
| 8 | Dispatch visualization (stacked generation by technology per hour) |
| 9 | Pareto frontier — Cost vs. Emissions across scenarios |
| 10 | Real-world impact analysis (CO₂ saved, trees equivalent, household cost) |

### Objective Function

The MILP minimises total **carbon-adjusted generation cost** over a 24-hour horizon:

```
Minimise: Σᵢ Σₜ (cᵢ + ρ·eᵢ) × Pᵢₜ × Δt

Where:
  i  ∈ {nuclear, hydro, gas, wind, solar, biofuel}  — generation technology
  t  ∈ {1, 2, ..., 24}                               — hour of day
  Pᵢₜ (MW)   — dispatched power from technology i at hour t  [decision variable]
  cᵢ ($/MWh) — marginal fuel cost per technology
  ρ ($/tonne CO₂) — carbon price policy parameter
  eᵢ (tonne CO₂/MWh) — emission intensity factor
  Δt = 1 hour — dispatch interval

The composite term (cᵢ + ρ·eᵢ) = carbon-adjusted marginal cost.
It increases with ρ for carbon-emitting technologies and stays flat for zero-emission sources.
```

### Generation Fleet Parameters

| Technology | Cost ($/MWh) | Emissions (t CO₂/MWh) | Capacity (MW) | Min Output (MW) | Ramp (MW/h) | Source |
|------------|-------------|----------------------|---------------|-----------------|-------------|--------|
| Nuclear | $29 | 0.012 | 9,600 | 8,160 (85%) | 200 | IESO |
| Hydro | $45 | 0.024 | 8,500 | 0 | 3,000 | IESO |
| Gas | $85 | 0.490 | 12,000 | 2,400 (if committed) | 3,500 | IESO |
| Wind | $35 | 0.000 | 5,300 | 0 | 5,500 | IRENA |
| Solar | $45 | 0.000 | 2,600 | 0 | 2,800 | IRENA |
| Biofuel | $105 | 0.230 | 495 | 0 | 150 | ECCC |

> Nuclear capacity reflects Pickering B retirement and Darlington/Bruce refurbishment schedules. Gas cost at $80/tonne: effective = $85 + 0.490×$80 = $124.2/MWh, exceeding biofuel at $105 + 0.230×$80 = $123.4/MWh — explaining the $80 threshold.

### Constraints (C1–C10)

| ID | Name | Description |
|----|------|-------------|
| **C1** | Power balance | Generation must exactly meet forecast demand Dₜ each hour |
| **C2** | Renewable availability | Solar/wind capped at Module 1 forecast outputs (cannot exceed availability) |
| **C3** | Capacity bounds | All generation bounded by nameplate capacity |
| **C4** | Nuclear baseload floor | Nuclear must remain ≥ 85% of nameplate (IESO reactor stability requirement) |
| **C5** | Ramp limits | Physical ramp rates per technology (MW/hour) enforced between consecutive hours |
| **C6** | Hydro daily budget | Reservoir water constraint calibrated by season (20–35% of daily demand) |
| **C7** | CO₂ ceiling | Ontario EPS regulatory daily emissions limit (50–65 kt/day by season) |
| **C8** | Renewable floor | Ontario clean energy procurement minimum (10–20% by season) |
| **C9** | Gas peak cap | IESO peaker dispatch rules — gas supply-demand gap capped at peak hours |
| **C10** | Gas unit commitment (MILP) | Binary variable uₜ: if gas committed, minimum stable output = 20% nameplate (2,400 MW); zero otherwise. **This makes the problem MILP rather than LP.** |

> C10 introduces 24 binary variables (one per hour) — the MILP component that enforces minimum stable output for gas turbines.

### Seasonal Constraint Calibration

| Season | Hydro % | CO₂ Ceiling (kt/day) | Renewable Floor | Notes |
|--------|---------|----------------------|-----------------|-------|
| Winter | 20% | 65 kt | 10% | Low solar, high demand |
| Spring | 35% | 50 kt | 15% | High hydro (snowmelt) |
| Summer | 25% | 65 kt | 20% | Peak solar, AC demand |
| Fall | 28% | 58 kt | 12% | Transition season |

### Carbon Price Scenarios

| Scenario | ρ ($/tonne) | Policy Anchor |
|----------|------------|---------------|
| No carbon pricing | $0 | Baseline economic dispatch — reference scenario |
| Light industrial | $50 | Below 2022 federal benchmark — tests sub-threshold behaviour |
| Ontario EPS 2025 | $72 | Ontario Emission Performance Units (2025 market traded price) |
| **Federal OBPS 2024** | **$80** | **Federal Output-Based Pricing System — benchmark for large emitters** |
| Federal schedule 2030 | $170 | Legislated rate under Bill C-12 carbon pricing amendments |
| Net-zero aligned | $250 | IPCC AR6 1.5°C pathway shadow price (high estimate) |

### Solver Details

| Component | Detail |
|-----------|--------|
| Library | PuLP v2.7 |
| Solver | Coin-or Branch-and-Cut (CBC) |
| Continuous variables | 144 (Pᵢₜ for 6 technologies × 24 hours) |
| Binary variables | 24 (uₜ — one per hour for gas unit commitment) |
| Total constraints | ~500 (10 types across 24 hours) |
| Solve time (1 scenario) | < 1 second (global optimality) |
| Solve time (6 scenarios) | ~6 seconds total |
| Compute platform | GCP Vertex AI Workbench (n1-standard-4: 4 vCPU, 16 GB RAM) |

---

## 8. 📈 Results & Findings

### 8.1 Forecasting Results — Final Test Set (Jul–Dec 2025)

| Target | GBM R² | RF R² | LSTM R² | **Ridge R²** | Ridge MAPE | Ridge MAE | Industry Benchmark |
|--------|--------|-------|---------|-------------|-----------|----------|--------------------|
| Demand | 0.830 | 0.804 | 0.826 | **0.832** | **1.59%** | 240.7 MW | < 3% MAPE ✅ |
| Solar | 0.984 | 0.983 | 0.985 | **0.985** | 41.79%* | 66.8 MW | R² > 0.95 ✅ |
| Wind | 0.302 | 0.301 | 0.305 | **0.306** | 89.49%* | 396.3 MW | Physically limited |

*\*High MAPE for solar/wind is inflated by near-zero nighttime values. R² is the primary metric for these targets.*

**Key observations:**
- **Demand:** MAPE=1.59% comfortably beats the 3% industry benchmark. MAE of 240.7 MW = 1.6% of average demand — operationally acceptable for day-ahead planning.
- **Solar:** R²=0.985 confirms solar generation is highly predictable given irradiance and temporal features.
- **Wind:** R²=0.306 is consistent across all three models (0.302, 0.301, 0.305) and all CV folds (0.274–0.279). This convergence confirms the limitation is **in the data** (inherent wind unpredictability), not the models. Ontario's wind fleet spans geographically dispersed regions (Southwestern Ontario, Bruce Peninsula, Simcoe County) with partially uncorrelated weather systems.

### 8.2 Optimization Results — June 10, 2026 (Summer)

Forecast demand: 385,198 MWh/day | Peak: 16,305 MW

| Scenario | ρ ($/t) | Cost ($M/day) | Emissions (t CO₂) | Renewable % | Gas Hours | Gas (MWh) |
|----------|---------|--------------|------------------|-------------|-----------|-----------|
| No carbon pricing | $0 | $15.02M | 24,054 | 30.1% | 13h | 38,730 |
| Light industrial | $50 | $15.02M | 24,054 | 30.1% | 13h | 38,730 |
| Ontario EPS 2025 | $72 | $15.02M | 24,054 | 30.1% | 13h | 38,730 |
| **Federal OBPS 2024** | **$80** | **$15.25M** | **20,965** | **33.2%** | **9h** | **26,850** |
| Federal 2030 | $170 | $15.25M | 20,965 | 33.2% | 9h | 26,850 |
| Net-zero aligned | $250 | $15.25M | 20,965 | 33.2% | 9h | 26,850 |

**Scenarios $0, $50, $72 are identical. Scenarios $80, $170, $250 are identical. Confirms discrete threshold effect.**

### 8.3 Real-World Impact Analysis

| Scenario | Gas (MWh) | CO₂ Saved (t/day) | Trees/Year Equiv. | HH Extra Cost/Month |
|----------|-----------|------------------|-------------------|---------------------|
| $0 | 38,730 | 0 | 0 | $0.00 |
| $50 | 38,730 | 0 | 0 | $0.00 |
| $72 | 38,730 | 0 | 0 | $0.00 |
| **$80** | **26,850** | **5,821** | **268,258** | **+$0.56** |
| $170 | 26,850 | 5,821 | 268,258 | +$0.56 |
| $250 | 26,850 | 5,821 | 268,258 | +$0.56 |

**Calculations:**
```
CO₂ saved = gas displaced (11,880 MWh) × 490 kg/MWh = 5,821,200 kg = 5,821 tonnes/day
Trees equiv = 5,821,200 ÷ 21.7 kg/tree/year = 268,258 trees/year
HH cost = $0.617/MWh extra × 0.9 MWh/month (average household) = $0.56/month
```

### 8.4 Critical Carbon Price Threshold Finding

A critical analysis was conducted to determine whether 50% gas displacement could be achieved within $0–$250/tonne.

**Result: 50% gas displacement is NOT achievable through carbon pricing alone within $0–$250/tonne.**

The maximum achievable displacement is **31%** (at $80+/tonne). Beyond $80, the dispatch ceiling is reached because biofuel nameplate capacity (495 MW) is exhausted — further gas displacement requires new capacity, not higher prices.

---

## 9. ✅ Hypothesis Testing Outcomes

| Hypothesis | Test Statement | Evidence | Outcome |
|-----------|---------------|----------|---------|
| **H1** — Ensemble Superiority | Ridge outperforms all individual base models | Ridge R²=0.832 > GBM=0.830 (demand); LSTM coef=0.905 for solar confirms automatic specialisation | **Partially Supported** |
| **H2** — Carbon Price Threshold | $80/tonne reduces CO₂ ≥10% vs $0 baseline | Gas: 38,730→26,850 MWh (-31%); CO₂: -5,821t (-13%); confirmed step-change response | **Fully Supported** |
| **H3** — Household Affordability | Household cost stays under $1.00/month at any carbon price | Additional cost at $80–$250/tonne = +$0.56/month per household (900 kWh/mo × $0.617/MWh) | **Fully Supported** |

---

## 10. 🏛️ Key Policy Findings

### Finding 1 — The $80/tonne Threshold Is Real and Empirically Validated

The federal OBPS benchmark of $80/tonne is not arbitrary — it is the **mathematically exact point** where gas becomes more expensive than biofuel in Ontario's merit order:

```
$72/tonne (below threshold): Gas = $85 + 0.490×72 = $120.3/MWh | Biofuel = $105 + 0.230×72 = $121.6/MWh
                              Gas still cheaper → No dispatch change

$80/tonne (at threshold):    Gas = $85 + 0.490×80 = $124.2/MWh | Biofuel = $105 + 0.230×80 = $123.4/MWh
                              Gas becomes more expensive → Dispatch switches
```

### Finding 2 — Carbon Pricing Alone Cannot Achieve Deep Decarbonisation

The 31% gas displacement ceiling at $250/tonne is a structural constraint — not a pricing failure. Ontario needs:
- New renewable capacity (wind farms, solar installations)
- Grid-scale battery storage or pumped hydro
- Expanded biofuel/biomass capacity (currently capped at 495 MW nameplate)
- Demand response programs

### Finding 3 — The Federal 2030 Schedule Adds No Incremental Value in Ontario's Electricity Sector

The legislated increase from $80 to $170/tonne by 2030 under Bill C-12 adds **zero incremental emissions reduction** in Ontario's electricity sector beyond what $80 already achieves. The dispatch ceiling is reached at $80 — not because the price is too low, but because there is no more biofuel capacity to dispatch.

### Policy Recommendations

1. **Maintain federal OBPS at minimum $80/tonne** — this is the empirically validated minimum effective price
2. **Invest in biofuel and renewable capacity** — adding 500–1,000 MW of biofuel or equivalent storage would extend carbon pricing's dispatch impact
3. **Implement demand response programs** — smart meters, interruptible commercial loads, EV charging schedules reduce residual gas need without capital-intensive new generation
4. **Extend OMEGA to 7-day MILP horizon** — multi-day unit commitment would better capture nuclear and gas commitment economics
5. **Apply OMEGA to winter scenarios** — winter peak demand (~22,000 MW) would reveal different threshold dynamics
6. **Communicate household affordability** — at $250/tonne, Ontario households pay only $0.56/month more — less than a coffee per month

---

## 11. 🛠️ Technology Stack

| Component | Tool | Version | Purpose |
|-----------|------|---------|---------|
| Language | Python | 3.11 | All modules |
| Tree models | scikit-learn | 1.3+ | GBM, RF, Ridge meta-learner, CV utilities |
| Deep learning | TensorFlow / Keras | 2.15 | LSTM architecture, training, early stopping |
| Optimization | PuLP + CBC solver | 2.7 | MILP formulation and solving |
| Data | pandas, numpy | — | Feature engineering, splitting, manipulation |
| Visualization | Plotly, Matplotlib | — | Interactive charts, dispatch stacked bar, Pareto frontier |
| Compute | GCP Vertex AI Workbench | n1-standard-4 | 4 vCPU, 16 GB RAM — all training and optimization |
| Version control | GitHub + Git LFS | — | Code; LFS for large .keras/.pkl model files |

---

## 12. 🚀 Setup & Usage

### Prerequisites

```bash
pip install scikit-learn tensorflow pulp pandas numpy plotly matplotlib
```

### Step 1 — Run Forecasting Module

```bash
jupyter notebook final_forcast.ipynb
```

Run all 14 steps sequentially. At **Step 14**, set `TARGET_DATE` to your desired date. This generates a file `forecast_for_milp_YYYY-MM-DD.csv` (24 rows × 4 columns: hour, demand, solar, wind).

### Step 2 — Run Optimization Module

```bash
jupyter notebook optimizer.ipynb
```

In **Step 1**, update:
```python
FORECAST_FILE = 'forecast_for_milp_YYYY-MM-DD.csv'
```

Run all 10 steps. The optimizer automatically detects season, calibrates constraints, runs all 6 carbon price scenarios, and outputs the Pareto frontier and real-world impact analysis.

### Output Files

| File | Description |
|------|-------------|
| `forecast_for_milp_YYYY-MM-DD.csv` | 24-hour forecast (input to optimizer) |
| `outputs/figures/` | All visualization PNGs |
| `outputs/optimization/` | MILP results, dispatch tables, Pareto frontier |
| `omega_models/` | Saved model files (reload without retraining) |

---

## 13. ⚠️ Limitations

### Forecasting Limitations
- **Wind R²=0.306** reflects inherent unpredictability consistent across all models and CV folds — this is a data challenge, not a model failure. Spatially resolved per-farm wind data could improve this.
- The **three-layer analog proxy** may underperform in extreme weather events (heat domes, ice storms) not well represented in historical analogs.
- **LSTM architecture** was fixed rather than tuned across multiple configurations due to training cost (~10 min/fold/target).
- ARIMA and SVR proposed in original plan were replaced by LSTM and RF — originally proposed ensemble comparison was not fully completed.

### Optimization Limitations
- **24-hour planning horizon** — real IESO dispatch uses 7+ day unit commitment windows capturing longer-run nuclear and gas commitment costs.
- **Energy storage not modelled** — Ontario's Sir Adam Beck pumped storage and emerging grid-scale batteries are omitted.
- **Demand response not included** — interruptible load programs not in supply-demand balance.
- **Single-node model** — transmission constraints abstracted; real dispatch respects N-1 security constraints.
- **Gas startup costs simplified** — non-convex startup costs and heat-up times beyond 20% minimum stable output are approximated.

### Data Limitations
- **CO₂ emission factors** from ECCC NIR are annual averages — hourly marginal emission factors would be more accurate.
- **Gas prices fixed** at $85/MWh — real prices vary with Henry Hub spot prices.
- **Dataset ends 2025** — Pickering B retirement (2026–2030) and refurbishment schedules reflected in capacity parameters but not training patterns.
- **NASA POWER weather data** provides gridded estimates, not weather station observations — potential spatial smoothing errors.

---

## 14. 📚 References

- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*, 785–794.
- Duman, A. C., & Güler, Ö. (2024). Machine learning applications in energy dispatch optimization. *Renewable and Sustainable Energy Reviews, 189*, 113874.
- Ehsani, B., Jafari, A., & Mousavi, M. J. (2024). Price forecasting in the Ontario electricity market via TriConvGRU. *Applied Energy, 357*, 122463.
- Environment and Climate Change Canada. (2024). National Inventory Report 1990–2022.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics, 29*(5), 1189–1232.
- Goulder, L. H., & Hafstead, M. A. C. (2018). *Confronting the Climate Challenge.* Columbia University Press.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780.
- IESO. (2025). Electricity demand in Ontario to grow by 75% by 2050. https://www.ieso.ca
- IRENA. (2024). Renewable power generation costs in 2023. https://www.irena.org
- Osman, H., et al. (2025). Ensemble forecasting for EV charging load prediction.
- Ren, Y., Suganthan, P. N., & Srikanth, N. (2015). Ensemble methods for wind and solar power forecasting. *Renewable and Sustainable Energy Reviews, 50*, 82–91.
- Shadoul, M., et al. (2024). MILP dispatch engine for hybrid renewable power stations. *Energies, 17*(13), 3281.
- Wolf, A., & Moler, B. (2022). Stacking ensemble methods for renewable energy forecasting. *Energy and AI, 8*, 100148.
- Zhou, Z., Wu, W., & Tang, S. (2022). Short-term wind power forecasting with stacking ensemble. *Renewable Energy, 189*, 618–631.

---

*OMEGA — DAMO 699 Capstone Project | Master of Data Analytics | University of Niagara Falls, Canada | June 2026*
