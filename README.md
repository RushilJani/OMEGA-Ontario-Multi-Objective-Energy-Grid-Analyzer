# ⚡ OMEGA — Ontario Multi-Objective Energy Grid Analyzer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/TensorFlow-LSTM-orange?style=for-the-badge&logo=tensorflow"/>
  <img src="https://img.shields.io/badge/XGBoost-Ensemble-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PuLP-Optimization-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PowerBI-Dashboard-yellow?style=for-the-badge&logo=powerbi"/>
</p>


> **DAMO 699 — Capstone Project | Master of Data Analytics**
> University of Niagara Falls | Spring 2026
> Supervisor: Prof. Hany Osman

---

## 📌 Project Overview

**OMEGA** is an intelligent energy dispatch optimization system for Ontario's electricity grid. It addresses a critical real-world challenge: Ontario's electricity demand is projected to grow by **75% by 2050**, yet grid operators currently dispatch energy sources manually without a data-driven system that simultaneously optimizes for both **cost** and **carbon emissions**.

OMEGA solves this by building a **three-layer analytical pipeline**:

1. **Custom Stacking Ensemble** — predicts hourly electricity demand, solar output, and wind output
2. **Multi-Objective Linear Programming Optimizer** — recommends the optimal hourly dispatch schedule to minimize cost and carbon emissions simultaneously
3. **Power BI Decision Support Dashboard** — visualizes results and enables scenario simulation for decision-makers

---

## 🧠 How It Works

### Layer 1 — Custom Stacking Ensemble (New Model)

Instead of using a single off-the-shelf model, OMEGA builds a **novel stacking ensemble architecture** where multiple base models make independent predictions, and a custom-built meta-learner learns the optimal weighted combination.

```
BASE MODELS (Level 0)              META-LEARNER (Level 1)
┌─────────────────────┐
│  LSTM               │ ──→
│  (temporal patterns)│          ┌──────────────────────┐
├─────────────────────┤          │  Ridge Regression    │ ──→  Final Prediction
│  XGBoost            │ ──→      │  (custom meta-model) │      (most accurate)
│  (feature relations)│          └──────────────────────┘
├─────────────────────┤
│  ARIMA              │ ──→
│  (seasonal cycles)  │
└─────────────────────┘
```

**Three separate ensembles are built:**

| Ensemble | Target Variable | Base Models | Meta-Learner |
|---|---|---|---|
| A — Demand | Hourly electricity demand (MW) | LSTM + XGBoost + ARIMA | Ridge Regression |
| B — Solar | Hourly solar generation (MW) | XGBoost + Random Forest + SVR | Ridge Regression |
| C — Wind | Hourly wind generation (MW) | XGBoost + Random Forest + LSTM | Ridge Regression |

---

### Layer 2 — Multi-Objective LP Optimization Engine

Ensemble predictions feed into a **linear programming optimizer** (Python PuLP) that solves for the optimal hourly dispatch across all 6 energy sources.

**Objective:** Minimize **(α × Total Cost) + (β × Total Carbon Emissions)**

**Subject to constraints:**
- ✅ Supply–demand balance: total dispatch = forecasted demand every hour
- ✅ Capacity limits: each source ≤ its maximum installed capacity
- ✅ Ramp rate constraints: gas cannot change beyond defined limits hour-to-hour
- ✅ Renewable priority: solar and wind dispatched at full available capacity first
- ✅ Non-negativity: all dispatch values ≥ 0 MW

---

### Layer 3 — Power BI Decision Support Dashboard

Interactive dashboard for grid operators and policymakers featuring:
- 📊 Optimized vs. historical actual dispatch comparison
- 💰 Cost savings (CAD) and CO₂ reduction (tonnes) KPIs
- 🚨 High-carbon hour alert system
- 🎛️ Scenario simulator with adjustable α/β weights

---

## 📊 Dataset

| Attribute | Value |
|---|---|
| Total Records | 96,264 hourly observations |
| Time Period | January 2015 – December 2025 (11 years) |
| Total Features | 46 columns |
| Missing Values | Zero — fully clean |
| Avg. Demand | 15,770 MW (range: 12,814 – 18,737 MW) |
| Avg. Gas Dispatch | 750 MW (peaks at 5,189 MW) |
| Avg. Renewable Share | 34.0% of total generation |
| Avg. Carbon Intensity | 42.9 g CO₂/kWh |
| Total CO₂ (study period) | 67.24 million tonnes |

**Data Sources:**

| Source | Variables |
|---|---|
| [Ontario IESO](https://ieso.ca/power-data) | Hourly demand, generation by fuel type (nuclear, hydro, solar, wind, gas, biofuel) |
| [NASA POWER](https://power.larc.nasa.gov) | Temperature, wind speed (10m/50m), solar irradiance, humidity |
| Derived Features | CO₂ tonnes per source, CAD cost per source, carbon intensity, demand lag features |
| [Our World in Data](https://ourworldindata.org/energy) | Long-term Canada energy trends for benchmarking |

---

## 🔬 Research Questions

- Can a custom stacking ensemble model accurately forecast hourly electricity demand and renewable generation for Ontario's grid?
- Which combination of base models (LSTM, XGBoost, ARIMA) produces the most accurate ensemble predictions?
- How can multi-objective linear programming optimize Ontario's hourly energy dispatch to minimize cost and carbon emissions simultaneously?
- What is the measurable reduction in CO₂ emissions and operational cost achievable through optimized versus historical dispatch?

---

## 📐 Hypotheses

**H1 — Ensemble Superiority**
- H0: The custom stacking ensemble will not achieve lower RMSE than individual base models
- H1: The custom stacking ensemble will achieve significantly lower RMSE than any individual base model for hourly demand forecasting

**H2 — Carbon & Cost Reduction**
- H0: The LP optimizer will not produce dispatch schedules with lower carbon emissions than historical dispatch
- H1: The LP optimizer will reduce carbon emissions by at least 10% vs. historical dispatch without increasing total cost

**H3 — Weather as a Predictor**
- H0: Solar irradiance and wind speed have no significant effect on renewable generation forecasting accuracy
- H1: Solar irradiance and wind speed are significant predictors that measurably improve solar and wind forecasting accuracy

---

## 🗂️ Repository Structure

```

├── README.md
└── requirements.txt
```

---
