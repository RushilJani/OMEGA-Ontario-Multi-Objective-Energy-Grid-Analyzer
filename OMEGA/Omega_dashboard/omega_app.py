"""
OMEGA — Ontario Multi-Objective Energy Grid Analyzer
Streamlit Dashboard — Real-Time Forecast + MILP Optimization
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pulp
import joblib
import os
from pathlib import Path
from tensorflow.keras.models import load_model as keras_load_model

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="OMEGA — Ontario Energy Grid",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

    .main { background-color: #0a0e1a; }
    .stApp { background-color: #0a0e1a; }

    h1, h2, h3 { font-family: 'Space Mono', monospace; color: #e2e8f0; }
    p, li, label { font-family: 'Inter', sans-serif; color: #94a3b8; }

    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 28px;
        min-height: 150px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 42px;
        font-weight: 800;
        color: #38bdf8;
        margin: 0;
    }
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 4px 0 0 0;
    }
    .omega-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #1e40af;
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
    }
    .tag {
        display: inline-block;
        background: #1e3a5f;
        color: #38bdf8;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 11px;
        font-family: 'Space Mono', monospace;
        margin: 2px;
    }
    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 13px;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 8px;
        margin: 20px 0 12px 0;
    }
    .stSelectbox label, .stDateInput label, .stSlider label {
        font-family: 'Inter', sans-serif;
        color: #94a3b8 !important;
        font-size: 13px;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Space Mono', monospace;
        color: #38bdf8;
    }
    .stAlert { border-radius: 8px; }
    .sidebar .stSelectbox { margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
FUELS = {
    'nuclear': {'cost': 29,  'emit': 12,  'cap': 9600,  'ramp': 200,  'color': '#1d4ed8'},
    'hydro':   {'cost': 45,  'emit': 24,  'cap': 8500,  'ramp': 3000, 'color': '#0891b2'},
    'gas':     {'cost': 85,  'emit': 490, 'cap': 12000, 'ramp': 3500, 'color': '#ea580c'},
    'wind':    {'cost': 35,  'emit': 0,   'cap': 5300,  'ramp': 5500, 'color': '#16a34a'},
    'solar':   {'cost': 45,  'emit': 0,   'cap': 2600,  'ramp': 2800, 'color': '#eab308'},
    'biofuel': {'cost': 105, 'emit': 230, 'cap': 495,   'ramp': 150,  'color': '#7c3aed'},
}
TRUE_RENEWABLES  = ['hydro', 'wind', 'solar', 'biofuel']
NUCLEAR_MIN_FRAC = 0.85
PEAK_HOURS       = list(range(9, 22))
GAS_MIN_STABLE_FRAC = 0.20

SEASON_MAP = {
    1:'winter', 2:'winter', 12:'winter',
    3:'spring', 4:'spring',  5:'spring',
    6:'summer', 7:'summer',  8:'summer',
    9:'fall',  10:'fall',   11:'fall',
}
SEASONAL_PARAMS = {
    'winter': {'hydro_frac':0.20,'co2_limit_kt':65.0,'renew_floor':0.10,'gas_peak_buffer':2000},
    'spring': {'hydro_frac':0.35,'co2_limit_kt':50.0,'renew_floor':0.15,'gas_peak_buffer':800},
    'summer': {'hydro_frac':0.25,'co2_limit_kt':65.0,'renew_floor':0.20,'gas_peak_buffer':1000},
    'fall':   {'hydro_frac':0.28,'co2_limit_kt':58.0,'renew_floor':0.12,'gas_peak_buffer':1000},
}
CARBON_PRICES = [
    ('No carbon pricing',        0.000),
    ('Light industrial ($50/t)', 0.050),
    ('Ontario EPS ($72/t)',      0.072),
    ('Federal OBPS ($80/t)',     0.080),
    ('Federal 2030 ($170/t)',    0.170),
    ('Net-zero ($250/t)',        0.250),
]
MODELS_DIR = 'omega_models'

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS (cached so they only load once)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading OMEGA models...")
def load_models():
    if not os.path.exists(MODELS_DIR):
        return None
    try:
        models = {
            'demand_gbm': joblib.load(f'{MODELS_DIR}/demand_gbm.pkl'),
            'solar_gbm':  joblib.load(f'{MODELS_DIR}/solar_gbm.pkl'),
            'wind_gbm':   joblib.load(f'{MODELS_DIR}/wind_gbm.pkl'),
            'demand_rf':  joblib.load(f'{MODELS_DIR}/demand_rf.pkl'),
            'solar_rf':   joblib.load(f'{MODELS_DIR}/solar_rf.pkl'),
            'wind_rf':    joblib.load(f'{MODELS_DIR}/wind_rf.pkl'),
            'ridge_d':    joblib.load(f'{MODELS_DIR}/ridge_d.pkl'),
            'ridge_s':    joblib.load(f'{MODELS_DIR}/ridge_s.pkl'),
            'ridge_w':    joblib.load(f'{MODELS_DIR}/ridge_w.pkl'),
            'demand_lstm':keras_load_model(f'{MODELS_DIR}/demand_lstm.keras'),
            'solar_lstm': keras_load_model(f'{MODELS_DIR}/solar_lstm.keras'),
            'wind_lstm':  keras_load_model(f'{MODELS_DIR}/wind_lstm.keras'),
            'scaler_d_X': joblib.load(f'{MODELS_DIR}/scaler_d_X.pkl'),
            'scaler_d_y': joblib.load(f'{MODELS_DIR}/scaler_d_y.pkl'),
            'scaler_s_X': joblib.load(f'{MODELS_DIR}/scaler_s_X.pkl'),
            'scaler_s_y': joblib.load(f'{MODELS_DIR}/scaler_s_y.pkl'),
            'scaler_w_X': joblib.load(f'{MODELS_DIR}/scaler_w_X.pkl'),
            'scaler_w_y': joblib.load(f'{MODELS_DIR}/scaler_w_y.pkl'),
            'config':     joblib.load(f'{MODELS_DIR}/feature_config.pkl'),
            'df':         pd.read_csv('merged_ontario_energy_dataset.csv'),
        }
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv('merged_ontario_energy_dataset.csv')

# ══════════════════════════════════════════════════════════════════════════════
# FORECAST FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def build_proxy_row(df_proxy, hour, month, day_of_week, season, is_weekend, config):
    """Build one proxy row using 3-layer analog system."""
    DEMAND_FEATURES = config['DEMAND_FEATURES']
    SOLAR_FEATURES  = config['SOLAR_FEATURES']
    WIND_FEATURES   = config['WIND_FEATURES']
    all_features    = list(set(DEMAND_FEATURES + SOLAR_FEATURES + WIND_FEATURES))

    df_h = df_proxy[df_proxy['hour'] == hour].copy() if 'hour' in df_proxy.columns else df_proxy.copy()

    # Layer 1 — same weekday + month + hour (50%)
    mask1 = (
        (df_h['month'] == month) &
        (df_h['day_of_week'] == day_of_week)
    ) if 'month' in df_h.columns and 'day_of_week' in df_h.columns else pd.Series([False]*len(df_h), index=df_h.index)

    # Layer 2 — same month + hour (30%)
    mask2 = (df_h['month'] == month) if 'month' in df_h.columns else pd.Series([False]*len(df_h), index=df_h.index)

    # Layer 3 — same season (20%)
    if 'season' in df_h.columns:
        mask3 = df_h['season'] == season
    elif 'season_num' in df_h.columns:
        season_num_map = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
        mask3 = df_h['season_num'] == season_num_map.get(season, 2)
    else:
        mask3 = pd.Series([True]*len(df_h), index=df_h.index)

    # Only keep features that actually exist in the dataframe
    available_features = [f for f in all_features if f in df_h.columns]

    if mask1.sum() >= 3:
        row = df_h[mask1][available_features].mean()
    elif mask2.sum() >= 3:
        row = df_h[mask2][available_features].mean()
    elif mask3.sum() >= 3:
        row = df_h[mask3][available_features].mean()
    else:
        row = df_h[available_features].mean()

    # Add date features if missing
    if 'month' not in row.index:        row['month']       = month
    if 'day_of_week' not in row.index:  row['day_of_week'] = day_of_week
    if 'is_weekend' not in row.index:   row['is_weekend']  = is_weekend
    if 'hour' not in row.index:         row['hour']        = hour

    # Add season_num if it's in features
    season_num_map = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
    if 'season_num' not in row.index:   row['season_num']  = season_num_map.get(season, 2)

    return row

def run_forecast(date_str, models):
    """Run the full GBM + RF + LSTM → Ridge forecast."""
    config = models['config']
    DEMAND_FEATURES = config['DEMAND_FEATURES']
    SOLAR_FEATURES  = config['SOLAR_FEATURES']
    WIND_FEATURES   = config['WIND_FEATURES']

    date        = pd.Timestamp(date_str)
    month       = date.month
    day_of_week = date.dayofweek
    season      = SEASON_MAP[month]
    is_weekend  = 1 if day_of_week >= 5 else 0

    df = models['df'].copy()
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        cutoff = df['datetime'].max() - pd.DateOffset(years=5)
        df_proxy = df[df['datetime'] >= cutoff].copy()
    else:
        df_proxy = df.copy()

    # Add time features if missing
    if 'hour' not in df_proxy.columns and 'datetime' in df_proxy.columns:
        df_proxy['hour'] = df_proxy['datetime'].dt.hour
    if 'month' not in df_proxy.columns and 'datetime' in df_proxy.columns:
        df_proxy['month'] = df_proxy['datetime'].dt.month
    if 'day_of_week' not in df_proxy.columns and 'datetime' in df_proxy.columns:
        df_proxy['day_of_week'] = df_proxy['datetime'].dt.dayofweek
    if 'season' not in df_proxy.columns:
        df_proxy['season'] = df_proxy['month'].map(SEASON_MAP) if 'month' in df_proxy.columns else season
    if 'is_weekend' not in df_proxy.columns and 'day_of_week' in df_proxy.columns:
        df_proxy['is_weekend'] = (df_proxy['day_of_week'] >= 5).astype(int)

    # Build proxy rows
    rows = [build_proxy_row(df_proxy, h, month, day_of_week, season, is_weekend, config) for h in range(24)]
    proxy_df = pd.DataFrame(rows)
    proxy_df['hour'] = range(24)

    # ── DEMAND ────────────────────────────────────────────────────────
    d_gbm = models['demand_gbm'].predict(proxy_df[DEMAND_FEATURES])
    d_rf  = models['demand_rf'].predict(proxy_df[DEMAND_FEATURES])
    d_seq = models['scaler_d_X'].transform(proxy_df[DEMAND_FEATURES].values)
    d_seq_3d = d_seq[np.newaxis, :, :]
    d_lstm_sc  = float(models['demand_lstm'].predict(d_seq_3d, verbose=0).flatten()[0])
    d_lstm_val = float(models['scaler_d_y'].inverse_transform([[d_lstm_sc]])[0][0])
    d_lstm = np.repeat(d_lstm_val, 24)
    demand_preds = models['ridge_d'].predict(np.column_stack([d_gbm, d_rf, d_lstm]))

    # ── SOLAR ─────────────────────────────────────────────────────────
    s_gbm = models['solar_gbm'].predict(proxy_df[SOLAR_FEATURES])
    s_rf  = models['solar_rf'].predict(proxy_df[SOLAR_FEATURES])
    s_seq = models['scaler_s_X'].transform(proxy_df[SOLAR_FEATURES].values)
    s_seq_3d = s_seq[np.newaxis, :, :]
    s_lstm_sc  = float(models['solar_lstm'].predict(s_seq_3d, verbose=0).flatten()[0])
    s_lstm_val = float(models['scaler_s_y'].inverse_transform([[s_lstm_sc]])[0][0])
    s_lstm = np.repeat(s_lstm_val, 24)
    solar_preds = np.clip(models['ridge_s'].predict(np.column_stack([s_gbm, s_rf, s_lstm])), 0, None)

    # ── WIND ──────────────────────────────────────────────────────────
    w_gbm = models['wind_gbm'].predict(proxy_df[WIND_FEATURES])
    w_rf  = models['wind_rf'].predict(proxy_df[WIND_FEATURES])
    w_seq = models['scaler_w_X'].transform(proxy_df[WIND_FEATURES].values)
    w_seq_3d = w_seq[np.newaxis, :, :]
    w_lstm_sc  = float(models['wind_lstm'].predict(w_seq_3d, verbose=0).flatten()[0])
    w_lstm_val = float(models['scaler_w_y'].inverse_transform([[w_lstm_sc]])[0][0])
    w_lstm = np.repeat(w_lstm_val, 24)
    wind_preds = np.clip(models['ridge_w'].predict(np.column_stack([w_gbm, w_rf, w_lstm])), 0, None)

    return {
        'demand': [round(float(v), 1) for v in demand_preds],
        'solar':  [round(float(v), 1) for v in solar_preds],
        'wind':   [round(float(v), 1) for v in wind_preds],
        'season': season,
        'date':   date_str,
    }

# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def run_optimizer(forecast_result, selected_rho_name):
    """Run MILP optimizer for selected carbon price."""
    demand  = np.array(forecast_result['demand'])
    solar   = np.array(forecast_result['solar'])
    wind    = np.array(forecast_result['wind'])
    season  = forecast_result['season']

    sp = SEASONAL_PARAMS[season]
    daily_demand = demand.sum()

    HYDRO_DAILY_MWH   = min(daily_demand * sp['hydro_frac'], FUELS['hydro']['cap'] * 24 * 0.55)
    DAILY_CO2_LIMIT_KG = sp['co2_limit_kt'] * 1_000_000
    RENEWABLE_FLOOR_FRAC = sp['renew_floor']

    peak_demand_avg = demand[PEAK_HOURS].mean()
    clean_peak_avg  = (
        NUCLEAR_MIN_FRAC * FUELS['nuclear']['cap']
        + solar[PEAK_HOURS].mean()
        + wind[PEAK_HOURS].mean()
    )
    GAS_PEAK_LIMIT_MW = max(2000, min(
        peak_demand_avg - clean_peak_avg + sp['gas_peak_buffer'],
        FUELS['gas']['cap']
    ))

    # Get rho
    rho_map = {name: rho for name, rho in CARBON_PRICES}
    rho = rho_map.get(selected_rho_name, 0.0)

    # Run all 6 scenarios
    all_results = []
    for name, rho_val in CARBON_PRICES:
        r = solve_one(rho_val, demand, solar, wind,
                      HYDRO_DAILY_MWH, DAILY_CO2_LIMIT_KG,
                      GAS_PEAK_LIMIT_MW, RENEWABLE_FLOOR_FRAC)
        r['name'] = name
        r['rho']  = rho_val * 1000
        all_results.append(r)

    return all_results, {
        'HYDRO_DAILY_MWH': HYDRO_DAILY_MWH,
        'DAILY_CO2_LIMIT_KG': DAILY_CO2_LIMIT_KG,
        'RENEWABLE_FLOOR_FRAC': RENEWABLE_FLOOR_FRAC,
        'GAS_PEAK_LIMIT_MW': GAS_PEAK_LIMIT_MW,
        'season': season,
    }

def solve_one(rho, demand, solar_cap, wind_cap,
              hydro_budget, co2_ceiling, gas_peak, ren_floor):
    """Solve single MILP scenario with fallback."""
    T     = len(demand)
    fuels = list(FUELS.keys())
    prob  = pulp.LpProblem('OMEGA_MILP', pulp.LpMinimize)

    P = {f: [pulp.LpVariable(f'P_{f}_{t}', lowBound=0, upBound=FUELS[f]['cap'])
             for t in range(T)] for f in fuels}
    u_gas = [pulp.LpVariable(f'u_gas_{t}', cat='Binary') for t in range(T)]

    prob += pulp.lpSum((FUELS[f]['cost'] + rho * FUELS[f]['emit']) * P[f][t]
                       for f in fuels for t in range(T))

    for t in range(T):
        prob += pulp.lpSum(P[f][t] for f in fuels) == demand[t], f'C1_balance_t{t}'
        prob += P['solar'][t] <= solar_cap[t], f'C2_solar_t{t}'
        prob += P['wind'][t]  <= wind_cap[t],  f'C2_wind_t{t}'
        nuc_min = NUCLEAR_MIN_FRAC * FUELS['nuclear']['cap']
        prob += P['nuclear'][t] >= nuc_min, f'C4_nuclear_min_t{t}'
        prob += (pulp.lpSum(P[f][t] for f in TRUE_RENEWABLES) >= ren_floor * demand[t],
                 f'C8_renew_floor_t{t}')
        gas_min_mw = GAS_MIN_STABLE_FRAC * FUELS['gas']['cap']
        prob += P['gas'][t] <= FUELS['gas']['cap'] * u_gas[t], f'C10_gas_ub_t{t}'
        prob += P['gas'][t] >= gas_min_mw * u_gas[t], f'C10_gas_lb_t{t}'

    for f in fuels:
        for t in range(1, T):
            prob += P[f][t] - P[f][t-1] <=  FUELS[f]['ramp'], f'C5_ramp_up_{f}_t{t}'
            prob += P[f][t] - P[f][t-1] >= -FUELS[f]['ramp'], f'C5_ramp_dn_{f}_t{t}'

    prob += pulp.lpSum(P['hydro'][t] for t in range(T)) <= hydro_budget, 'C6_hydro_budget'
    prob += pulp.lpSum(FUELS[f]['emit'] * P[f][t] for f in fuels for t in range(T)) <= co2_ceiling, 'C7_co2_ceiling'

    for t in PEAK_HOURS:
        prob += P['gas'][t] <= gas_peak, f'C9_gas_peak_t{t}'

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]

    # Fallback if infeasible
    if status != 'Optimal':
        prob2 = pulp.LpProblem('OMEGA_MILP_relaxed', pulp.LpMinimize)
        P2    = {f: [pulp.LpVariable(f'P2_{f}_{t}', lowBound=0, upBound=FUELS[f]['cap'])
                     for t in range(T)] for f in fuels}
        u2    = [pulp.LpVariable(f'u2_gas_{t}', cat='Binary') for t in range(T)]
        prob2 += pulp.lpSum((FUELS[f]['cost'] + rho * FUELS[f]['emit']) * P2[f][t]
                            for f in fuels for t in range(T))
        for t in range(T):
            prob2 += pulp.lpSum(P2[f][t] for f in fuels) == demand[t]
            prob2 += P2['solar'][t] <= solar_cap[t]
            prob2 += P2['wind'][t]  <= wind_cap[t]
            prob2 += P2['nuclear'][t] >= NUCLEAR_MIN_FRAC * FUELS['nuclear']['cap']
            gas_min_mw = GAS_MIN_STABLE_FRAC * FUELS['gas']['cap']
            prob2 += P2['gas'][t] <= FUELS['gas']['cap'] * u2[t]
            prob2 += P2['gas'][t] >= gas_min_mw * u2[t]
        for f in fuels:
            for t in range(1, T):
                prob2 += P2[f][t] - P2[f][t-1] <=  FUELS[f]['ramp']
                prob2 += P2[f][t] - P2[f][t-1] >= -FUELS[f]['ramp']
        prob2 += pulp.lpSum(P2['hydro'][t] for t in range(T)) <= hydro_budget * 1.5
        prob2 += pulp.lpSum(FUELS[f]['emit'] * P2[f][t] for f in fuels for t in range(T)) <= co2_ceiling * 1.5
        prob2.solve(pulp.PULP_CBC_CMD(msg=0))
        P, u_gas = P2, u2
        status = 'Relaxed'

    dispatch = {f: [pulp.value(P[f][t]) or 0.0 for t in range(T)] for f in fuels}
    gas_commitment = [int(round(pulp.value(u_gas[t]) or 0)) for t in range(T)]
    total_cost = sum(FUELS[f]['cost'] * sum(dispatch[f]) for f in fuels)
    total_emit = sum(FUELS[f]['emit'] * sum(dispatch[f]) for f in fuels) / 1000
    ren_vals   = sum(sum(dispatch[f]) for f in TRUE_RENEWABLES)
    total_gen  = sum(sum(dispatch[f]) for f in fuels)
    ren_pct    = ren_vals / total_gen * 100 if total_gen > 0 else 0

    return {
        'status':     status,
        'dispatch':   dispatch,
        'commitment': gas_commitment,
        'cost':       total_cost,
        'emissions':  total_emit,
        'renewable':  ren_pct,
        'gas_hours':  sum(gas_commitment),
    }

# ══════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def forecast_chart(forecast_result, highlight_hour=None):
    hours  = list(range(24))
    demand = forecast_result['demand']
    solar  = forecast_result['solar']
    wind   = forecast_result['wind']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=demand, name='Demand',
        line=dict(color='#38bdf8', width=2.5),
        fill='tozeroy', fillcolor='rgba(56,189,248,0.08)'))
    fig.add_trace(go.Scatter(x=hours, y=solar, name='Solar',
        line=dict(color='#eab308', width=2),
        fill='tozeroy', fillcolor='rgba(234,179,8,0.08)'))
    fig.add_trace(go.Scatter(x=hours, y=wind, name='Wind',
        line=dict(color='#22c55e', width=2),
        fill='tozeroy', fillcolor='rgba(34,197,94,0.08)'))

    if highlight_hour is not None:
        fig.add_vline(
            x=highlight_hour, line_width=2,
            line_dash="dash", line_color="#f8fafc",
            annotation_text=f"{highlight_hour:02d}:00",
            annotation_font_color="#f8fafc",
            annotation_position="top right",
        )

    fig.update_layout(
        title=dict(text=f"24-Hour Forecast — {forecast_result['date']} ({forecast_result['season'].capitalize()})",
                   font=dict(color='#e2e8f0', family='Space Mono', size=14)),
        xaxis=dict(title='Hour', gridcolor='#1e293b', color='#64748b',
                   tickvals=list(range(0,24,2)),
                   ticktext=[f'{h:02d}:00' for h in range(0,24,2)]),
        yaxis=dict(title='MW', gridcolor='#1e293b', color='#64748b'),
        plot_bgcolor='#0f172a', paper_bgcolor='#0f172a',
        legend=dict(bgcolor='#1e293b', font=dict(color='#94a3b8')),
        font=dict(color='#94a3b8'),
        height=350,
    )
    return fig

def dispatch_chart(result, date_str, highlight_hour=None):
    hours = list(range(24))
    fig   = go.Figure()

    for fuel, data in result['dispatch'].items():
        fig.add_trace(go.Bar(
            x=hours, y=data,
            name=fuel.capitalize(),
            marker_color=FUELS[fuel]['color'],
        ))

    # Add vertical line for selected hour
    if highlight_hour is not None:
        fig.add_vline(
            x=highlight_hour, line_width=2,
            line_dash="dash", line_color="#f8fafc",
            annotation_text=f"{highlight_hour:02d}:00",
            annotation_font_color="#f8fafc",
            annotation_position="top",
        )

    fig.update_layout(
        barmode='stack',
        title=dict(text=f"Hourly Dispatch — {result['name']}  (ρ=${result['rho']:.0f}/t)",
                   font=dict(color='#e2e8f0', family='Space Mono', size=13)),
        xaxis=dict(title='Hour', gridcolor='#1e293b', color='#64748b',
                   tickvals=list(range(0,24,2)),
                   ticktext=[f'{h:02d}:00' for h in range(0,24,2)]),
        yaxis=dict(title='MW', gridcolor='#1e293b', color='#64748b'),
        plot_bgcolor='#0f172a', paper_bgcolor='#0f172a',
        legend=dict(bgcolor='#1e293b', font=dict(color='#94a3b8')),
        font=dict(color='#94a3b8'),
        height=380,
    )
    return fig

def pareto_chart(all_results):
    names  = [r['name'] for r in all_results]
    costs  = [r['cost']/1e6 for r in all_results]
    emits  = [r['emissions'] for r in all_results]
    rhos   = [r['rho'] for r in all_results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=emits, y=costs,
        mode='lines+markers+text',
        text=[f"${r:.0f}/t" for r in rhos],
        textposition='top right',
        textfont=dict(color='#94a3b8', size=10),
        marker=dict(size=12, color=rhos, colorscale='Blues',
                    showscale=True,
                    colorbar=dict(title=dict(text='ρ ($/t)'), tickfont=dict(color='#94a3b8'))),
        line=dict(color='#38bdf8', width=2),
        name='Pareto frontier',
    ))

    fig.update_layout(
        title=dict(text='Cost–Emissions Pareto Frontier',
                   font=dict(color='#e2e8f0', family='Space Mono', size=14)),
        xaxis=dict(title='Daily CO₂ Emissions (tonnes)', gridcolor='#1e293b', color='#64748b'),
        yaxis=dict(title='Daily Cost ($M CAD)', gridcolor='#1e293b', color='#64748b'),
        plot_bgcolor='#0f172a', paper_bgcolor='#0f172a',
        font=dict(color='#94a3b8'),
        height=380,
    )
    return fig

def scenario_bar_chart(all_results):
    names  = [r['name'].replace(' pricing','').replace(' industrial','') for r in all_results]
    costs  = [r['cost']/1e6 for r in all_results]
    emits  = [r['emissions'] for r in all_results]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Cost ($M)', x=names, y=costs,
                         marker_color='#38bdf8', opacity=0.85))
    fig.add_trace(go.Bar(name='Emissions (kt)', x=names,
                         y=[e/1000 for e in emits],
                         marker_color='#ea580c', opacity=0.85))

    fig.update_layout(
        barmode='group',
        title=dict(text='Cost vs Emissions by Scenario',
                   font=dict(color='#e2e8f0', family='Space Mono', size=14)),
        xaxis=dict(gridcolor='#1e293b', color='#64748b', tickangle=-20),
        yaxis=dict(gridcolor='#1e293b', color='#64748b'),
        plot_bgcolor='#0f172a', paper_bgcolor='#0f172a',
        legend=dict(bgcolor='#1e293b', font=dict(color='#94a3b8')),
        font=dict(color='#94a3b8'),
        height=350,
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="omega-header">
        <h1 style="margin:0;font-size:28px;letter-spacing:0.05em;">⚡ OMEGA</h1>
        <p style="margin:4px 0 8px 0;font-size:14px;color:#38bdf8;font-family:'Space Mono',monospace;">
            Ontario Multi-Objective Energy Grid Analyzer
        </p>
        <span class="tag">GBM + RF + LSTM → Ridge</span>
        <span class="tag">MILP Optimizer</span>
        <span class="tag">6 Carbon Scenarios</span>
        <span class="tag">Ontario IESO 2015–2025</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar Controls ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Controls")

        date_input = st.date_input(
            "Select Date",
            value=pd.Timestamp('2026-06-10'),
            min_value=pd.Timestamp('2026-01-01'),
            max_value=pd.Timestamp('2026-12-31'),
        )
        date_str = str(date_input)

        carbon_scenario = st.selectbox(
            "Carbon Price Scenario",
            [name for name, _ in CARBON_PRICES],
            index=3,  # Federal OBPS default
        )

        selected_hour = st.slider(
            "Focus Hour (for hourly detail)",
            min_value=0, max_value=23, value=18,
            format="%d:00",
            help="Select a specific hour to see detailed dispatch for that hour"
        )

        show_all_hours = st.checkbox("Show full 24-hour view", value=True)

        run_btn = st.button("⚡ Run Analysis", use_container_width=True, type="primary")

        st.markdown("---")
        st.markdown("### 📋 About")
        st.markdown("""
        <p style="font-size:12px;color:#64748b;">
        DAMO 699 Capstone<br>
        University of Niagara Falls<br>
        Supervisor: Prof. Hany Osman<br><br>
        Team: Akash · Nidhi · Aena<br>
        Rushil · Shraddha
        </p>
        """, unsafe_allow_html=True)

    # ── Load Models ───────────────────────────────────────────────────────────
    models = load_models()

    if models is None:
        st.error("⚠️ Models not found. Make sure the `omega_models/` folder is in the same directory as this app.")
        st.info("""
        **To set up:**
        1. Run your forecast notebook (Forcast_FINAL.ipynb)
        2. Run Step 12b to save all models
        3. Download the `omega_models/` folder
        4. Place it in the same folder as `omega_app.py`
        5. Also place `merged_ontario_energy_dataset.csv` here
        """)
        return

    # ── Run on button click ───────────────────────────────────────────────────
    if run_btn:
        with st.spinner(f"🔮 Forecasting {date_str}..."):
            try:
                forecast_result = run_forecast(date_str, models)
                st.session_state['forecast'] = forecast_result
                st.session_state['date_str'] = date_str
            except Exception as e:
                st.error(f"Forecast error: {e}")
                return

        with st.spinner("⚙️ Running MILP optimization..."):
            try:
                all_results, constraints = run_optimizer(forecast_result, carbon_scenario)
                st.session_state['all_results']  = all_results
                st.session_state['constraints']  = constraints
                st.session_state['selected_rho'] = carbon_scenario
            except Exception as e:
                st.error(f"Optimizer error: {e}")
                return

        st.success(f"✅ Analysis complete for {date_str}")

    # ── Display Results ───────────────────────────────────────────────────────
    if 'forecast' not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:60px;color:#334155;">
            <h2 style="color:#334155;">Select a date and click Run Analysis</h2>
            <p>The app will forecast demand, solar & wind, then optimize dispatch across 6 carbon price scenarios.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    forecast_result = st.session_state['forecast']
    all_results     = st.session_state['all_results']
    constraints     = st.session_state['constraints']
    selected_rho    = st.session_state['selected_rho']

    # Find selected scenario result
    selected_result = next((r for r in all_results if r['name'] == selected_rho), all_results[3])

    season = forecast_result['season']
    demand = forecast_result['demand']

    # ── Top KPI Cards ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Key Metrics — ' + selected_rho + '</p>', unsafe_allow_html=True)
    
    st.title("⚡ OMEGA — Ontario Multi-Objective Energy Grid Analyzer")
    st.info("""
    Forecasting Engine: GBM + RF + LSTM base learners, Ridge meta-learner
    
    Optimization Engine: Mixed Integer Linear Programming (MILP)
    
    Objective: Minimize Cost and CO₂ Emissions for Ontario's Grid
    """)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{sum(demand)/1000:.0f} GWh</p>
            <p class="metric-label">Daily Demand</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">${selected_result['cost']/1e6:.2f}M</p>
            <p class="metric-label">Daily Cost</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{selected_result['emissions']:,.0f}t</p>
            <p class="metric-label">CO₂ Emissions</p>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{selected_result['renewable']:.1f}%</p>
            <p class="metric-label">Renewable Share</p>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{selected_result['gas_hours']}h</p>
            <p class="metric-label">Gas Committed</p>
        </div>""", unsafe_allow_html=True)
    
    st.info("""
    Key Finding:
    An $80/t carbon price produced the best trade-off between cost and emissions.
    Gas generation decreased while renewable penetration increased,
    resulting in significant CO₂ reductions.
    """)
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── Hour Detail Cards ─────────────────────────────────────────────────
    hour_label = f"{selected_hour:02d}:00"
    is_peak    = "⚡ PEAK HOUR" if selected_hour in PEAK_HOURS else "🌙 OFF-PEAK"

    st.markdown(f'<p class="section-title">Hour Detail — {hour_label}  {is_peak}</p>',
                unsafe_allow_html=True)

    d_hr = forecast_result["demand"][selected_hour]
    s_hr = forecast_result["solar"][selected_hour]
    w_hr = forecast_result["wind"][selected_hour]

    # Dispatch for selected hour
    hr_dispatch = {f: selected_result["dispatch"][f][selected_hour]
                   for f in FUELS.keys()}
    hr_gas_on   = selected_result["commitment"][selected_hour]
    hr_emit     = sum(FUELS[f]["emit"] * hr_dispatch[f] for f in FUELS) / 1000
    hr_cost     = sum(FUELS[f]["cost"] * hr_dispatch[f] for f in FUELS)
    hr_ren      = sum(hr_dispatch[f] for f in TRUE_RENEWABLES)
    hr_total    = sum(hr_dispatch[f] for f in FUELS)
    hr_ren_pct  = hr_ren / hr_total * 100 if hr_total > 0 else 0

    h1, h2, h3, h4, h5, h6 = st.columns(6)
    with h1:
        st.markdown(f'''<div class="metric-card">
            <p class="metric-value">{d_hr:,.0f}</p>
            <p class="metric-label">Demand MW</p></div>''', unsafe_allow_html=True)
    with h2:
        st.markdown(f'''<div class="metric-card">
            <p class="metric-value" style="color:#eab308">{s_hr:,.0f}</p>
            <p class="metric-label">Solar MW</p></div>''', unsafe_allow_html=True)
    with h3:
        st.markdown(f'''<div class="metric-card">
            <p class="metric-value" style="color:#22c55e">{w_hr:,.0f}</p>
            <p class="metric-label">Wind MW</p></div>''', unsafe_allow_html=True)
    with h4:
        st.markdown(f'''<div class="metric-card">
            <p class="metric-value">${hr_cost:,.0f}</p>
            <p class="metric-label">Hour Cost CAD</p></div>''', unsafe_allow_html=True)
    with h5:
        st.markdown(f'''<div class="metric-card">
            <p class="metric-value" style="color:#f87171">{hr_emit:,.1f}t</p>
            <p class="metric-label">Hour CO₂</p></div>''', unsafe_allow_html=True)
    with h6:
        gas_color = "#ea580c" if hr_gas_on else "#22c55e"
        gas_label = "COMMITTED" if hr_gas_on else "OFF"
        st.markdown(f'''<div class="metric-card">
            <p class="metric-value" style="color:{gas_color}">{gas_label}</p>
            <p class="metric-label">Gas Status</p></div>''', unsafe_allow_html=True)

    # Hour dispatch breakdown
    with st.expander(f"🔍 Full dispatch breakdown at {hour_label}", expanded=True):
        hd1, hd2 = st.columns(2)
        with hd1:
            st.markdown("**Generation by fuel:**")
            for fuel in FUELS:
                mw    = hr_dispatch[fuel]
                pct   = mw / hr_total * 100 if hr_total > 0 else 0
                color = FUELS[fuel]["color"]
                bar   = "█" * int(pct / 5)
                st.markdown(
                    f'''<p style="font-family:'Space Mono',monospace;font-size:12px;color:#94a3b8;margin:2px 0;">
                    <span style="color:{color}">■</span>
                    {fuel.capitalize():<8} {mw:>7,.0f} MW  {pct:>5.1f}%  {bar}
                    </p>''', unsafe_allow_html=True)
        with hd2:
            st.markdown("**Renewable mix:**")
            fig_hr = go.Figure(go.Pie(
                labels=[f.capitalize() for f in FUELS],
                values=[hr_dispatch[f] for f in FUELS],
                marker_colors=[FUELS[f]["color"] for f in FUELS],
                hole=0.5,
                textfont=dict(color="white", size=11),
            ))
            fig_hr.update_layout(
                height=220, margin=dict(t=10,b=10,l=10,r=10),
                paper_bgcolor="#0f172a",
                legend=dict(font=dict(color="#94a3b8", size=10),
                           bgcolor="#0f172a"),
                showlegend=True,
            )
            st.plotly_chart(fig_hr, use_container_width=True)

    st.markdown("---")

    # ── Forecast + Dispatch Charts ────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-title">Forecast Inputs</p>', unsafe_allow_html=True)
        st.plotly_chart(forecast_chart(forecast_result, highlight_hour=selected_hour), use_container_width=True)

    with col2:
        st.markdown('<p class="section-title">Optimal Dispatch</p>', unsafe_allow_html=True)
        st.plotly_chart(dispatch_chart(selected_result, date_str, highlight_hour=selected_hour), use_container_width=True)

    # ── Scenario Comparison + Pareto ──────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<p class="section-title">Scenario Comparison</p>', unsafe_allow_html=True)
        st.plotly_chart(scenario_bar_chart(all_results), use_container_width=True)

    with col4:
        st.markdown('<p class="section-title">Pareto Frontier</p>', unsafe_allow_html=True)
        st.plotly_chart(pareto_chart(all_results), use_container_width=True)

    # ── Scenario Summary Table ────────────────────────────────────────────────
    st.markdown('<p class="section-title">All Scenarios Summary</p>', unsafe_allow_html=True)

    table_data = []
    for r in all_results:
        table_data.append({
            'Scenario':      r['name'],
            'ρ ($/tonne)':   f"${r['rho']:.0f}",
            'Cost ($M)':     f"${r['cost']/1e6:.3f}",
            'Emissions (t)': f"{r['emissions']:,.0f}",
            'Renewable %':   f"{r['renewable']:.1f}%",
            'Gas Hours':     r['gas_hours'],
            'Status':        r['status'],
        })

    st.dataframe(
        pd.DataFrame(table_data),
        use_container_width=True,
        hide_index=True,
    )

    # ── Real-World Impact ─────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Real-World Impact vs $0/tonne Baseline</p>',
                unsafe_allow_html=True)

    baseline = all_results[0]
    baseline_gas = sum(baseline['dispatch']['gas'])
    baseline_cost_mwh = baseline['cost'] / sum(demand)

    impact_cols = st.columns(len(all_results))
    for col, r in zip(impact_cols, all_results):
        gas_mwh       = sum(r['dispatch']['gas'])
        displaced     = baseline_gas - gas_mwh
        co2_saved     = displaced * FUELS['gas']['emit'] / 1000
        trees         = co2_saved * 1000 / 21.7
        cost_mwh      = r['cost'] / sum(demand)
        hh_monthly    = (cost_mwh - baseline_cost_mwh) * 30 / 1000 * 30
        rho_label     = f"${r['rho']:.0f}/t"

        with col:
            st.markdown(f"""
            <div class="metric-card" style="padding:12px;">
                <p style="font-family:'Space Mono';font-size:11px;color:#38bdf8;margin:0;">{rho_label}</p>
                <p style="font-size:13px;color:#22c55e;margin:4px 0;">🌳 {trees:,.0f}</p>
                <p style="font-size:10px;color:#64748b;margin:0;">trees/yr equivalent</p>
                <p style="font-size:13px;color:#f59e0b;margin:4px 0;">+${hh_monthly:.2f}/mo</p>
                <p style="font-size:10px;color:#64748b;margin:0;">household cost</p>
            </div>""", unsafe_allow_html=True)

    # ── Constraints Info ──────────────────────────────────────────────────────
    with st.expander("⚙️ Active Constraints & Parameters"):
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f"""
            **Season:** {constraints['season'].capitalize()}
            **Hydro budget:** {constraints['HYDRO_DAILY_MWH']:,.0f} MWh/day
            **CO₂ ceiling:** {constraints['DAILY_CO2_LIMIT_KG']/1e6:.0f} kt/day
            """)
        with cc2:
            st.markdown(f"""
            **Renewable floor:** {constraints['RENEWABLE_FLOOR_FRAC']:.0%} per hour
            **Gas peak cap:** {constraints['GAS_PEAK_LIMIT_MW']:,.0f} MW
            **Gas min stable:** {GAS_MIN_STABLE_FRAC * FUELS['gas']['cap']:,.0f} MW when committed
            """)

if __name__ == '__main__':
    main()
