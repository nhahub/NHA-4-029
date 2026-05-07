import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import math
from datetime import datetime

st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════
#  CSS — Military HUD / Threat Assessment Terminal
# ═══════════════════════════════════════════════════════════
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Share+Tech+Mono&family=Exo+2:ital,wght@0,200;0,300;0,400;0,600;1,300&display=swap');

/* ── Reset & Root ── */
:root {
  --bg:       #010509;
  --bg2:      #020c10;
  --panel:    #040f14;
  --card:     #061219;
  --border:   #0d3040;
  --border2:  #0e4558;
  --glow:     #00ffe0;
  --glow2:    #00b8a0;
  --danger:   #ff2244;
  --danger2:  #ff6070;
  --warn:     #ffcc00;
  --dim:      #1a4050;
  --text:     #c8e8e0;
  --muted:    #3a7060;
  --muted2:   #1f4a40;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── Body & App ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Exo 2', sans-serif !important;
}

/* CRT scanlines overlay */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.07) 2px,
    rgba(0,0,0,0.07) 4px
  );
  pointer-events: none;
  z-index: 9998;
}

/* Vignette */
[data-testid="stAppViewContainer"]::after {
  content: '';
  position: fixed;
  inset: 0;
  background: radial-gradient(ellipse at center,
    transparent 50%,
    rgba(0,0,0,0.65) 100%
  );
  pointer-events: none;
  z-index: 9997;
}

[data-testid="stHeader"]    { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
footer { display: none !important; }
[data-testid="stSidebar"]   { background: var(--bg2) !important; }

[data-testid="stMainBlockContainer"] {
  max-width: 800px !important;
  padding: 0 1.25rem 5rem !important;
}

/* ── Typography ── */
h1, h2, h3 { font-family: 'Orbitron', monospace !important; }

/* ── Widgets override ── */
label[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] p {
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.18em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
  margin-bottom: 0.35rem !important;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
  background: var(--card) !important;
  border: 1px solid var(--border2) !important;
  color: var(--glow) !important;
  border-radius: 4px !important;
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 1rem !important;
  padding: 0.55rem 0.8rem !important;
  transition: border-color 0.2s, box-shadow 0.2s;
  caret-color: var(--glow);
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus {
  border-color: var(--glow) !important;
  box-shadow: 0 0 0 2px rgba(0,255,224,0.12),
              0 0 16px rgba(0,255,224,0.06) !important;
  outline: none !important;
}

div[data-testid="stSelectbox"] > div > div {
  background: var(--card) !important;
  border: 1px solid var(--border2) !important;
  color: var(--glow) !important;
  border-radius: 4px !important;
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 0.85rem !important;
}

/* Slider */
div[data-testid="stSlider"] > div > div > div {
  background: var(--dim) !important;
}
div[data-testid="stSlider"] [role="slider"] {
  background: var(--glow) !important;
  border-color: var(--glow) !important;
  box-shadow: 0 0 10px var(--glow), 0 0 20px rgba(0,255,224,0.3) !important;
  width: 14px !important;
  height: 14px !important;
}

/* Submit button */
div[data-testid="stFormSubmitButton"] button {
  background: transparent !important;
  border: 1px solid var(--glow) !important;
  color: var(--glow) !important;
  font-family: 'Orbitron', monospace !important;
  font-weight: 700 !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.22em !important;
  border-radius: 4px !important;
  width: 100% !important;
  padding: 1rem !important;
  margin-top: 0.5rem !important;
  text-transform: uppercase !important;
  cursor: pointer !important;
  transition: all 0.2s !important;
  box-shadow: 0 0 12px rgba(0,255,224,0.15),
              inset 0 0 12px rgba(0,255,224,0.03) !important;
  position: relative !important;
  overflow: hidden !important;
}
div[data-testid="stFormSubmitButton"] button::before {
  content: '';
  position: absolute;
  top: 0; left: -100%;
  width: 100%; height: 100%;
  background: linear-gradient(90deg,
    transparent, rgba(0,255,224,0.08), transparent);
  transition: left 0.5s;
}
div[data-testid="stFormSubmitButton"] button:hover::before { left: 100%; }
div[data-testid="stFormSubmitButton"] button:hover {
  background: rgba(0,255,224,0.06) !important;
  box-shadow: 0 0 24px rgba(0,255,224,0.3),
              inset 0 0 16px rgba(0,255,224,0.06) !important;
}

/* Expander */
details {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  padding: 0.5rem !important;
}
details summary {
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.1em !important;
  color: var(--muted) !important;
}

/* ── Custom components ── */

/* Top bar / HUD header */
.hud-topbar {
  border-bottom: 1px solid var(--border2);
  padding: 0.6rem 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.25rem;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.12em;
  color: var(--muted);
}
.hud-topbar .active { color: var(--glow); }

/* Main title block */
.hud-hero {
  padding: 2.5rem 0 1.75rem;
  text-align: center;
  position: relative;
}
.hud-pre {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.4em;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 1rem;
}
.hud-title {
  font-family: 'Orbitron', monospace;
  font-size: clamp(1.8rem, 5.5vw, 3rem);
  font-weight: 900;
  letter-spacing: 0.08em;
  color: var(--glow);
  text-shadow:
    0 0 20px rgba(0,255,224,0.6),
    0 0 50px rgba(0,255,224,0.2),
    0 0 100px rgba(0,255,224,0.08);
  line-height: 1;
  margin-bottom: 0.5rem;
}
.hud-subtitle {
  font-family: 'Exo 2', sans-serif;
  font-weight: 200;
  font-style: italic;
  font-size: 0.85rem;
  color: var(--muted);
  letter-spacing: 0.08em;
}

/* Horizontal rule with label */
.hud-rule {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin: 1.75rem 0 1.1rem;
}
.hud-rule::before, .hud-rule::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border2));
}
.hud-rule::before { background: linear-gradient(90deg, transparent, var(--border2)); }
.hud-rule::after  { background: linear-gradient(90deg, var(--border2), transparent); }
.hud-rule span {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.58rem;
  letter-spacing: 0.28em;
  color: var(--muted2);
  white-space: nowrap;
  text-transform: uppercase;
}

/* Input panel */
.input-panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1.25rem;
  position: relative;
  margin-bottom: 0.75rem;
}
.input-panel::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg,
    transparent 0%, var(--glow2) 30%, var(--glow) 50%, var(--glow2) 70%, transparent 100%);
  opacity: 0.4;
}

/* Corner brackets on panels */
.bracket {
  position: absolute;
  width: 12px; height: 12px;
  opacity: 0.5;
}
.bracket.tl { top: 6px; left: 6px;
  border-top: 1px solid var(--glow); border-left: 1px solid var(--glow); }
.bracket.tr { top: 6px; right: 6px;
  border-top: 1px solid var(--glow); border-right: 1px solid var(--glow); }
.bracket.bl { bottom: 6px; left: 6px;
  border-bottom: 1px solid var(--glow); border-left: 1px solid var(--glow); }
.bracket.br { bottom: 6px; right: 6px;
  border-bottom: 1px solid var(--glow); border-right: 1px solid var(--glow); }

/* Threat level ring / circular gauge */
.gauge-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2rem 1rem;
}
.gauge-svg { filter: drop-shadow(0 0 12px rgba(0,255,224,0.3)); }

/* Verdict panel */
.verdict-panel {
  border-radius: 6px;
  padding: 0;
  position: relative;
  overflow: hidden;
  margin: 0.5rem 0;
}
.verdict-inner {
  padding: 2.5rem 2rem;
  text-align: center;
  position: relative;
  z-index: 1;
}
.verdict-bg-safe {
  background:
    radial-gradient(ellipse at 50% 0%, rgba(0,255,224,0.08) 0%, transparent 65%),
    var(--panel);
  border: 1px solid var(--border2);
  box-shadow: 0 0 40px rgba(0,255,224,0.06),
              inset 0 1px 0 rgba(0,255,224,0.1);
}
.verdict-bg-fraud {
  background:
    radial-gradient(ellipse at 50% 0%, rgba(255,34,68,0.12) 0%, transparent 65%),
    var(--panel);
  border: 1px solid rgba(255,34,68,0.4);
  box-shadow: 0 0 50px rgba(255,34,68,0.1),
              inset 0 1px 0 rgba(255,34,68,0.2);
}
.verdict-bg-idle {
  background: var(--panel);
  border: 1px solid var(--border);
}

.verdict-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.35em;
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}
.verdict-big {
  font-family: 'Orbitron', monospace;
  font-size: clamp(1.5rem, 4vw, 2.2rem);
  font-weight: 900;
  letter-spacing: 0.05em;
  line-height: 1;
  margin-bottom: 0.75rem;
}
.verdict-pct {
  font-family: 'Orbitron', monospace;
  font-size: clamp(4rem, 14vw, 7rem);
  font-weight: 900;
  letter-spacing: -0.04em;
  line-height: 0.9;
}
.verdict-pct-sub {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  margin-top: 0.75rem;
}

/* Animated threat ring around verdict */
@keyframes rotate-ring {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
@keyframes rotate-ring-rev {
  from { transform: rotate(0deg); }
  to   { transform: rotate(-360deg); }
}
.ring-outer {
  position: absolute;
  inset: -2px;
  border-radius: 6px;
  border: 1px dashed;
  opacity: 0.15;
  pointer-events: none;
}

/* Glow pulse */
@keyframes glow-pulse-green {
  0%,100% { box-shadow: 0 0 40px rgba(0,255,224,0.06), inset 0 1px 0 rgba(0,255,224,0.1); }
  50%      { box-shadow: 0 0 70px rgba(0,255,224,0.14), inset 0 1px 0 rgba(0,255,224,0.15); }
}
@keyframes glow-pulse-red {
  0%,100% { box-shadow: 0 0 50px rgba(255,34,68,0.1), inset 0 1px 0 rgba(255,34,68,0.2); }
  50%      { box-shadow: 0 0 90px rgba(255,34,68,0.22), inset 0 1px 0 rgba(255,34,68,0.28); }
}
.pulse-safe  { animation: glow-pulse-green 3s ease-in-out infinite; }
.pulse-fraud { animation: glow-pulse-red 2s ease-in-out infinite; }

/* Metric grid */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.5rem;
  margin: 0.75rem 0;
}
.metric-cell {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.85rem 0.5rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s;
}
.metric-cell:hover { border-color: var(--border2); }
.metric-cell::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 2px;
  background: var(--glow);
  opacity: 0;
  transition: opacity 0.2s;
}
.metric-cell:hover::before { opacity: 0.3; }
.metric-val {
  font-family: 'Orbitron', monospace;
  font-size: 1.05rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  line-height: 1;
}
.metric-lbl {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.55rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
  margin-top: 0.35rem;
}

/* Probability bar */
.prob-track {
  background: var(--dim);
  border-radius: 2px;
  height: 4px;
  position: relative;
  margin: 0.6rem 0;
  overflow: visible;
}
.prob-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
  position: relative;
}
.prob-fill::after {
  content: '';
  position: absolute;
  right: -1px; top: -3px;
  width: 10px; height: 10px;
  border-radius: 50%;
  background: inherit;
  filter: blur(4px);
  opacity: 0.8;
}
.prob-labels {
  display: flex;
  justify-content: space-between;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.58rem;
  color: var(--muted2);
  letter-spacing: 0.08em;
  margin-bottom: 3px;
}
.thr-tick {
  position: absolute;
  top: -5px; bottom: -5px;
  width: 1px;
  background: var(--warn);
  opacity: 0.7;
}
.thr-tick::before {
  content: '▲';
  position: absolute;
  bottom: -14px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 0.45rem;
  color: var(--warn);
}

/* Risk badges */
.risk-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  margin: 0.5rem 0;
}
.risk-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  border-radius: 3px;
  padding: 0.3rem 0.65rem;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.05em;
  border: 1px solid;
}
.rb-threat {
  background: rgba(255,34,68,0.07);
  border-color: rgba(255,34,68,0.25);
  color: #ff6070;
}
.rb-clear {
  background: rgba(0,255,224,0.04);
  border-color: rgba(0,255,224,0.15);
  color: var(--glow2);
}

/* Threat level text */
.threat-level-tag {
  display: inline-block;
  font-family: 'Orbitron', monospace;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.2em;
  padding: 0.2rem 0.7rem;
  border-radius: 2px;
  margin-bottom: 0.5rem;
}
.tl-critical { background: rgba(255,34,68,0.15); color: var(--danger); border: 1px solid rgba(255,34,68,0.3); }
.tl-high     { background: rgba(255,140,0,0.12); color: #ff9922;      border: 1px solid rgba(255,140,0,0.3); }
.tl-medium   { background: rgba(255,204,0,0.1);  color: var(--warn);  border: 1px solid rgba(255,204,0,0.25); }
.tl-low      { background: rgba(0,255,224,0.06); color: var(--glow);  border: 1px solid rgba(0,255,224,0.2); }
.tl-none     { background: rgba(0,255,224,0.04); color: var(--glow2); border: 1px solid rgba(0,255,224,0.15); }

/* Status indicator */
.status-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.6rem;
  margin: 0.75rem 0 0;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.15em;
  color: var(--muted);
}
.dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--glow);
  box-shadow: 0 0 6px var(--glow);
  animation: blink 2.2s ease-in-out infinite;
}
.dot.red {
  background: var(--danger);
  box-shadow: 0 0 6px var(--danger);
  animation: blink 1.1s ease-in-out infinite;
}
@keyframes blink {
  0%,100% { opacity: 1; }
  50%      { opacity: 0.25; }
}

/* Radar animation for idle */
.radar-wrap {
  display: flex;
  justify-content: center;
  padding: 1.5rem 0;
}
@keyframes radar-sweep {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
@keyframes ping {
  0%   { r: 0; opacity: 0.8; }
  100% { r: 60; opacity: 0; }
}

/* Footer */
.hud-footer {
  text-align: center;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.55rem;
  letter-spacing: 0.25em;
  color: var(--muted2);
  text-transform: uppercase;
  margin-top: 3rem;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371 * 2 * math.asin(math.sqrt(a))

@st.cache_resource
def load_artifacts():
    if not os.path.exists("models/fraud_model.pkl"):
        return None, 0.5
    with open("models/fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    thr = 0.5
    if os.path.exists("models/threshold.pkl"):
        with open("models/threshold.pkl", "rb") as f:
            thr = pickle.load(f)
    return model, thr

model, THRESHOLD = load_artifacts()

def threat_level(prob):
    if prob >= 0.9:   return "CRITICAL", "tl-critical"
    if prob >= 0.7:   return "HIGH",     "tl-high"
    if prob >= 0.5:   return "MEDIUM",   "tl-medium"
    if prob >= 0.25:  return "LOW",      "tl-low"
    return "NONE",    "tl-none"

# ─────────────────────────────────────────────
# HUD Top Bar
# ─────────────────────────────────────────────
now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
model_status = '<span class="active">■ ONLINE</span>' if model else '<span style="color:#ff2244">■ OFFLINE</span>'

st.markdown(f"""
<div class="hud-topbar">
  <span> &nbsp;·&nbsp; NODE: FRAUD-INTEL</span>
  <span>{model_status} &nbsp;·&nbsp; XGBOOST &nbsp;·&nbsp; THR:{THRESHOLD:.3f}</span>
  <span>{now_str}</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hud-hero">
  
  <h1 class="hud-title">FRAUD DETECTION</h1>
  <p class="hud-subtitle">Real-time fraud detection &amp; transaction analysis</p>
  <div class="status-row">
    <span class="dot"></span>
    <span>ALL SYSTEMS OPERATIONAL</span>
    <span style="color:#1a4050">·</span>
    <span>XGBOOST ENGINE ACTIVE</span>
    <span style="color:#1a4050">·</span>
    <span>THRESHOLD {:.1f}%</span>
  </div>
</div>
""".format(THRESHOLD * 100), unsafe_allow_html=True)

if model is None:
    st.error("⚠️ fraud_model.pkl not found — place it next to this file.")
    st.stop()

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CATEGORIES = [
    "grocery_pos","gas_transport","home","shopping_pos","kids_pets",
    "food_dining","personal_care","health_fitness","entertainment",
    "shopping_net","misc_net","grocery_net","travel","misc_pos",
]
JOBS = [
    "Engineer","Teacher","Doctor","Nurse","Accountant","Manager",
    "Sales Representative","Lawyer","Psychologist, counselling",
    "Nature conservation officer","Patent attorney","Other",
]
DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
HIGH_RISK_CATS = {"shopping_net","misc_net","grocery_net","misc_pos"}

# ─────────────────────────────────────────────
# Form
# ─────────────────────────────────────────────
st.markdown('<div class="hud-rule"><span>01 · Transaction Parameters</span></div>', unsafe_allow_html=True)

with st.form("fraud_form"):

    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input("💳 Transaction Amount (USD)",
            min_value=0.01, max_value=50_000.0, value=150.0, step=1.0)
    with c2:
        category = st.selectbox("🏷️ Merchant Category", CATEGORIES)

    c3, c4 = st.columns(2)
    with c3:
        trans_hour = st.slider("🕐 Hour of Transaction", 0, 23, 14)
    with c4:
        age = st.number_input("👤 Cardholder Age", 18, 100, 35, step=1)

    c5, c6 = st.columns(2)
    with c5:
        distance_km = st.number_input("📍 Distance to Merchant (km)", 0.0, 5000.0, 12.0, step=0.5)
    with c6:
        gender = st.selectbox("👤 Gender", ["M", "F"])

    submitted = st.form_submit_button("⚡  INITIATE THREAT ANALYSIS")

# ─────────────────────────────────────────────
# Prediction & Results
# ─────────────────────────────────────────────
if submitted:
    now     = datetime.now()
    amt_log = np.log1p(amount)

    input_df = pd.DataFrame([{
        "amt":             amount,
        "zip":             28654,
        "lat":             36.0,
        "long":            -82.0,
        "city_pop":        50000,
        "merch_lat":       36.0 + distance_km / 111.0,
        "merch_long":      -82.0,
        "trans_hour":      trans_hour,
        "trans_dayofweek": now.weekday(),
        "trans_month":     now.month,
        "trans_day":       now.day,
        "age":             int(age),
        "amt_log":         amt_log,
        "distance_km":     distance_km,
        "merchant":        "fraud_Rippin, Kub and Mann",
        "category":        category,
        "gender":          gender,
        "city":            "Springfield",
        "state":           "NC",
        "job":             "Engineer",
    }])

    try:
        fraud_prob = float(model.predict_proba(input_df)[:, 1][0])
        is_fraud   = fraud_prob >= THRESHOLD
        tl_text, tl_cls = threat_level(fraud_prob)
        pct_str = f"{fraud_prob * 100:.1f}%"

        # ── VERDICT ──────────────────────────────
        st.markdown('<div class="hud-rule"><span>02 · Threat Assessment</span></div>', unsafe_allow_html=True)

        if is_fraud:
            vcol   = "#ff2244"
            vglow  = "rgba(255,34,68,0.6)"
            v_cls  = "verdict-bg-fraud pulse-fraud"
            v_icon = "⚠"
            v_text = "FRAUD DETECTED"
            dot_cls = "dot red"
            status_msg = "TRANSACTION FLAGGED — IMMEDIATE REVIEW REQUIRED"
        else:
            vcol   = "#00ffe0"
            vglow  = "rgba(0,255,224,0.6)"
            v_cls  = "verdict-bg-safe pulse-safe"
            v_icon = "✓"
            v_text = "TRANSACTION CLEARED"
            dot_cls = "dot"
            status_msg = "NO THREAT DETECTED — TRANSACTION APPROVED"

        # SVG circular gauge
        radius   = 70
        stroke   = 8
        circ     = 2 * math.pi * radius
        fill_len = circ * fraud_prob
        gap_len  = circ - fill_len
        thr_angle = THRESHOLD * 360 - 90
        thr_x = 90 + radius * math.cos(math.radians(thr_angle))
        thr_y = 90 + radius * math.sin(math.radians(thr_angle))

        tl_fill_col = "#ff2244" if is_fraud else "#00ffe0"
        tx1 = f"{90 + (radius-12)*math.cos(math.radians(thr_angle)):.1f}"
        ty1 = f"{90 + (radius-12)*math.sin(math.radians(thr_angle)):.1f}"
        tx2 = f"{90 + (radius+6)*math.cos(math.radians(thr_angle)):.1f}"
        ty2 = f"{90 + (radius+6)*math.sin(math.radians(thr_angle)):.1f}"
        gauge_svg = (
            f'<svg width="180" height="180" viewBox="0 0 180 180" class="gauge-svg">'
            f'<defs><filter id="glow-g">'
            f'<feGaussianBlur stdDeviation="4" result="blur"/>'
            f'<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
            f'</filter></defs>'
            f'<circle cx="90" cy="90" r="{radius}" fill="none" stroke="#0d3040" stroke-width="{stroke}"/>'
            f'<circle cx="90" cy="90" r="{radius}" fill="none" stroke="{vcol}" stroke-width="{stroke}"'
            f' stroke-dasharray="{fill_len:.1f} {gap_len:.1f}" stroke-dashoffset="{circ/4:.1f}"'
            f' stroke-linecap="round" filter="url(#glow-g)"/>'
            f'<line x1="{tx1}" y1="{ty1}" x2="{tx2}" y2="{ty2}"'
            f' stroke="#ffcc00" stroke-width="2" opacity="0.8"/>'
            f'<text x="90" y="82" text-anchor="middle" font-family="Orbitron, monospace"'
            f' font-size="22" font-weight="900" fill="{vcol}" filter="url(#glow-g)">{fraud_prob*100:.0f}</text>'
            f'<text x="90" y="97" text-anchor="middle" font-family="Share Tech Mono, monospace"'
            f' font-size="8" fill="{vcol}" opacity="0.7">PERCENT</text>'
            f'<text x="90" y="114" text-anchor="middle" font-family="Share Tech Mono, monospace"'
            f' font-size="7" fill="#1a4050">FRAUD PROB</text>'
            f'<text x="90" y="136" text-anchor="middle" font-family="Orbitron, monospace"'
            f' font-size="7" font-weight="700" fill="{tl_fill_col}" letter-spacing="2">{tl_text}</text>'
            f'</svg>'
        )

        border_top_color = 'rgba(255,34,68,0.15)' if is_fraud else 'rgba(0,255,224,0.1)'
        status_color = '#ff6070' if is_fraud else '#00ffe0'
        st.markdown(f"""
        <div class="verdict-panel {v_cls}">
          <div class="verdict-inner">
            <div style="display:flex;align-items:center;justify-content:center;gap:2rem;flex-wrap:wrap">
              <div>{gauge_svg}</div>
              <div style="flex:1;min-width:200px">
                <div class="threat-level-tag {tl_cls}">THREAT LEVEL: {tl_text}</div>
                <div class="verdict-big" style="color:{vcol};text-shadow:0 0 20px {vglow},0 0 40px {vglow}30">
                  {v_icon} {v_text}
                </div>
                <div class="verdict-pct" style="color:{vcol};text-shadow:0 0 30px {vglow},0 0 60px {vglow}40">
                  {pct_str}
                </div>
                <div class="verdict-pct-sub" style="color:{vcol}60">FRAUD PROBABILITY INDEX</div>
              </div>
            </div>
            <div class="status-row" style="margin-top:1.5rem;padding-top:1rem;border-top:1px solid {border_top_color}">
              <span class="{dot_cls}"></span>
              <span style="color:{status_color}">{status_msg}</span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Probability bar ──────────────────────
        bar_pct = min(fraud_prob * 100, 100)
        thr_pct = min(THRESHOLD * 100, 100)
        st.markdown(f"""
        <div style="margin:1rem 0 1.5rem">
          <div class="prob-labels">
            <span>0.0%</span>
            <span style="color:#ffcc00">▲ THRESHOLD {THRESHOLD*100:.1f}%</span>
            <span>100%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width:{bar_pct}%;
              background:linear-gradient(90deg,{vcol}55,{vcol})">
            </div>
            <div class="thr-tick" style="left:{thr_pct}%"></div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Metrics ──────────────────────────────
        st.markdown('<div class="hud-rule"><span>03 · Signal Metrics</span></div>', unsafe_allow_html=True)

        delta  = fraud_prob - THRESHOLD
        d_col  = "#ff2244" if delta > 0 else "#00ffe0"
        d_sign = "+" if delta > 0 else ""

        def mc(val, lbl, col="#c8e8e0"):
            return f"""<div class="metric-cell">
              <div class="metric-val" style="color:{col}">{val}</div>
              <div class="metric-lbl">{lbl}</div>
            </div>"""

        cat_display = category.replace("_"," ").upper()
        st.markdown(f"""
        <div class="metric-grid" style="grid-template-columns:repeat(3,1fr)">
          {mc(f"${amount:,.0f}", "AMOUNT")}
          {mc(cat_display, "CATEGORY")}
          {mc(f"{trans_hour:02d}:00", "HOUR")}
        </div>
        <div class="metric-grid" style="grid-template-columns:repeat(3,1fr)">
          {mc(str(age), "AGE")}
          {mc(f"{distance_km:.0f} km", "DISTANCE")}
          {mc(gender, "GENDER")}
        </div>""", unsafe_allow_html=True)

        # ── Risk Analysis ────────────────────────
        st.markdown('<div class="hud-rule"><span>04 · Risk Signal Analysis</span></div>', unsafe_allow_html=True)

        signals = []
        if amount > 1000:   signals.append(("threat", f"💰 Very high amount — ${amount:,.0f}"))
        elif amount > 400:  signals.append(("threat", f"💰 Elevated amount — ${amount:,.0f}"))
        else:               signals.append(("clear",  f"💰 Amount within range — ${amount:,.0f}"))

        if trans_hour < 6 or trans_hour > 22:
            signals.append(("threat", f"🌙 Off-hours transaction — {trans_hour:02d}:00"))
        else:
            signals.append(("clear",  f"🕐 Normal business hours — {trans_hour:02d}:00"))

        if category in HIGH_RISK_CATS:
            signals.append(("threat", f"🛒 High-risk category — {category}"))
        else:
            signals.append(("clear",  f"🏷 Low-risk category — {category}"))

        if distance_km > 200:  signals.append(("threat", f"📍 Long-range merchant — {distance_km:.0f} km"))
        elif distance_km > 80: signals.append(("threat", f"📍 Distant merchant — {distance_km:.0f} km"))
        else:                  signals.append(("clear",  f"📍 Local merchant — {distance_km:.0f} km"))

        # Age risk logic:
        # 18–25 → young adults, higher impulsive/targeted fraud risk
        # 26–60 → prime working age, lowest fraud risk
        # 61–75 → seniors, slightly elevated (scam targeting)
        # 76+   → high risk (elder fraud)
        if age <= 25:
            signals.append(("threat", f"👤 Young adult — {age} yrs (higher targeted risk)"))
        elif age <= 60:
            signals.append(("clear",  f"👤 Prime age — {age} yrs (low risk)"))
        elif age <= 75:
            signals.append(("threat", f"👤 Senior — {age} yrs (elevated scam risk)"))
        else:
            signals.append(("threat", f"👤 Elder — {age} yrs (high fraud vulnerability)"))

        threats = sum(1 for s in signals if s[0] == "threat")
        html_badges = "".join(
            f'<span class="risk-badge rb-threat">⚑ {t}</span>' if k == "threat"
            else f'<span class="risk-badge rb-clear">✓ {t}</span>'
            for k, t in signals
        )
        st.markdown(f"""
        <div style="margin-bottom:0.5rem;font-family:'Share Tech Mono',monospace;
             font-size:0.62rem;letter-spacing:0.15em;color:var(--muted)">
          SIGNALS DETECTED: <span style="color:{'#ff6070' if threats else '#00ffe0'}">{threats} THREAT{"S" if threats!=1 else ""}</span>
          &nbsp;·&nbsp; {len(signals)-threats} CLEAR
        </div>
        <div class="risk-grid">{html_badges}</div>
        """, unsafe_allow_html=True)

        # ── Audit log line ───────────────────────
        st.markdown(f"""
        <div style="margin-top:1.25rem;padding:0.8rem 1rem;background:var(--panel);
                    border:1px solid var(--border);border-left:3px solid {vcol};
                    border-radius:4px;font-family:'Share Tech Mono',monospace;
                    font-size:0.65rem;letter-spacing:0.08em;color:var(--muted);
                    line-height:1.8">
          <span style="color:{vcol}">■</span> AUDIT LOG &nbsp;·&nbsp;
          {now.strftime('%Y-%m-%d %H:%M:%S')} &nbsp;·&nbsp;
          MODEL: XGBOOST &nbsp;·&nbsp;
          THRESHOLD: {THRESHOLD:.4f} &nbsp;·&nbsp;
          P(FRAUD): {fraud_prob:.6f} &nbsp;·&nbsp;
          VERDICT: {v_text} &nbsp;·&nbsp;
          AMT_LOG: {amt_log:.4f}
        </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"[ERROR] Prediction failed: {e}")
        st.code(str(input_df.dtypes))

else:
    # ── Idle state with animated radar ──────────
    st.markdown('<div class="hud-rule"><span>02 · Threat Assessment</span></div>', unsafe_allow_html=True)

    idle_time = datetime.now().strftime("%H:%M:%S")
    radar_svg = (
        '<svg width="160" height="160" viewBox="0 0 160 160">'
        '<defs><radialGradient id="rg">'
        '<stop offset="0%" stop-color="#00ffe0" stop-opacity="0.03"/>'
        '<stop offset="100%" stop-color="#00ffe0" stop-opacity="0"/>'
        '</radialGradient></defs>'
        '<circle cx="80" cy="80" r="20" fill="none" stroke="#0d3040" stroke-width="0.8"/>'
        '<circle cx="80" cy="80" r="40" fill="none" stroke="#0d3040" stroke-width="0.8"/>'
        '<circle cx="80" cy="80" r="60" fill="none" stroke="#0d3040" stroke-width="0.8"/>'
        '<line x1="80" y1="15" x2="80" y2="145" stroke="#0d3040" stroke-width="0.6"/>'
        '<line x1="15" y1="80" x2="145" y2="80" stroke="#0d3040" stroke-width="0.6"/>'
        '<g style="transform-origin:80px 80px;animation:radar-sweep 4s linear infinite">'
        '<path d="M80,80 L80,20 A60,60 0 0,1 137,110 Z" fill="url(#rg)" opacity="0.8"/>'
        '<line x1="80" y1="80" x2="80" y2="20" stroke="#00ffe0" stroke-width="1" opacity="0.6"/>'
        '</g>'
        '<circle cx="104" cy="52" r="2.5" fill="#00ffe0" opacity="0.6"/>'
        '<circle cx="58" cy="100" r="2" fill="#00ffe0" opacity="0.4"/>'
        '<circle cx="80" cy="80" r="3" fill="#00ffe0" style="filter:drop-shadow(0 0 4px #00ffe0)"/>'
        '<text x="80" y="155" text-anchor="middle" font-family="Share Tech Mono, monospace"'
        ' font-size="6" fill="#1a4050" letter-spacing="3">SCANNING</text>'
        '</svg>'
    )
    st.markdown(f"""
    <div class="verdict-panel verdict-bg-idle" style="padding:2rem 1.5rem;text-align:center">
      <div class="radar-wrap">{radar_svg}</div>
      <div style="font-family:'Orbitron',monospace;font-size:0.7rem;font-weight:700;
                  letter-spacing:0.3em;color:#1a4050;margin-bottom:0.5rem">AWAITING INPUT</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;
                  color:#0d3040;letter-spacing:0.1em">
        ENTER TRANSACTION PARAMETERS ABOVE &nbsp;&#183;&nbsp; THEN INITIATE ANALYSIS
      </div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;
                  color:#0d2530;margin-top:0.75rem;letter-spacing:0.08em">
        SYS TIME: {idle_time} &nbsp;&#183;&nbsp;  &nbsp;&#183;&nbsp; MONITORING
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class="hud-footer">
  FRAUD INTELLIGENCE ENGINE · XGBOOST CLASSIFIER · ALL RIGHTS RESERVED
  <br style="margin:0.3rem">
  <span style="color:#0d2530">FOR AUTHORIZED USE ONLY · CLASSIFICATION: INTERNAL</span>
</div>
""", unsafe_allow_html=True)