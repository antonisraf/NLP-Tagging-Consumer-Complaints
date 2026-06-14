import os
import time
import re
import json
import streamlit as st
import pandas as pd

from complaint_generator import generate_synthetic_complaints
from model_pipeline import HierarchicalComplaintClassifier, DEFAULT_REJECTION_THRESHOLD

st.set_page_config(page_title="Complaint Classifier AI", layout="wide")

# ── SESSION STATE DEFAULTS ────────────────────────────────────────────────────
defaults = {
    "dark_mode":          True,
    "results_df":         None,
    "chat_messages":      [],
    "results_open":       True,
    "chat_open":          True,
    "chat_fullscreen":    False,
    "results_fullscreen": False,
    "groq_api_key":       None,
    "rate_limit_count":   0,
    "rate_limit_ts":      0.0,
    "key_validated":      False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── RATE LIMIT CONFIG ─────────────────────────────────────────────────────────
RATE_LIMIT_MAX    = 5
RATE_LIMIT_WINDOW = 3600

def check_rate_limit():
    now     = time.time()
    elapsed = now - st.session_state.rate_limit_ts
    if elapsed > RATE_LIMIT_WINDOW:
        st.session_state.rate_limit_count = 0
        st.session_state.rate_limit_ts    = now
    if st.session_state.rate_limit_count >= RATE_LIMIT_MAX:
        remaining = int(RATE_LIMIT_WINDOW - elapsed)
        return False, max(remaining, 0)
    return True, 0

def increment_rate_limit():
    if st.session_state.rate_limit_count == 0:
        st.session_state.rate_limit_ts = time.time()
    st.session_state.rate_limit_count += 1

def validate_api_key(key: str) -> bool:
    return isinstance(key, str) and key.strip().startswith("gsk_") and len(key.strip()) > 20

active_key = st.session_state.groq_api_key

# ── THEME ─────────────────────────────────────────────────────────────────────
DARK_BG      = "#0f0e17"
DARK_SURFACE = "#1a1828"
LIGHT_BG     = "#ffffff"
LIGHT_SURFACE= "#f8f7ff"

bg      = DARK_BG      if st.session_state.dark_mode else LIGHT_BG
surface = DARK_SURFACE if st.session_state.dark_mode else LIGHT_SURFACE
text    = "#fffffe"    if st.session_state.dark_mode else "#0f0e17"
muted   = "#a89fc0"    if st.session_state.dark_mode else "#6b6b80"
border  = "rgba(124,58,237,0.25)" if st.session_state.dark_mode else "rgba(124,58,237,0.15)"
is_dark = "true" if st.session_state.dark_mode else "false"

st.markdown(f"""
<style>
:root {{
    --g: linear-gradient(45deg, #7c3aed, #db2777);
    --g-hover: linear-gradient(45deg, #6d28d9, #be185d);
    --bg: {bg};
    --surface: {surface};
    --text: {text};
    --muted: {muted};
    --border: {border};
}}
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
.main, .block-container {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    padding-top: 0 !important;
}}
[data-testid="stHeader"] {{ background-color: var(--bg) !important; }}
[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {{
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}}
p, span, label, li, div, h1, h2, h3, h4, caption, th, td {{
    color: var(--text) !important;
}}
[data-testid="stSidebar"] h2 {{
    background: var(--g);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
div.stButton > button[kind="primary"] {{
    background: var(--g) !important;
    color: white !important; border: none !important; font-weight: 600;
}}
div.stButton > button[kind="primary"]:hover {{
    background: var(--g-hover) !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.4);
}}
div.stButton > button {{
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    transition: all 0.2s ease;
}}
div.stButton > button:hover {{
    border-color: #db2777 !important;
    color: #db2777 !important;
}}
.panel-btn-row {{ display: flex; gap: 8px; }}
.panel-btn {{
    display: inline-flex; align-items: center; justify-content: center;
    gap: 5px; padding: 6px 14px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--surface);
    color: var(--muted) !important; font-size: 0.8rem; font-weight: 600;
    cursor: pointer; white-space: nowrap; transition: all 0.2s ease; min-width: 80px;
}}
.panel-btn:hover {{ border-color: #7c3aed; color: #7c3aed !important; }}
.panel-btn.active {{
    background: linear-gradient(45deg,rgba(124,58,237,0.15),rgba(219,39,119,0.15));
    border-color: #7c3aed; color: var(--text) !important;
}}
div[data-baseweb="slider"] [role="slider"] {{
    background: #db2777 !important;
    border: 2px solid {bg} !important;
    box-shadow: 0 0 0 2px #7c3aed !important;
}}
div[data-baseweb="slider"] [class*="Track"]:not([class*="Fill"]) {{
    background: rgba(124,58,237,0.15) !important;
}}
div[data-testid="stAlert"] {{
    background: linear-gradient(90deg, rgba(124,58,237,0.15), rgba(219,39,119,0.15)) !important;
    border: none !important; border-left: 4px solid #db2777 !important;
    border-radius: 8px !important;
}}
div[data-testid="stAlert"] * {{ color: var(--text) !important; }}
div[data-testid="stAlert"] svg {{ display: none !important; }}
.chat-container {{
    display: flex; flex-direction: column; gap: 12px;
    padding: 1rem 0; max-height: 420px; overflow-y: auto;
}}
.chat-bubble {{ display: flex; flex-direction: column; max-width: 80%; }}
.chat-bubble.user {{ align-self: flex-end; align-items: flex-end; }}
.chat-bubble.assistant {{ align-self: flex-start; align-items: flex-start; }}
.chat-bubble .bubble-inner {{
    padding: 10px 16px; border-radius: 16px; font-size: 0.9rem; line-height: 1.5;
}}
.chat-bubble.user .bubble-inner {{
    background: linear-gradient(45deg, #7c3aed, #db2777);
    color: white !important; border-bottom-right-radius: 4px;
}}
.chat-bubble.assistant .bubble-inner {{
    background: var(--surface); border: 1px solid var(--border);
    color: var(--text) !important; border-bottom-left-radius: 4px;
}}
.chat-bubble .bubble-label {{ font-size: 0.72rem; opacity: 0.5; margin-bottom: 4px; padding: 0 4px; }}
.panel-header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 18px; background: var(--surface);
    border: 1px solid var(--border); border-radius: 12px; margin-bottom: 4px;
}}
.panel-header-title {{
    font-weight: 700; font-size: 1rem; background: var(--g);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.key-modal {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 2.5rem 2rem;
    max-width: 480px; margin: 4rem auto; text-align: center;
}}
.key-modal-title {{
    font-size: 1.4rem; font-weight: 700; background: var(--g);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;
}}
.key-modal-sub {{ font-size: 0.9rem; opacity: 0.6; margin-bottom: 1.5rem; line-height: 1.5; }}
hr {{ border-color: var(--border) !important; }}
div[data-testid="stSpinner"] > div {{ border-top-color: #db2777 !important; }}
a[data-testid="stDownloadButton"] button {{
    background: var(--g) !important; color: white !important; border: none !important;
}}
.stCaption, [data-testid="stCaptionContainer"] * {{ color: var(--muted) !important; }}
div[data-testid="stTextInput"] input {{
    background: var(--bg) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
}}
div[data-testid="stTextInput"] input:focus {{
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}}
</style>
""", unsafe_allow_html=True)

st.components.v1.html(f"""
<script>
function fixSliders() {{
    document.querySelectorAll('[data-baseweb="slider"]').forEach(slider => {{
        slider.querySelectorAll('[role="slider"]').forEach(t => {{
            t.style.background = '#db2777';
            t.style.borderColor = '{bg}';
            t.style.boxShadow = '0 0 0 2px #7c3aed';
        }});
        slider.querySelectorAll('[class*="TrackFill"],[class*="trackFill"]').forEach(f => {{
            f.style.background = 'linear-gradient(45deg,#7c3aed,#db2777)';
        }});
        slider.querySelectorAll('[class*="Track"]:not([class*="Fill"])').forEach(t => {{
            t.style.background = 'rgba(124,58,237,0.15)';
        }});
    }});
}}
fixSliders();
new MutationObserver(fixSliders).observe(document.body, {{childList:true,subtree:true,attributes:true}});
</script>
""", height=0)

@st.cache_resource
def load_classifier():
    return HierarchicalComplaintClassifier()

def render_chat(height="420px"):
    if not st.session_state.chat_messages:
        st.markdown(f"""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;
            padding:2.5rem;text-align:center;color:var(--muted);height:180px;
            display:flex;align-items:center;justify-content:center;flex-direction:column;gap:8px">
  <div style="font-size:2rem">🤖</div>
  <div style="font-size:0.95rem">Hit <b>Generate & Classify</b> to start the pipeline.<br>Results will appear here.</div>
</div>
""", unsafe_allow_html=True)
    else:
        bubbles = ""
        for msg in st.session_state.chat_messages:
            role    = msg["role"]
            content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', msg["content"])
            content = content.replace("\n", "<br>")
            label   = "You" if role == "user" else "Pipeline AI"
            bubbles += f"""
<div class="chat-bubble {role}">
  <div class="bubble-label">{label}</div>
  <div class="bubble-inner">{content}</div>
</div>"""
        st.markdown(f"""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;
            padding:1rem 1.2rem;overflow-y:auto;height:{height}">
  <div class="chat-container">{bubbles}</div>
</div>
""", unsafe_allow_html=True)

def panel_header(title, is_open, is_full, key_fs, key_toggle):
    col_title, col_btns = st.columns([7, 3])
    with col_title:
        arrow = "▼" if is_open else "▶"
        st.markdown(f"""
<div class="panel-header">
  <span class="panel-header-title">{title}</span>
  <span style="opacity:0.5;font-size:0.85rem">{arrow}</span>
</div>
""", unsafe_allow_html=True)
    with col_btns:
        b1, b2 = st.columns(2)
        with b1:
            fs_label = "⤡ Exit" if is_full else "⤢ Full"
            clicked_fs = st.button(fs_label, key=key_fs, use_container_width=True)
        with b2:
            tog_label = "▲ Hide" if is_open else "▼ Show"
            clicked_tog = st.button(tog_label, key=key_toggle, use_container_width=True)
    return clicked_fs, clicked_tog


def render_dashboard(results: pd.DataFrame, dark_mode: bool):
    cols_needed = [
        "complaint_text", "true_issue", "predicted_issue_broad",
        "issue_correct", "true_subissue", "predicted_subissue",
        "subissue_correct", "joint_confidence", "needs_review",
    ]
    df = results[cols_needed].copy()
    df["issue_correct"]    = df["issue_correct"].astype(bool)
    df["subissue_correct"] = df["subissue_correct"].astype(bool)
    df["needs_review"]     = df["needs_review"].astype(bool)
    records_json = df.to_json(orient="records")

    theme = {
        "bg":      "#0f0e17" if dark_mode else "#ffffff",
        "surface": "#1a1828" if dark_mode else "#f8f7ff",
        "text":    "#fffffe" if dark_mode else "#0f0e17",
        "muted":   "#a89fc0" if dark_mode else "#6b6b80",
        "border":  "rgba(124,58,237,0.25)" if dark_mode else "rgba(124,58,237,0.15)",
        "isDark":  "true" if dark_mode else "false",
    }

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}}
:root{{
  --bg:{theme['bg']};--surface:{theme['surface']};--text:{theme['text']};
  --muted:{theme['muted']};--border:{theme['border']};
  --g:linear-gradient(45deg,#7c3aed,#db2777);
  --purple:#7c3aed;--pink:#db2777;--purple-light:#a78bfa;--pink-light:#f472b6;
}}
body{{background:var(--bg);color:var(--text);padding:12px 4px 24px}}

/* ── filter bar ── */
.filter-bar{{
  display:flex;flex-wrap:wrap;gap:8px;align-items:center;
  padding:10px 14px;background:var(--surface);
  border:1px solid var(--border);border-radius:12px;margin-bottom:14px
}}
.filter-bar label{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}}
.filter-bar select{{
  font-size:11px;padding:4px 7px;border-radius:6px;
  border:1px solid var(--border);background:var(--bg);
  color:var(--text);cursor:pointer;outline:none
}}
.filter-bar select:focus{{border-color:#7c3aed}}
.filter-bar input[type=range]{{width:80px;cursor:pointer;accent-color:#db2777}}
.fval{{font-size:11px;font-weight:600;color:var(--text);min-width:30px}}
.fsep{{width:1px;height:18px;background:var(--border);margin:0 2px;flex-shrink:0}}
.reset-btn{{
  margin-left:auto;font-size:11px;padding:4px 10px;border-radius:6px;
  border:1px solid var(--border);background:transparent;color:var(--muted);cursor:pointer
}}
.reset-btn:hover{{color:var(--text);border-color:#7c3aed}}
.active-pill{{
  font-size:10px;padding:2px 8px;border-radius:20px;
  background:rgba(219,39,119,0.15);color:#db2777;font-weight:600;border:1px solid #db2777
}}

/* ── metrics ── */
.metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:14px}}
.metric{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px}}
.metric-label{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}}
.metric-value{{font-size:22px;font-weight:700;color:var(--text);line-height:1}}
.metric-value.accent{{background:var(--g);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.metric-value.needs-review-val{{color:#f472b6}}
.metric-sub{{font-size:10px;color:var(--muted);margin-top:4px}}

/* ── tabs ── */
.tabs{{display:flex;border-bottom:1px solid var(--border);margin-bottom:14px}}
.tab{{
  padding:7px 16px;font-size:12px;color:var(--muted);cursor:pointer;
  border-bottom:2px solid transparent;transition:all .15s;user-select:none
}}
.tab.active{{color:#7c3aed;border-bottom-color:#db2777;font-weight:600}}
.tab-panel{{display:none}}.tab-panel.active{{display:block}}

/* ── grid layouts ── */
.row2{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}
.row3{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-bottom:10px}}

/* ── cards ── */
.card{{
  background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:14px 16px
}}
.card-title{{
  font-size:10px;font-weight:600;color:var(--muted);
  text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px
}}

/* ── bar rows ── */
.bar-row{{display:flex;align-items:center;gap:8px;margin-bottom:7px}}
.bar-lbl{{font-size:10px;color:var(--muted);width:114px;flex-shrink:0;text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.bar-track{{flex:1;height:7px;background:rgba(124,58,237,0.12);border-radius:4px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:4px;transition:width .4s ease}}
.bar-val{{font-size:10px;color:var(--text);width:30px;flex-shrink:0;font-weight:600}}

/* ── breakdown tables ── */
.btbl{{width:100%;border-collapse:collapse;font-size:11px}}
.btbl th{{
  text-align:left;padding:6px 8px;font-size:10px;text-transform:uppercase;
  letter-spacing:.05em;color:var(--muted);border-bottom:1px solid var(--border);font-weight:600
}}
.btbl td{{padding:7px 8px;border-bottom:1px solid var(--border);color:var(--text)}}
.btbl tr:last-child td{{border-bottom:none}}
.btbl tr:hover td{{background:rgba(124,58,237,0.07)}}

/* ── detail table ── */
.dtbl-wrap{{overflow-x:auto;overflow-y:auto;max-height:480px;border-radius:12px;border:1px solid var(--border)}}
.dtbl{{width:100%;border-collapse:collapse;font-size:11px;min-width:820px}}
.dtbl th{{
  text-align:left;padding:8px 10px;font-size:10px;text-transform:uppercase;
  letter-spacing:.05em;color:var(--muted);border-bottom:1px solid var(--border);
  font-weight:600;cursor:pointer;user-select:none;white-space:nowrap;
  background:var(--surface);position:sticky;top:0;z-index:2
}}
.dtbl th:hover{{color:var(--text)}}
.dtbl th .si{{opacity:.35;font-size:9px;margin-left:3px}}
.dtbl th .si.on{{opacity:1;color:#db2777}}
.dtbl td{{
  padding:8px 10px;border-bottom:1px solid var(--border);
  color:var(--text);vertical-align:middle;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:180px
}}
.dtbl tr:last-child td{{border-bottom:none}}
.dtbl tr:hover td{{background:rgba(124,58,237,0.06)}}

/* ── badges & chips — purple/pink palette ── */
.badge{{display:inline-block;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:600}}
.b-auto{{background:rgba(124,58,237,0.15);color:#a78bfa;border:1px solid rgba(124,58,237,0.4)}}
.b-review{{background:rgba(219,39,119,0.15);color:#f472b6;border:1px solid rgba(219,39,119,0.4)}}
.conf-chip{{display:inline-block;padding:2px 7px;border-radius:12px;font-size:10px;font-weight:600}}
.c-hi{{background:rgba(124,58,237,0.2);color:#a78bfa}}
.c-mid{{background:rgba(167,48,132,0.18);color:#e879b0}}
.c-lo{{background:rgba(219,39,119,0.2);color:#f472b6}}
.match-y{{color:#a78bfa;font-weight:700}}
.match-n{{color:#db2777;font-weight:700}}

/* ── legend ── */
.legend{{display:flex;flex-wrap:wrap;gap:12px;font-size:10px;color:var(--muted);margin-bottom:8px}}
.lsq{{width:8px;height:8px;border-radius:2px;display:inline-block;margin-right:4px;flex-shrink:0}}

/* ── empty state ── */
.empty{{padding:28px;text-align:center;font-size:12px;color:var(--muted)}}
</style>
</head>
<body>

<!-- FILTER BAR -->
<div class="filter-bar">
  <label>Status</label>
  <select id="f-status">
    <option value="all">All</option>
    <option value="auto">Auto only</option>
    <option value="review">Review only</option>
  </select>
  <div class="fsep"></div>
  <label>Broad issue</label>
  <select id="f-issue"><option value="all">All</option></select>
  <div class="fsep"></div>
  <label>Sub-issue</label>
  <select id="f-sub"><option value="all">All</option></select>
  <div class="fsep"></div>
  <label>L1</label>
  <select id="f-l1">
    <option value="all">All</option>
    <option value="correct">Correct</option>
    <option value="wrong">Wrong</option>
  </select>
  <div class="fsep"></div>
  <label>L2</label>
  <select id="f-l2">
    <option value="all">All</option>
    <option value="correct">Correct</option>
    <option value="wrong">Wrong</option>
  </select>
  <div class="fsep"></div>
  <label>Conf ≥</label>
  <input type="range" id="f-conf" min="0" max="100" value="0" step="5">
  <span class="fval" id="fval">0%</span>
  <button class="reset-btn" id="reset-btn">↺ Reset</button>
  <span class="active-pill" id="fpill" style="display:none"></span>
</div>

<!-- METRICS -->
<div class="metrics">
  <div class="metric">
    <div class="metric-label">Showing</div>
    <div class="metric-value" id="m-total">—</div>
    <div class="metric-sub" id="m-total-sub">of — total</div>
  </div>
  <div class="metric">
    <div class="metric-label">Auto-labelled</div>
    <div class="metric-value accent" id="m-auto">—</div>
    <div class="metric-sub" id="m-auto-pct">—</div>
  </div>
  <div class="metric">
    <div class="metric-label">Needs review</div>
    <div class="metric-value needs-review-val" id="m-review">—</div>
    <div class="metric-sub" id="m-review-pct">—</div>
  </div>
  <div class="metric">
    <div class="metric-label">L1 accuracy</div>
    <div class="metric-value accent" id="m-l1">—</div>
    <div class="metric-sub">auto-labelled only</div>
  </div>
  <div class="metric">
    <div class="metric-label">L2 accuracy</div>
    <div class="metric-value accent" id="m-l2">—</div>
    <div class="metric-sub">auto-labelled only</div>
  </div>
</div>

<!-- TABS -->
<div class="tabs">
  <div class="tab active" data-tab="overview">Overview</div>
  <div class="tab" data-tab="breakdown">Category breakdown</div>
  <div class="tab" data-tab="detail">Per-complaint detail</div>
</div>

<!-- TAB: OVERVIEW -->
<div id="tab-overview" class="tab-panel active">
  <div class="row3">
    <div class="card">
      <div class="card-title">Accuracy — auto-labelled only</div>
      <div id="acc-bars"></div>
    </div>
    <div class="card">
      <div class="card-title">Confidence distribution</div>
      <div style="position:relative;height:148px">
        <canvas id="confChart" role="img" aria-label="Histogram of joint confidence scores">Confidence histogram.</canvas>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Routing split</div>
      <div class="legend">
        <span><span class="lsq" style="background:#7c3aed"></span>Auto</span>
        <span><span class="lsq" style="background:#db2777"></span>Review</span>
      </div>
      <div style="position:relative;height:120px">
        <canvas id="routeChart" role="img" aria-label="Donut chart — auto vs review routing split">Routing donut.</canvas>
      </div>
    </div>
  </div>
  <div class="row2">
    <div class="card">
      <div class="card-title">L1 accuracy by broad issue</div>
      <div id="l1-cat"></div>
    </div>
    <div class="card">
      <div class="card-title">L2 accuracy by sub-issue</div>
      <div id="l2-cat"></div>
    </div>
  </div>
</div>

<!-- TAB: BREAKDOWN -->
<div id="tab-breakdown" class="tab-panel">
  <div class="row2" style="margin-top:4px">
    <div class="card">
      <div class="card-title">Broad issue — truth vs predicted</div>
      <table class="btbl" id="tbl-l1">
        <thead><tr><th>Category</th><th>Truth</th><th>Predicted</th><th>Diff</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="card">
      <div class="card-title">Sub-issue — truth vs predicted</div>
      <table class="btbl" id="tbl-l2">
        <thead><tr><th>Category</th><th>Truth</th><th>Predicted</th><th>Diff</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>

<!-- TAB: DETAIL -->
<div id="tab-detail" class="tab-panel">
  <div class="dtbl-wrap" style="margin-top:4px">
    <table class="dtbl" id="dtbl">
      <thead>
        <tr>
          <th style="width:32px" data-col="idx"># <span class="si" id="si-idx"></span></th>
          <th style="width:200px">Complaint</th>
          <th style="width:60px">Status</th>
          <th data-col="true_issue">True issue <span class="si" id="si-true_issue"></span></th>
          <th data-col="predicted_issue_broad">Pred issue <span class="si" id="si-predicted_issue_broad"></span></th>
          <th style="width:32px" data-col="issue_correct">L1 <span class="si" id="si-issue_correct"></span></th>
          <th data-col="true_subissue">True sub-issue <span class="si" id="si-true_subissue"></span></th>
          <th data-col="predicted_subissue">Pred sub-issue <span class="si" id="si-predicted_subissue"></span></th>
          <th style="width:32px" data-col="subissue_correct">L2 <span class="si" id="si-subissue_correct"></span></th>
          <th style="width:62px" data-col="joint_confidence">Conf <span class="si" id="si-joint_confidence"></span></th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
    <div class="empty" id="detail-empty" style="display:none">No complaints match the current filters.</div>
  </div>
</div>

<script>
const ALL = {records_json};
const TOTAL = ALL.length;
const DARK  = {theme['isDark']};

let sortCol = 'idx', sortDir = 1;
let confChartInst, routeChartInst;

/* ── helpers ── */
function unique(arr){{ return [...new Set(arr)].sort(); }}

function fillSelect(id, vals){{
  const sel = document.getElementById(id);
  const cur = sel.value;
  while(sel.options.length > 1) sel.remove(1);
  vals.forEach(v => {{ const o = document.createElement('option'); o.value = v; o.textContent = v; sel.appendChild(o); }});
  if([...sel.options].some(o => o.value === cur)) sel.value = cur;
}}

function getFilters(){{
  return {{
    status: document.getElementById('f-status').value,
    issue:  document.getElementById('f-issue').value,
    sub:    document.getElementById('f-sub').value,
    l1:     document.getElementById('f-l1').value,
    l2:     document.getElementById('f-l2').value,
    conf:   parseInt(document.getElementById('f-conf').value) / 100,
  }};
}}

function filtered(){{
  const f = getFilters();
  return ALL.filter((r, i) => {{
    if(f.status === 'auto'    &&  r.needs_review)   return false;
    if(f.status === 'review'  && !r.needs_review)   return false;
    if(f.issue  !== 'all'     &&  r.true_issue            !== f.issue) return false;
    if(f.sub    !== 'all'     &&  r.true_subissue          !== f.sub)   return false;
    if(f.l1     === 'correct' && !r.issue_correct)    return false;
    if(f.l1     === 'wrong'   &&  r.issue_correct)    return false;
    if(f.l2     === 'correct' && !r.subissue_correct) return false;
    if(f.l2     === 'wrong'   &&  r.subissue_correct) return false;
    if(r.joint_confidence < f.conf) return false;
    return true;
  }}).map(r => ({{ ...r, idx: ALL.indexOf(r) }}));
}}

function activeFilterCount(){{
  const f = getFilters(); let c = 0;
  if(f.status !== 'all') c++;
  if(f.issue  !== 'all') c++;
  if(f.sub    !== 'all') c++;
  if(f.l1     !== 'all') c++;
  if(f.l2     !== 'all') c++;
  if(f.conf   >  0)      c++;
  return c;
}}

function barRow(label, val, color){{
  const pct = isNaN(val) ? 0 : Math.round(val * 100);
  const short = label.length > 18 ? label.slice(0,18) + '…' : label;
  return `<div class="bar-row">
    <div class="bar-lbl" title="${{label}}">${{short}}</div>
    <div class="bar-track"><div class="bar-fill" style="width:${{pct}}%;background:${{color}}"></div></div>
    <div class="bar-val">${{pct}}%</div>
  </div>`;
}}

/* ── main render ── */
function render(){{
  const data   = filtered();
  const autoR  = data.filter(r => !r.needs_review);
  const revR   = data.filter(r =>  r.needs_review);
  const l1Acc  = autoR.length ? autoR.filter(r => r.issue_correct).length    / autoR.length : 0;
  const l2Acc  = autoR.length ? autoR.filter(r => r.subissue_correct).length / autoR.length : 0;

  /* metrics */
  document.getElementById('m-total').textContent      = data.length;
  document.getElementById('m-total-sub').textContent  = `of ${{TOTAL}} total`;
  document.getElementById('m-auto').textContent       = autoR.length;
  document.getElementById('m-auto-pct').textContent   = data.length ? Math.round(autoR.length / data.length * 100) + '% of filtered' : '—';
  document.getElementById('m-review').textContent     = revR.length;
  document.getElementById('m-review-pct').textContent = data.length ? Math.round(revR.length  / data.length * 100) + '% of filtered' : '—';
  document.getElementById('m-l1').textContent         = autoR.length ? Math.round(l1Acc * 100) + '%' : '—';
  document.getElementById('m-l2').textContent         = autoR.length ? Math.round(l2Acc * 100) + '%' : '—';

  /* filter badge */
  const ac = activeFilterCount();
  const pill = document.getElementById('fpill');
  pill.textContent = ac + ' filter' + (ac > 1 ? 's' : '') + ' active';
  pill.style.display = ac > 0 ? '' : 'none';

  /* accuracy bars — purple for correct, pink for incorrect */
  document.getElementById('acc-bars').innerHTML =
    barRow('L1 correct',   l1Acc,     '#7c3aed') +
    barRow('L1 incorrect', 1 - l1Acc, '#db2777') +
    barRow('L2 correct',   l2Acc,     '#7c3aed') +
    barRow('L2 incorrect', 1 - l2Acc, '#db2777');

  /* confidence histogram */
  const bins = [0,0,0,0,0,0,0,0,0,0];
  data.forEach(r => {{ const b = Math.min(9, Math.floor(r.joint_confidence * 10)); bins[b]++; }});
  confChartInst.data.datasets[0].data = bins;
  confChartInst.update('none');

  /* routing donut — purple / pink */
  routeChartInst.data.datasets[0].data = [autoR.length, revR.length];
  routeChartInst.update('none');

  /* L1 per-category bars */
  const issues = unique(data.map(r => r.true_issue));
  document.getElementById('l1-cat').innerHTML = issues.length
    ? issues.map(iss => {{
        const rows = autoR.filter(r => r.true_issue === iss);
        return barRow(iss, rows.length ? rows.filter(r => r.issue_correct).length / rows.length : 0, '#7c3aed');
      }}).join('')
    : '<div style="font-size:11px;color:var(--muted);padding:8px 0">No data for current filters.</div>';

  /* L2 per-category bars */
  const subs = unique(data.map(r => r.true_subissue));
  document.getElementById('l2-cat').innerHTML = subs.length
    ? subs.map(s => {{
        const rows = autoR.filter(r => r.true_subissue === s);
        return barRow(s, rows.length ? rows.filter(r => r.subissue_correct).length / rows.length : 0, '#9d3adb');
      }}).join('')
    : '<div style="font-size:11px;color:var(--muted);padding:8px 0">No data for current filters.</div>';

  /* breakdown tables */
  const allIss = unique([...data.map(r => r.true_issue), ...data.map(r => r.predicted_issue_broad)]);
  document.querySelector('#tbl-l1 tbody').innerHTML = allIss.map(i => {{
    const gt = data.filter(r => r.true_issue === i).length;
    const pr = data.filter(r => r.predicted_issue_broad === i).length;
    const d  = pr - gt;
    const col = d === 0 ? '#a78bfa' : d > 0 ? '#e879b0' : '#db2777';
    return `<tr><td>${{i}}</td><td>${{gt}}</td><td>${{pr}}</td><td style="color:${{col}};font-weight:600">${{d > 0 ? '+' : ''}}${{d}}</td></tr>`;
  }}).join('') || '<tr><td colspan="4" style="color:var(--muted);padding:10px">No data.</td></tr>';

  const allSub = unique([...data.map(r => r.true_subissue), ...data.map(r => r.predicted_subissue)]);
  document.querySelector('#tbl-l2 tbody').innerHTML = allSub.map(s => {{
    const gt = data.filter(r => r.true_subissue === s).length;
    const pr = data.filter(r => r.predicted_subissue === s).length;
    const d  = pr - gt;
    const col = d === 0 ? '#a78bfa' : d > 0 ? '#e879b0' : '#db2777';
    return `<tr><td>${{s}}</td><td>${{gt}}</td><td>${{pr}}</td><td style="color:${{col}};font-weight:600">${{d > 0 ? '+' : ''}}${{d}}</td></tr>`;
  }}).join('') || '<tr><td colspan="4" style="color:var(--muted);padding:10px">No data.</td></tr>';

  /* detail table with sort */
  const sorted = [...data].sort((a, b) => {{
    let av = sortCol === 'idx' ? a.idx : a[sortCol];
    let bv = sortCol === 'idx' ? b.idx : b[sortCol];
    if(typeof av === 'boolean') {{ av = av ? 1 : 0; bv = bv ? 1 : 0; }}
    if(typeof av === 'string')  return sortDir * av.localeCompare(bv);
    return sortDir * (av - bv);
  }});

  document.querySelector('#dtbl tbody').innerHTML = sorted.map(r => {{
    const snip   = r.complaint_text.length > 55 ? r.complaint_text.slice(0, 55) + '…' : r.complaint_text;
    const badge  = r.needs_review
      ? '<span class="badge b-review">Review</span>'
      : '<span class="badge b-auto">Auto</span>';
    const ic = r.issue_correct    ? '<span class="match-y">✓</span>' : '<span class="match-n">✗</span>';
    const sc = r.subissue_correct ? '<span class="match-y">✓</span>' : '<span class="match-n">✗</span>';
    const conf = Math.round(r.joint_confidence * 100);
    const cls  = conf >= 80 ? 'c-hi' : conf >= 60 ? 'c-mid' : 'c-lo';
    return `<tr>
      <td style="font-weight:600">${{r.idx + 1}}</td>
      <td title="${{r.complaint_text}}" style="font-size:10px">${{snip}}</td>
      <td>${{badge}}</td>
      <td style="font-size:10px">${{r.true_issue}}</td>
      <td style="font-size:10px">${{r.predicted_issue_broad}}</td>
      <td>${{ic}}</td>
      <td style="font-size:10px">${{r.true_subissue}}</td>
      <td style="font-size:10px">${{r.predicted_subissue}}</td>
      <td>${{sc}}</td>
      <td><span class="conf-chip ${{cls}}">${{conf}}%</span></td>
    </tr>`;
  }}).join('');
  document.getElementById('detail-empty').style.display = sorted.length ? 'none' : '';

  fillSelect('f-issue', unique(ALL.map(r => r.true_issue)));
  fillSelect('f-sub',   unique(ALL.map(r => r.true_subissue)));
}}

/* ── init charts ── */
const tickColor = DARK ? '#a89fc0' : '#6b7280';
const gridColor = DARK ? 'rgba(255,255,255,.06)' : 'rgba(0,0,0,.06)';

confChartInst = new Chart(document.getElementById('confChart'), {{
  type: 'bar',
  data: {{
    labels: ['<10%','10%','20%','30%','40%','50%','60%','70%','80%','90%+'],
    datasets: [{{ data: new Array(10).fill(0), backgroundColor: '#7c3aed', borderRadius: 3, borderSkipped: false }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ font: {{ size: 9 }}, color: tickColor }}, grid: {{ display: false }} }},
      y: {{ ticks: {{ font: {{ size: 9 }}, color: tickColor, stepSize: 1 }}, grid: {{ color: gridColor }} }}
    }}
  }}
}});

routeChartInst = new Chart(document.getElementById('routeChart'), {{
  type: 'doughnut',
  data: {{
    labels: ['Auto-labelled', 'Needs review'],
    datasets: [{{ data: [0, 0], backgroundColor: ['#7c3aed', '#db2777'], borderWidth: 0, hoverOffset: 3 }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false, animation: false, cutout: '65%',
    plugins: {{ legend: {{ display: false }} }}
  }}
}});

/* ── event listeners ── */
document.getElementById('f-conf').addEventListener('input', function() {{
  document.getElementById('fval').textContent = this.value + '%';
  render();
}});
['f-status','f-issue','f-sub','f-l1','f-l2'].forEach(id => {{
  document.getElementById(id).addEventListener('change', render);
}});
document.getElementById('reset-btn').addEventListener('click', () => {{
  document.getElementById('f-status').value = 'all';
  document.getElementById('f-issue').value  = 'all';
  document.getElementById('f-sub').value    = 'all';
  document.getElementById('f-l1').value     = 'all';
  document.getElementById('f-l2').value     = 'all';
  document.getElementById('f-conf').value   = 0;
  document.getElementById('fval').textContent = '0%';
  render();
}});
document.querySelectorAll('.tabs .tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
  }});
}});
document.querySelectorAll('#dtbl th[data-col]').forEach(th => {{
  th.addEventListener('click', () => {{
    const col = th.dataset.col;
    if(sortCol === col) sortDir *= -1; else {{ sortCol = col; sortDir = 1; }}
    document.querySelectorAll('.si').forEach(s => {{ s.textContent = ''; s.classList.remove('on'); }});
    const si = document.getElementById('si-' + col);
    if(si) {{ si.textContent = sortDir === 1 ? '↑' : '↓'; si.classList.add('on'); }}
    render();
  }});
}});

/* ── initial populate & render ── */
fillSelect('f-issue', {json.dumps(sorted(df['true_issue'].unique().tolist()))});
fillSelect('f-sub',   {json.dumps(sorted(df['true_subissue'].unique().tolist()))});
render();
</script>
</body>
</html>
"""
    # Height: base for overview/breakdown tabs + extra for each complaint row in detail tab
    # overview needs ~580px, detail tab needs ~60px per row header + rows
    row_height   = 38  # px per complaint row in detail table
    detail_h     = 60 + min(len(results), 25) * row_height  # capped — table scrolls internally
    overview_h   = 620  # enough for 3-card row + 2-card row + filter bar + metrics + tabs
    height       = max(overview_h, detail_h + 200)
    st.components.v1.html(html, height=height, scrolling=False)


# ══════════════════════════════════════════════════════════════════════════════
# API KEY GATE
# ══════════════════════════════════════════════════════════════════════════════
if not active_key:
    st.markdown("""
<div style="padding:1.2rem 0 1rem 0;text-align:center">
  <div style="font-size:2rem;font-weight:700;background:linear-gradient(45deg,#7c3aed,#db2777);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent">
    Student Loan Classifier AI
  </div>
  <div style="font-size:0.95rem;opacity:0.6;margin-top:4px">Hierarchical NLP Pipeline Evaluation Dashboard</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="key-modal">
  <div class="key-modal-title">🔑 Groq API Key Required</div>
  <div class="key-modal-sub">
    This app uses the <b>Groq API</b> to generate synthetic complaints.<br>
    Your key is used only for this session and is never stored.<br><br>
    Don't have a key? Get one free at
    <a href="https://console.groq.com" target="_blank" style="color:#7c3aed">console.groq.com</a>
  </div>
</div>
""", unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        entered_key = st.text_input(
            "Paste your Groq API key",
            type="password",
            placeholder="gsk_...",
            label_visibility="collapsed",
        )
        if st.button("Connect →", type="primary", use_container_width=True):
            if validate_api_key(entered_key):
                st.session_state.groq_api_key = entered_key.strip()
                st.session_state.key_validated = True
                st.rerun()
            else:
                st.error("Invalid key format. Groq keys start with **gsk_**.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# FULL APP
# ══════════════════════════════════════════════════════════════════════════════
active_key = st.session_state.groq_api_key

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Parameters")

    icon = "☀️ Light mode" if st.session_state.dark_mode else "🌙 Dark mode"
    if st.button(icon, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    # FIX: max_value changed from 100 → 25, step changed from 5 → 1
    n_complaints = st.slider("Number of complaints", min_value=10, max_value=25, value=10, step=1)
    threshold    = st.slider("Joint Confidence Threshold", 0.0, 1.0, DEFAULT_REJECTION_THRESHOLD, 0.01)

    st.write("")
    generate_clicked = st.button("Generate & Classify", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**System Status**")
    st.success("API Connected ✓")

    allowed, secs_left = check_rate_limit()
    st.caption(f"Runs this session: {st.session_state.rate_limit_count} / {RATE_LIMIT_MAX}")
    if not allowed:
        mins = secs_left // 60
        st.warning(f"⏳ Rate limit reached. Resets in ~{mins} min.")

    st.divider()
    if st.button("🔑 Change API Key", use_container_width=True):
        st.session_state.groq_api_key  = None
        st.session_state.key_validated = False
        st.rerun()

    if st.button("Reset Dashboard", use_container_width=True):
        st.session_state.results_df    = None
        st.session_state.chat_messages = []
        st.rerun()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1.2rem 0 1rem 0;text-align:center">
  <div style="font-size:2rem;font-weight:700;background:linear-gradient(45deg,#7c3aed,#db2777);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent">
    Student Loan Classifier AI
  </div>
  <div style="font-size:0.95rem;opacity:0.6;margin-top:4px">Hierarchical NLP Pipeline Evaluation Dashboard</div>
</div>
""", unsafe_allow_html=True)

# ── GENERATE & CLASSIFY ───────────────────────────────────────────────────────
if generate_clicked:
    allowed, secs_left = check_rate_limit()
    if not allowed:
        mins = secs_left // 60
        st.error(f"⏳ Rate limit reached ({RATE_LIMIT_MAX} runs/hour). Please wait ~{mins} minute(s).")
        st.stop()

    user_msg = f"Classify {n_complaints} student loan complaints — threshold {threshold:.2f}"
    st.session_state.chat_messages = [
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": "⏳ I'm working on it..."},
    ]
    st.session_state.chat_open    = True
    st.session_state.results_open = True

    with st.spinner("Running pipeline..."):
        try:
            records = generate_synthetic_complaints(
                n=n_complaints,
                api_key=active_key,
                topics=[f"Please generate {n_complaints} standard complaints regarding student loans."]
            )

            increment_rate_limit()

            if not records:
                st.session_state.chat_messages[-1] = {
                    "role": "assistant",
                    "content": "❌ Generator returned 0 records. Check Groq API response."
                }
                st.rerun()

            complaint_texts, true_issues, true_subissues = [], [], []
            for r in records:
                if isinstance(r, dict):
                    complaint_texts.append(r.get("complaint_text", "").strip())
                    true_issues.append(r.get("true_issue", "Unknown"))
                    true_subissues.append(r.get("true_subissue", "Unknown"))
                else:
                    complaint_texts.append(str(r).strip())
                    true_issues.append("Unknown")
                    true_subissues.append("Unknown")

            valid = [(t, i, s) for t, i, s in zip(complaint_texts, true_issues, true_subissues) if t]
            if not valid:
                st.session_state.chat_messages[-1] = {
                    "role": "assistant",
                    "content": "❌ All complaint texts were empty after parsing."
                }
                st.rerun()

            complaint_texts, true_issues, true_subissues = zip(*valid)
            complaint_texts = list(complaint_texts)
            true_issues     = list(true_issues)
            true_subissues  = list(true_subissues)

            clf     = load_classifier()
            results = clf.predict(complaint_texts, threshold=threshold)
            results["true_issue"]       = true_issues
            results["true_subissue"]    = true_subissues
            results["issue_correct"]    = results["predicted_issue_broad"] == results["true_issue"]
            results["subissue_correct"] = results["predicted_subissue"]    == results["true_subissue"]
            st.session_state.results_df = results

            st.session_state.chat_messages[-1] = {
                "role": "assistant",
                "content": f"✅ Done! **{len(results)}** complaints classified. Check the Results & Analysis section below 👇"
            }

        except Exception as e:
            st.session_state.chat_messages[-1] = {
                "role": "assistant",
                "content": f"❌ Pipeline failed: {e}"
            }
    st.rerun()

# ── SECTION: CHAT ─────────────────────────────────────────────────────────────
clicked_fs_chat, clicked_tog_chat = panel_header(
    "💬 Pipeline Log",
    st.session_state.chat_open,
    st.session_state.chat_fullscreen,
    key_fs="chat_fs", key_toggle="toggle_chat"
)
if clicked_fs_chat:
    st.session_state.chat_fullscreen = not st.session_state.chat_fullscreen
    st.rerun()
if clicked_tog_chat:
    st.session_state.chat_open = not st.session_state.chat_open
    st.rerun()

if st.session_state.chat_open:
    chat_height = "calc(100vh - 180px)" if st.session_state.chat_fullscreen else "420px"
    render_chat(chat_height)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── SECTION: RESULTS ──────────────────────────────────────────────────────────
clicked_fs_res, clicked_tog_res = panel_header(
    "📊 Results & Analysis",
    st.session_state.results_open,
    st.session_state.results_fullscreen,
    key_fs="res_fs", key_toggle="toggle_results"
)
if clicked_fs_res:
    st.session_state.results_fullscreen = not st.session_state.results_fullscreen
    st.rerun()
if clicked_tog_res:
    st.session_state.results_open = not st.session_state.results_open
    st.rerun()

if st.session_state.results_open and st.session_state.results_df is not None:
    results = st.session_state.results_df

    render_dashboard(results, st.session_state.dark_mode)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name="classifier_evaluation_results.csv",
        mime="text/csv",
        type="primary",
    )

elif st.session_state.results_open and st.session_state.results_df is None:
    st.markdown(f"""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;
            padding:2.5rem;text-align:center;color:var(--muted);
            display:flex;align-items:center;justify-content:center;flex-direction:column;gap:8px">
  <div style="font-size:2rem">📭</div>
  <div style="font-size:0.95rem">No results yet. Generate complaints to see the analysis here.</div>
</div>
""", unsafe_allow_html=True)