import os
import time
import re
import json
import base64
import streamlit as st
import pandas as pd

from complaint_generator import generate_synthetic_complaints
from model_pipeline import HierarchicalComplaintClassifier, DEFAULT_REJECTION_THRESHOLD
from human_review_section import render_review_queue

st.set_page_config(page_title="ComplaintIQ", layout="wide")

# ── SESSION STATE DEFAULTS ────────────────────────────────────────────────────
defaults = {
    "dark_mode":           True,
    "results_df":          None,
    "chat_messages":       [],
    "results_open":        True,
    "chat_open":           True,
    "chat_fullscreen":     False,
    "results_fullscreen":  False,
    "groq_api_key":        None,
    "rate_limit_count":    0,
    "rate_limit_ts":       0.0,
    "key_validated":       False,
    "review_decisions":    {},
    "review_finalised":    False,
    "results_with_review": None,
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

# ── THEME (DARK MODE ONLY) ────────────────────────────────────────────────────
DARK_BG       = "#0f172a"
DARK_SURFACE  = "#1e293b"

PRIMARY       = "#6366f1"
PRIMARY_HOVER = "#4f46e5"
ACCENT        = "#10b981"
DANGER        = "#ef4444"

bg      = DARK_BG
surface = DARK_SURFACE
text    = "#f1f5f9"
muted   = "#94a3b8"
border  = "rgba(255,255,255,0.08)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
    --primary: {PRIMARY};
    --primary-hover: {PRIMARY_HOVER};
    --bg: {bg};
    --surface: {surface};
    --text: {text};
    --muted: {muted};
    --border: {border};
    --font: 'Inter', sans-serif;
}}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
.main, .block-container {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
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
    font-family: var(--font) !important;
}}

[data-testid="stSidebar"] h2 {{
    color: var(--text) !important;
    font-weight: 700;
}}

div.stButton > button[kind="primary"] {{
    background-color: var(--primary) !important;
    color: white !important; border: none !important; font-weight: 600;
}}
div.stButton > button[kind="primary"]:hover {{
    background-color: var(--primary-hover) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
}}
div.stButton > button {{
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    transition: all 0.2s ease;
}}
div.stButton > button:hover {{
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}}

div[data-baseweb="slider"] [role="slider"] {{
    background: var(--primary) !important;
    border: 2px solid {bg} !important;
}}
div[data-baseweb="slider"] [class*="Track"]:not([class*="Fill"]) {{
    background: var(--border) !important;
}}

div[data-testid="stAlert"] {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-left: 4px solid var(--primary) !important;
    border-radius: 8px !important;
}}
div[data-testid="stAlert"] * {{ color: var(--text) !important; }}
div[data-testid="stAlert"] svg {{ display: none !important; }}

.chat-container {{
    display: flex; flex-direction: column; gap: 8px;
    padding: 1rem; background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px;
}}
.chat-bubble {{ display: flex; flex-direction: column; max-width: 90%; margin-bottom: 8px; }}
.chat-bubble.user {{ align-self: flex-end; align-items: flex-end; }}
.chat-bubble.assistant {{ align-self: flex-start; align-items: flex-start; }}
.chat-bubble .bubble-inner {{
    padding: 10px 16px; border-radius: 8px; font-size: 0.85rem; line-height: 1.5;
}}
.chat-bubble.user .bubble-inner {{
    background: var(--primary);
    color: white !important;
}}
.chat-bubble.assistant .bubble-inner {{
    background: var(--bg); border: 1px solid var(--border);
    color: var(--text) !important;
}}
.chat-bubble .bubble-label {{ font-size: 0.7rem; color: var(--muted); margin-bottom: 4px; font-weight: 600; text-transform: uppercase; }}

.panel-header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 18px; background: var(--surface);
    border: 1px solid var(--border); border-radius: 8px;
    margin-bottom: 0 !important;
}}
.panel-header-title {{
    font-weight: 600; font-size: 0.95rem; color: var(--text);
}}

/* Collapse Streamlit default gap between consecutive markdown blocks and iframes */
[data-testid="stVerticalBlock"] > [data-testid="element-container"] + [data-testid="element-container"],
[data-testid="stVerticalBlock"] > [data-testid="element-container"] + [data-testid="stIframe"],
[data-testid="stVerticalBlock"] > [data-testid="stIframe"] + [data-testid="element-container"] {{
    margin-top: -1rem !important;
}}
/* Gap between pipeline log panel header and results panel header = same as chart-row gap (16px) */
[data-testid="stVerticalBlock"] > [data-testid="stIframe"] + [data-testid="element-container"]:has(.panel-header) {{
    margin-top: 0 !important;
}}

.key-modal {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 2.5rem 2rem;
    max-width: 480px; margin: 4rem auto; text-align: center;
}}
.key-modal-title {{
    font-size: 1.4rem; font-weight: 700; color: var(--text); margin-bottom: 0.5rem;
}}
.key-modal-sub {{ font-size: 0.9rem; color: var(--muted); margin-bottom: 1.5rem; line-height: 1.5; }}

/* Tighten vertical gaps between Streamlit block elements */
[data-testid="stVerticalBlock"] > [data-testid="element-container"] {{
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
    margin-top: 0 !important;
    padding-top: 0 !important;
}}
[data-testid="stVerticalBlock"] > [data-testid="element-container"] + [data-testid="element-container"] {{
    margin-top: 8px !important;
}}
/* Collapse gap between panel header and its content (iframe or markdown) */
[data-testid="stVerticalBlock"] > [data-testid="element-container"]:has(.panel-header) + * {{
    margin-top: 4px !important;
}}
/* iframe containers - flush against their panel header */
[data-testid="stCustomComponentV1"] {{
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}}
/* The element after an iframe (Human Review Queue panel header) - same 8px as between other elements */
[data-testid="stVerticalBlock"] > [data-testid="stCustomComponentV1"] + [data-testid="element-container"] {{
    margin-top: 8px !important;
}}
/* Normalize all children of stVerticalBlock that Streamlit wraps in stVerticalBlockBorderWrapper */
[data-testid="stVerticalBlockBorderWrapper"] {{
    padding: 0 !important;
    margin: 0 !important;
}}

hr {{ border-color: var(--border) !important; }}
div[data-testid="stSpinner"] > div {{ border-top-color: var(--primary) !important; }}
a[data-testid="stDownloadButton"] button {{
    background: var(--primary) !important; color: white !important; border: none !important;
}}
.stCaption, [data-testid="stCaptionContainer"] * {{ color: var(--muted) !important; }}
div[data-testid="stTextInput"] input {{
    background: var(--bg) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 6px !important;
}}
div[data-testid="stTextInput"] input:focus {{
    border-color: var(--primary) !important;
}}

div[data-baseweb="select"] > div,
div[data-baseweb="select"] ul {{
    background-color: var(--surface) !important;
    border-color: var(--border) !important;
}}
div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {{
    color: var(--text) !important;
}}
div[role="listbox"], div[role="listbox"] ul {{
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
}}
div[role="option"], div[role="option"] * {{
    background-color: var(--surface) !important;
    color: var(--text) !important;
}}
div[role="option"]:hover, div[role="option"]:hover * {{
    background-color: var(--primary) !important;
    color: white !important;
}}

/* Hide broken Material icon text, replace with CSS pseudo-element on button */
[data-testid="stIconMaterial"] {{
    font-size: 0 !important;
    color: transparent !important;
    width: 0 !important;
    overflow: hidden !important;
}}
/* Sidebar toggle button - works both open and closed */
[data-testid="stBaseButton-headerNoPadding"] {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    width: 2rem !important;
    height: 2rem !important;
}}
[data-testid="stIconMaterial"] {{
    font-size: 0 !important;
    color: transparent !important;
    width: 0 !important;
    overflow: hidden !important;
}}
[data-testid="stSidebarCollapseButton"] [data-testid="stBaseButton-headerNoPadding"]::before {{
    content: "«";
    font-size: 18px;
    color: {text};
    font-family: 'Inter', sans-serif;
    line-height: 1;
}}
[data-testid="stSidebarCollapsedControl"] [data-testid="stBaseButton-headerNoPadding"]::before {{
    content: "»";
    font-size: 18px;
    color: {text};
    font-family: 'Inter', sans-serif;
    line-height: 1;
}}

li[role="option"] {{
    background-color: var(--surface) !important;
    color: var(--text) !important;
}}
</style>
""", unsafe_allow_html=True)

st.components.v1.html(f"""
<script>
function fixSliders() {{
    var p = window.parent.document;
    p.querySelectorAll('[data-baseweb="slider"]').forEach(slider => {{
        slider.querySelectorAll('[role="slider"]').forEach(t => {{
            t.style.background = '{PRIMARY}';
            t.style.borderColor = '{bg}';
            t.style.boxShadow = 'none';
        }});
        slider.querySelectorAll('[class*="TrackFill"],[class*="trackFill"]').forEach(f => {{
            f.style.background = '{PRIMARY}';
        }});
        slider.querySelectorAll('[class*="Track"]:not([class*="Fill"])').forEach(t => {{
            t.style.background = '{border}';
        }});
    }});
}}

function openSidebar() {{
    try {{
        var p = window.parent.document;
        var sidebar = p.querySelector('[data-testid="stSidebar"]');
        var btn = p.querySelector('[data-testid="stSidebarCollapseButton"] button');
        if (sidebar && btn) {{
            if (sidebar.getAttribute('aria-expanded') === 'false') btn.click();
        }} else {{
            setTimeout(openSidebar, 300);
        }}
    }} catch(e) {{}}
}}

fixSliders();
openSidebar();
new MutationObserver(fixSliders).observe(window.parent.document.body, {{childList: true, subtree: true, attributes: true}});
</script>
""", height=0)

@st.cache_resource
def load_classifier():
    return HierarchicalComplaintClassifier()

def render_chat(height="420px"):
    if not st.session_state.chat_messages:
        st.markdown(f"""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;
            padding:2.5rem;text-align:center;color:var(--muted);height:180px;
            display:flex;align-items:center;justify-content:center;flex-direction:column;gap:8px">
  <div style="font-size:0.9rem; font-weight:500;">System Idle</div>
  <div style="font-size:0.8rem; opacity:0.7;">Hit <b>Generate & Classify</b> to start the pipeline.</div>
</div>
""", unsafe_allow_html=True)
    else:
        bubbles = ""
        for msg in st.session_state.chat_messages:
            role    = msg["role"]
            content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', msg["content"])
            content = content.replace("\n", "<br>")
            label   = "COMMAND" if role == "user" else "PIPELINE"
            bubbles += f"""
<div class="chat-bubble {role}">
  <div class="bubble-label">{label}</div>
  <div class="bubble-inner">{content}</div>
</div>"""
        st.markdown(f"""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;
            padding:1rem 1.2rem;overflow-y:auto;height:{height}">
  <div class="chat-container">{bubbles}</div>
</div>
""", unsafe_allow_html=True)

def panel_header(title):
    st.markdown(f"""
<div class="panel-header">
  <span class="panel-header-title">{title}</span>
</div>
""", unsafe_allow_html=True)


def render_dashboard(results: pd.DataFrame, dark_mode: bool):
    df = results.copy()

    # Ensure all needed columns exist with fallbacks
    if "needs_review" not in df.columns:
        df["needs_review"] = False
    if "joint_confidence" not in df.columns:
        df["joint_confidence"] = 0.0
    if "predicted_issue_broad" not in df.columns:
        df["predicted_issue_broad"] = df.get("predicted_issue", "Unknown")
    if "predicted_subissue" not in df.columns:
        df["predicted_subissue"] = df.get("predicted_sub_issue", "Unknown")
    if "true_issue" not in df.columns:
        df["true_issue"] = "Unknown"
    if "true_subissue" not in df.columns:
        df["true_subissue"] = "Unknown"
    if "issue_correct" not in df.columns:
        df["issue_correct"] = df["predicted_issue_broad"] == df["true_issue"]
    if "subissue_correct" not in df.columns:
        df["subissue_correct"] = df["predicted_subissue"] == df["true_subissue"]
    if "review_source" not in df.columns:
        df["review_source"] = "model"
    if "complaint_text" not in df.columns:
        df["complaint_text"] = ""

    cols_needed = [
        "complaint_text", "true_issue", "predicted_issue_broad",
        "issue_correct", "true_subissue", "predicted_subissue",
        "subissue_correct", "joint_confidence", "needs_review", "review_source",
    ]
    df = df[cols_needed].copy()
    df["issue_correct"]    = df["issue_correct"].fillna(False).astype(bool)
    df["subissue_correct"] = df["subissue_correct"].fillna(False).astype(bool)
    df["needs_review"]     = df["needs_review"].fillna(False).astype(bool)
    records_json = df.to_json(orient="records")
    records_b64  = base64.b64encode(records_json.encode("utf-8")).decode("ascii")

    theme = {
        "bg":      DARK_BG,
        "surface": DARK_SURFACE,
        "text":    "#f1f5f9",
        "muted":   "#94a3b8",
        "border":  "rgba(255,255,255,0.08)",
        "primary": PRIMARY,
        "accent":  ACCENT,
        "danger":  DANGER,
        "isDark":  "true",
    }

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:'Inter',sans-serif}}
:root{{
  --bg:{theme['bg']};--surface:{theme['surface']};--text:{theme['text']};
  --muted:{theme['muted']};--border:{theme['border']};--primary:{theme['primary']};
  --accent:{theme['accent']};--danger:{theme['danger']};
}}
body{{background:var(--bg);color:var(--text);padding:12px 4px 8px}}

.metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px}}
.metric{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px}}
.metric-label{{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px}}
.metric-value{{font-size:28px;font-weight:700;color:var(--text)}}
.metric-sub{{font-size:12px;color:var(--muted);margin-top:4px}}

.tabs{{display:flex;gap:4px;margin-bottom:12px;border-bottom:1px solid var(--border);padding-bottom:1px}}
.tab{{
  padding:8px 16px;font-size:14px;font-weight:500;color:var(--muted);
  cursor:pointer;border-bottom:2px solid transparent;transition:all .2s
}}
.tab:hover{{color:var(--text)}}
.tab.active{{color:var(--primary);border-bottom-color:var(--primary)}}
.tab-panel{{display:none}}.tab-panel.active{{display:block}}

.card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:20px;margin-bottom:16px}}
.card-title{{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:16px}}

.bar-row{{display:flex;align-items:center;gap:10px;margin-bottom:10px}}
.bar-lbl{{font-size:13px;width:140px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.bar-track{{flex:1;height:7px;background:var(--bg);border-radius:3px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:3px;transition:width .3s}}
.bar-val{{font-size:13px;font-weight:600;width:38px;text-align:right}}

.btbl{{width:100%;border-collapse:collapse;font-size:13px}}
.btbl th{{text-align:left;padding:9px;color:var(--muted);border-bottom:1px solid var(--border)}}
.btbl td{{padding:9px;border-bottom:1px solid var(--border)}}

.dtbl-wrap{{overflow-x:auto;background:var(--surface);border:1px solid var(--border);border-radius:8px}}
.dtbl{{width:100%;border-collapse:collapse;font-size:13px;min-width:1000px}}
.dtbl th{{
  text-align:left;padding:12px 8px;color:var(--muted);background:var(--surface);
  border-bottom:1px solid var(--border);position:sticky;top:0;cursor:pointer;font-weight:600
}}
.dtbl th:hover{{color:var(--text)}}
.dtbl td{{padding:10px 8px;border-bottom:1px solid var(--border);vertical-align:top}}
.dtbl tr:hover{{background:rgba(255,255,255,0.02)}}

.badge{{padding:3px 9px;border-radius:4px;font-size:12px;font-weight:600}}
.b-auto{{background:rgba(16,185,129,0.1);color:var(--accent)}}
.b-review{{background:rgba(239,68,68,0.1);color:var(--danger)}}
.b-human{{background:rgba(99,102,241,0.15);color:var(--primary)}}
.match-y{{color:var(--accent);font-weight:700}}.match-n{{color:var(--danger);font-weight:700}}
.conf-chip{{padding:3px 7px;border-radius:4px;font-weight:600;font-size:12px}}
.c-hi{{background:rgba(16,185,129,0.15);color:var(--accent)}}
.c-mid{{background:rgba(99,102,241,0.15);color:var(--primary)}}
.c-lo{{background:rgba(239,68,68,0.15);color:var(--danger)}}
.si{{font-size:11px;margin-left:4px;opacity:0.3}}.si.on{{opacity:1;color:var(--primary)}}
</style>
</head>
<body>
  <div class="metrics">
    <div class="metric"><div class="metric-label">Total</div><div class="metric-value" id="m-total">0</div><div class="metric-sub" id="m-total-sub"></div></div>
    <div class="metric"><div class="metric-label">Auto-Labelled</div><div class="metric-value" id="m-auto">0</div><div class="metric-sub" id="m-auto-pct"></div></div>
    <div class="metric"><div class="metric-label">Needs Review</div><div class="metric-value" id="m-review">0</div><div class="metric-sub" id="m-review-pct"></div></div>
    <div class="metric"><div class="metric-label">L1 Accuracy</div><div class="metric-value" id="m-l1">0%</div><div class="metric-sub">all complaints</div></div>
    <div class="metric"><div class="metric-label">L2 Accuracy</div><div class="metric-value" id="m-l2">0%</div><div class="metric-sub">all complaints</div></div>
  </div>

  <div class="tabs">
    <div class="tab active" data-tab="overview">Overview</div>
    <div class="tab" data-tab="breakdown">Category breakdown</div>
    <div class="tab" data-tab="detail">Per-complaint detail</div>
  </div>

  <div id="tab-overview" class="tab-panel active">
    <div style="display:grid;grid-template-columns:1.2fr 1fr 1fr;gap:16px;margin-bottom:16px">
      <div class="card"><div class="card-title">Confidence Distribution</div><div class="chart-wrap" style="height:160px"><canvas id="confChart"></canvas></div></div>
      <div class="card"><div class="card-title">Overall Accuracy</div><div id="acc-bars" style="margin-top:10px"></div></div>
      <div class="card"><div class="card-title">Routing Split</div><div class="chart-wrap" style="height:160px"><canvas id="routeChart"></canvas></div></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
      <div class="card"><div class="card-title">L1 accuracy by broad issue</div><div id="l1-cat"></div></div>
      <div class="card"><div class="card-title">L2 accuracy by sub-issue</div><div id="l2-cat"></div></div>
    </div>
  </div>

  <div id="tab-breakdown" class="tab-panel">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
      <div class="card"><div class="card-title">Broad issue — truth vs predicted</div><table class="btbl" id="tbl-l1"><thead><tr><th>Category</th><th>Truth</th><th>Predicted</th><th>Diff</th></tr></thead><tbody></tbody></table></div>
      <div class="card"><div class="card-title">Sub-issue — truth vs predicted</div><table class="btbl" id="tbl-l2"><thead><tr><th>Category</th><th>Truth</th><th>Predicted</th><th>Diff</th></tr></thead><tbody></tbody></table></div>
    </div>
  </div>

  <div id="tab-detail" class="tab-panel">
    <div class="dtbl-wrap" style="margin-top:4px">
      <table class="dtbl" id="dtbl">
        <thead>
          <tr><th style="width:32px" data-col="idx"># <span class="si" id="si-idx"></span></th><th style="width:200px">Complaint</th><th style="width:60px">Status</th><th data-col="true_issue">True issue <span class="si" id="si-true_issue"></span></th><th data-col="predicted_issue_broad">Pred issue <span class="si" id="si-predicted_issue_broad"></span></th><th style="width:32px" data-col="issue_correct">L1 <span class="si" id="si-issue_correct"></span></th><th data-col="true_subissue">True sub-issue <span class="si" id="si-true_subissue"></span></th><th data-col="predicted_subissue">Pred sub-issue <span class="si" id="si-predicted_subissue"></span></th><th style="width:32px" data-col="subissue_correct">L2 <span class="si" id="si-subissue_correct"></span></th><th style="width:62px" data-col="joint_confidence">Conf <span class="si" id="si-joint_confidence"></span></th></tr>
        </thead>
        <tbody></tbody>
      </table>
      <div class="empty" id="detail-empty" style="display:none;padding:40px;text-align:center;color:var(--muted)">No complaints match the current filters.</div>
    </div>
  </div>

  <div style="height:8px"></div>

<script>
const ALL = JSON.parse(atob('{records_b64}'));
const TOTAL = ALL.length;
let sortCol = 'idx', sortDir = 1;
let confChartInst, routeChartInst;

function unique(arr){{ return [...new Set(arr)].sort(); }}

function barRow(label, val, color){{
  const pct = isNaN(val) ? 0 : Math.round(val * 100);
  const short = label.length > 25 ? label.slice(0,25) + '\u2026' : label;
  return `<div class="bar-row">
    <div class="bar-lbl" title="${{label}}">${{short}}</div>
    <div class="bar-track"><div class="bar-fill" style="width:${{pct}}%;background:${{color}}"></div></div>
    <div class="bar-val">${{pct}}%</div>
  </div>`;
}}

function render(){{
  const data = ALL.map((r, i) => ({{ ...r, idx: i }}));
  const autoR = data.filter(r => !r.needs_review);
  const revR  = data.filter(r =>  r.needs_review);
  // L1/L2 accuracy is computed over ALL complaints (raw model performance),
  // not just the auto-labelled subset — rows are routed to human review
  // precisely when the model got L1 wrong, so restricting to autoR would
  // always show 100% L1 accuracy regardless of real performance.
  const l1Acc = data.length ? data.filter(r => r.issue_correct).length / data.length : 0;
  const l2Acc = data.length ? data.filter(r => r.subissue_correct).length / data.length : 0;

  document.getElementById('m-total').textContent = data.length;
  document.getElementById('m-total-sub').textContent = `${{TOTAL}} complaints`;
  document.getElementById('m-auto').textContent = autoR.length;
  document.getElementById('m-auto-pct').textContent = data.length ? Math.round(autoR.length/data.length*100) + '% of total' : '--';
  document.getElementById('m-review').textContent = revR.length;
  document.getElementById('m-review-pct').textContent = data.length ? Math.round(revR.length/data.length*100) + '% of total' : '--';
  document.getElementById('m-l1').textContent = data.length ? Math.round(l1Acc*100) + '%' : '--';
  document.getElementById('m-l2').textContent = data.length ? Math.round(l2Acc*100) + '%' : '--';

  if (!data.length) {{
    const ph = '<div style="height:60px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:11px">No data yet</div>';
    document.getElementById('acc-bars').innerHTML = ph;
    document.getElementById('l1-cat').innerHTML = ph;
    document.getElementById('l2-cat').innerHTML = ph;
    document.getElementById('confChart').style.display = 'none';
    document.getElementById('routeChart').style.display = 'none';
    document.querySelectorAll('.chart-wrap').forEach(el => el.style.height = '60px');
    return;
  }}

  document.getElementById('acc-bars').innerHTML = 
    barRow('L1 correct', l1Acc, 'var(--primary)') + 
    barRow('L1 incorrect', 1 - l1Acc, 'var(--danger)') +
    barRow('L2 correct', l2Acc, 'var(--primary)') +
    barRow('L2 incorrect', 1 - l2Acc, 'var(--danger)');

  const bins = new Array(10).fill(0);
  data.forEach(r => {{ const b = Math.min(9, Math.floor(r.joint_confidence * 10)); bins[b]++; }});
  confChartInst.data.datasets[0].data = bins;
  confChartInst.update('none');

  routeChartInst.data.datasets[0].data = [autoR.length, revR.length];
  routeChartInst.update('none');

  const issues = unique(data.map(r => r.true_issue));
  document.getElementById('l1-cat').innerHTML = issues.map(iss => {{
    const rows = data.filter(r => r.true_issue === iss);
    return barRow(iss, rows.length ? rows.filter(r => r.issue_correct).length/rows.length : 0, 'var(--primary)');
  }}).join('') || '<div style="font-size:11px;color:var(--muted);padding:8px 0">No data.</div>';

  const subs = unique(data.map(r => r.true_subissue));
  document.getElementById('l2-cat').innerHTML = subs.map(s => {{
    const rows = data.filter(r => r.true_subissue === s);
    return barRow(s, rows.length ? rows.filter(r => r.subissue_correct).length/rows.length : 0, 'var(--primary)');
  }}).join('') || '<div style="font-size:11px;color:var(--muted);padding:8px 0">No data.</div>';

  const allIss = unique([...data.map(r => r.true_issue), ...data.map(r => r.predicted_issue_broad)]);
  document.querySelector('#tbl-l1 tbody').innerHTML = allIss.map(i => {{
    const gt = data.filter(r => r.true_issue === i).length;
    const pr = data.filter(r => r.predicted_issue_broad === i).length;
    const d  = pr - gt;
    const col = d === 0 ? 'var(--muted)' : d > 0 ? 'var(--accent)' : 'var(--danger)';
    return `<tr><td>${{i}}</td><td>${{gt}}</td><td>${{pr}}</td><td style="color:${{col}};font-weight:600">${{d > 0 ? '+' : ''}}${{d}}</td></tr>`;
  }}).join('') || '<tr><td colspan="4" style="color:var(--muted);padding:10px">No data.</td></tr>';

  const allSub = unique([...data.map(r => r.true_subissue), ...data.map(r => r.predicted_subissue)]);
  document.querySelector('#tbl-l2 tbody').innerHTML = allSub.map(s => {{
    const gt = data.filter(r => r.true_subissue === s).length;
    const pr = data.filter(r => r.predicted_subissue === s).length;
    const d  = pr - gt;
    const col = d === 0 ? 'var(--muted)' : d > 0 ? 'var(--accent)' : 'var(--danger)';
    return `<tr><td>${{s}}</td><td>${{gt}}</td><td>${{pr}}</td><td style="color:${{col}};font-weight:600">${{d > 0 ? '+' : ''}}${{d}}</td></tr>`;
  }}).join('') || '<tr><td colspan="4" style="color:var(--muted);padding:10px">No data.</td></tr>';

  const sorted = [...data].sort((a, b) => {{
    let av = sortCol === 'idx' ? a.idx : a[sortCol];
    let bv = sortCol === 'idx' ? b.idx : b[sortCol];
    if(typeof av === 'boolean') {{ av = av ? 1 : 0; bv = bv ? 1 : 0; }}
    if(typeof av === 'string')  return sortDir * av.localeCompare(bv);
    return sortDir * (av - bv);
  }});

  document.querySelector('#dtbl tbody').innerHTML = sorted.map(r => {{
    const snip = r.complaint_text.length > 55 ? r.complaint_text.slice(0, 55) + '\u2026' : r.complaint_text;
    const badge = r.review_source === 'human'
      ? '<span class="badge b-human">Human</span>'
      : (r.needs_review ? '<span class="badge b-review">Review</span>' : '<span class="badge b-auto">Auto</span>');
    const ic = r.issue_correct ? '<span class="match-y">✓</span>' : '<span class="match-n">✗</span>';
    const sc = r.subissue_correct ? '<span class="match-y">✓</span>' : '<span class="match-n">✗</span>';
    const conf = Math.round(r.joint_confidence * 100);
    const cls = conf >= 80 ? 'c-hi' : conf >= 60 ? 'c-mid' : 'c-lo';
    return `<tr>
      <td style="font-weight:600">${{r.idx + 1}}</td>
      <td title="${{r.complaint_text}}" style="font-size:12px">${{snip}}</td>
      <td>${{badge}}</td>
      <td style="font-size:12px">${{r.true_issue}}</td>
      <td style="font-size:12px">${{r.predicted_issue_broad}}</td>
      <td>${{ic}}</td>
      <td style="font-size:12px">${{r.true_subissue}}</td>
      <td style="font-size:12px">${{r.predicted_subissue}}</td>
      <td>${{sc}}</td>
      <td><span class="conf-chip ${{cls}}">${{conf}}%</span></td>
    </tr>`;
  }}).join('');
  document.getElementById('detail-empty').style.display = sorted.length ? 'none' : '';
}}

const tickColor = '#94a3b8';
const gridColor = 'rgba(255,255,255,.04)';

confChartInst = new Chart(document.getElementById('confChart'), {{
  type: 'bar',
  data: {{
    labels: ['<10%','10%','20%','30%','40%','50%','60%','70%','80%','90%+'],
    datasets: [{{ data: new Array(10).fill(0), backgroundColor: '{PRIMARY}', borderRadius: 4 }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ font: {{ size: 11 }}, color: tickColor }}, grid: {{ display: false }} }},
      y: {{ ticks: {{ font: {{ size: 11 }}, color: tickColor, stepSize: 1 }}, grid: {{ color: gridColor }} }}
    }}
  }}
}});

routeChartInst = new Chart(document.getElementById('routeChart'), {{
  type: 'doughnut',
  data: {{
    labels: ['Auto-labelled', 'Needs review'],
    datasets: [{{ data: [0, 0], backgroundColor: ['{PRIMARY}', '{DANGER}'], borderWidth: 0, hoverOffset: 3 }}]
  }},
  options: {{ responsive: true, maintainAspectRatio: false, animation: false, cutout: '70%', plugins: {{ legend: {{ display: false }} }} }}
}});

document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {{
  document.querySelectorAll('.tab, .tab-panel').forEach(el => el.classList.remove('active'));
  tab.classList.add('active'); document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
}}));
document.querySelectorAll('#dtbl th[data-col]').forEach(th => th.addEventListener('click', () => {{
  const col = th.dataset.col;
  if(sortCol === col) sortDir *= -1; else {{ sortCol = col; sortDir = 1; }}
  document.querySelectorAll('.si').forEach(s => s.textContent = '');
  const si = document.getElementById('si-' + col);
  if(si) {{ si.textContent = sortDir === 1 ? '\u2191' : '\u2193'; si.classList.add('on'); }}
  render();
}}));

render();
</script>
</body>
</html>
"""
    st.components.v1.html(html, height=720, scrolling=True)


# ── SVG BADGE (themed, inline) ────────────────────────────────────────────────
def _badge_svg() -> str:
    return (
        f'<svg width="360" height="100" viewBox="0 0 360 100" xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="1" y="1" width="358" height="98" rx="18" fill="{surface}" stroke="{border}" stroke-width="1.5"/>'
        f'<circle cx="50" cy="50" r="28" fill="{bg}"/>'
        f'<path d="M 36 50 L 45 59 L 64 38" fill="none" stroke="{PRIMARY}" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="64" cy="38" r="4.5" fill="{ACCENT}" stroke="{bg}" stroke-width="1.5"/>'
        f'<line x1="94" y1="22" x2="94" y2="78" stroke="{border}" stroke-width="1"/>'
        f'<text x="112" y="48" font-family="Georgia,serif" font-size="25" font-weight="700" letter-spacing="0.3" fill="{text}">ComplaintIQ</text>'
        f'<text x="112" y="68" font-family="Arial,sans-serif" font-size="11" font-weight="600" letter-spacing="1.6" fill="{muted}">AI-POWERED &#183; HUMAN-BACKED</text>'
        f'</svg>'
    )

def _badge_html() -> str:
    b64 = base64.b64encode(_badge_svg().encode("utf-8")).decode("ascii")
    return f'<div style="text-align:center;padding:1.2rem 0 1rem 0"><img src="data:image/svg+xml;base64,{b64}" style="max-width:360px;width:100%;height:auto"/></div>'

# ── API KEY GATE ──────────────────────────────────────────────────────────────
if not active_key:
    st.markdown(_badge_html(), unsafe_allow_html=True)

    st.markdown(f"""
<div class="key-modal">
  <div class="key-modal-title">Groq API Key Required</div>
  <div class="key-modal-sub">
    This app uses the <b>Groq API</b> to generate synthetic complaints.<br>
    Your key is used only for this session and is never stored.<br><br>
    Don't have a key? Get one free at
    <a href="https://console.groq.com" target="_blank" style="color:{PRIMARY}">console.groq.com</a>
  </div>
</div>
""", unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        entered_key = st.text_input("Paste your Groq API key", type="password", placeholder="gsk_...", label_visibility="collapsed")
        if st.button("Connect", type="primary", use_container_width=True):
            if validate_api_key(entered_key):
                st.session_state.groq_api_key = entered_key.strip()
                st.session_state.key_validated = True
                st.rerun()
            else:
                st.error("Invalid key format. Groq keys start with **gsk_**.")
    st.stop()

# ── FULL APP ──────────────────────────────────────────────────────────────────
active_key = st.session_state.groq_api_key

with st.sidebar:
    st.header("Parameters")

    n_complaints = st.slider("Number of complaints", min_value=10, max_value=25, value=10, step=1)
    threshold    = st.slider("Joint Confidence Threshold", 0.0, 1.0, DEFAULT_REJECTION_THRESHOLD, 0.01)

    st.write("")
    generate_clicked = st.button("Generate & Classify", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**System Status**")
    st.success("API Connected")

    allowed, secs_left = check_rate_limit()
    st.caption(f"Runs this session: {st.session_state.rate_limit_count} / {RATE_LIMIT_MAX}")
    if not allowed:
        mins = secs_left // 60
        st.warning(f"Rate limit reached. Resets in ~{mins} min.")

    st.divider()
    if st.button("Change API Key", use_container_width=True):
        st.session_state.groq_api_key  = None
        st.session_state.key_validated = False
        st.rerun()

    if st.button("Reset Dashboard", use_container_width=True):
        st.session_state.results_df    = None
        st.session_state.chat_messages = []
        st.rerun()

    if st.session_state.results_df is not None:
        csv = st.session_state.results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="classifier_results.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

st.markdown(_badge_html(), unsafe_allow_html=True)

if generate_clicked:
    allowed, secs_left = check_rate_limit()
    if not allowed:
        mins = secs_left // 60
        st.error(f"Rate limit reached. Please wait ~{mins} minute(s).")
        st.stop()

    user_msg = f"Classify {n_complaints} student loan complaints — threshold {threshold:.2f}"
    st.session_state.chat_messages = [{"role": "user", "content": user_msg}, {"role": "assistant", "content": "Processing pipeline..."}]
    st.session_state.review_decisions = {}; st.session_state.review_finalised = False; st.session_state.results_with_review = None
    st.session_state.chat_open = True; st.session_state.results_open = True

    with st.spinner("Running pipeline..."):
        try:
            records = generate_synthetic_complaints(n=n_complaints, api_key=active_key, topics=[f"Please generate {n_complaints} standard complaints regarding student loans."])
            increment_rate_limit()
            if not records:
                st.session_state.chat_messages[-1] = {"role": "assistant", "content": "Generator returned 0 records."}; st.rerun()
            
            complaint_texts, true_issues, true_subissues = [], [], []
            for r in records:
                if isinstance(r, dict):
                    complaint_texts.append(r.get("complaint_text", "").strip())
                    true_issues.append(r.get("true_issue", "Unknown"))
                    true_subissues.append(r.get("true_subissue", "Unknown"))
                else:
                    complaint_texts.append(str(r).strip()); true_issues.append("Unknown"); true_subissues.append("Unknown")

            valid = [(t, i, s) for t, i, s in zip(complaint_texts, true_issues, true_subissues) if t]
            if not valid:
                st.session_state.chat_messages[-1] = {"role": "assistant", "content": "All complaint texts were empty."}; st.rerun()

            complaint_texts, true_issues, true_subissues = zip(*valid)
            clf = load_classifier()
            results = clf.predict(list(complaint_texts), threshold=threshold)
            results["true_issue"] = list(true_issues); results["true_subissue"] = list(true_subissues)
            results["issue_correct"] = results["predicted_issue_broad"] == results["true_issue"]
            results["subissue_correct"] = results["predicted_subissue"] == results["true_subissue"]
            # If the model got the broad issue (L1) wrong, route it straight to human
            # review — regardless of how confident it was.
            if "needs_review" not in results.columns:
                results["needs_review"] = False
            results["needs_review"] = results["needs_review"] | (~results["issue_correct"])
            st.session_state.results_df = results
            st.session_state.chat_messages[-1] = {"role": "assistant", "content": f"Done! **{len(results)}** complaints classified. Check the results below."}
        except Exception as e:
            st.session_state.chat_messages[-1] = {"role": "assistant", "content": f"Pipeline failed: {e}"}
    st.rerun()

panel_header("Pipeline Log")
if st.session_state.chat_open:
    render_chat("200px")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
panel_header("Results & Analysis")

if st.session_state.results_open and st.session_state.results_df is not None:
    results = st.session_state.results_df

    # Once the human review is finalised, the dashboard list refreshes to show
    # the human-corrected issue/sub-issue values instead of the model's raw
    # guess for those rows (the "Human" badge marks which ones changed).
    if st.session_state.review_finalised and st.session_state.results_with_review is not None:
        display_results = st.session_state.results_with_review.copy()
        display_results["predicted_issue_broad"] = display_results["reviewed_issue"]
        display_results["predicted_subissue"]    = display_results["reviewed_subissue"]
    else:
        display_results = results

    render_dashboard(display_results, st.session_state.dark_mode)
    panel_header("Human Review Queue")
    if st.session_state.get("review_open", True): render_review_queue(results, st.session_state.dark_mode)
elif st.session_state.results_open:
    st.markdown("""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;
            padding:2.5rem;text-align:center;color:var(--muted);
            display:flex;align-items:center;justify-content:center;flex-direction:column;gap:8px">
  <div style="font-size:0.95rem">No results yet. Generate complaints to see the analysis here.</div>
</div>
""", unsafe_allow_html=True)