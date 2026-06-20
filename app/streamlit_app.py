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

# ── THEME (DARK MODE ONLY, refined glass palette) ─────────────────────────────
DARK_BG       = "#0b0f1a"
DARK_SURFACE  = "#1e293b"                  # opaque, used where legibility > depth (sidebar)
GLASS_SURFACE = "rgba(30, 41, 59, 0.6)"    # translucent, used for cards/panels/bubbles

PRIMARY       = "#818cf8"
PRIMARY_HOVER = "#6366f1"
ACCENT        = "#10b981"
DANGER        = "#dc2626"

bg            = DARK_BG
surface       = GLASS_SURFACE
surface_solid = DARK_SURFACE
text          = "#f8fafc"
muted         = "#94a3b8"
border        = "rgba(255,255,255,0.06)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
    --primary: {PRIMARY};
    --primary-hover: {PRIMARY_HOVER};
    --bg: {bg};
    --surface: {surface};
    --surface-solid: {surface_solid};
    --text: {text};
    --muted: {muted};
    --border: {border};
    --font: 'Inter', sans-serif;
}}

html, body,
[data-testid="stHeader"] {{
    background-color: var(--bg) !important;
}}

[data-testid="stAppViewContainer"] {{
    background-color: var(--bg) !important;
    background-image:
        radial-gradient(circle at 12% 8%, rgba(129, 140, 248, 0.14), transparent 42%),
        radial-gradient(circle at 88% 0%, rgba(16, 185, 129, 0.07), transparent 38%),
        radial-gradient(circle at 50% 100%, rgba(99, 102, 241, 0.10), transparent 48%);
    background-attachment: fixed;
    color: var(--text) !important;
    font-family: var(--font) !important;
}}

[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
.main, .block-container {{
    background-color: transparent !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
    padding-top: 0 !important;
}}

[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {{
    background-color: var(--surface-solid) !important;
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
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}}
div.stButton > button[kind="primary"]:hover {{
    background-color: var(--primary-hover) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
}}
div.stButton > button {{
    background-color: var(--surface) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 500 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
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
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--border) !important;
    border-left: 4px solid var(--primary) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}}
div[data-testid="stAlert"] * {{ color: var(--text) !important; }}
div[data-testid="stAlert"] svg {{ display: none !important; }}

.chat-container {{
    display: flex; flex-direction: column; gap: 8px;
    padding: 1rem; background: var(--surface); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}}
.chat-bubble {{ display: flex; flex-direction: column; max-width: 90%; margin-bottom: 8px; }}
.chat-bubble.user {{ align-self: flex-end; align-items: flex-end; }}
.chat-bubble.assistant {{ align-self: flex-start; align-items: flex-start; }}
.chat-bubble .bubble-inner {{
    padding: 10px 16px; border-radius: 16px; font-size: 0.85rem; line-height: 1.6;
}}
.chat-bubble.user .bubble-inner {{
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-hover) 100%);
    color: white !important;
}}
.chat-bubble.assistant .bubble-inner {{
    background: rgba(11, 15, 26, 0.6); border: 1px solid var(--border);
    color: var(--text) !important;
}}
.chat-bubble .bubble-label {{ font-size: 0.7rem; color: var(--muted); margin-bottom: 4px; font-weight: 600; text-transform: uppercase; }}

.panel-header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 18px; background: var(--surface); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--border); border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    margin-bottom: 0 !important;
}}
.panel-header-title {{
    font-weight: 600; font-size: 0.95rem; color: var(--text); letter-spacing: -0.02em;
}}

[data-testid="stVerticalBlock"] > [data-testid="element-container"] + [data-testid="element-container"],
[data-testid="stVerticalBlock"] > [data-testid="element-container"] + [data-testid="stIframe"],
[data-testid="stVerticalBlock"] > [data-testid="stIframe"] + [data-testid="element-container"] {{
    margin-top: -1rem !important;
}}
[data-testid="stVerticalBlock"] > [data-testid="stIframe"] + [data-testid="element-container"]:has(.panel-header) {{
    margin-top: 0 !important;
}}

.key-modal {{
    background: var(--surface); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 12px; padding: 2.5rem 2rem;
    max-width: 480px; margin: 4rem auto; text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.37);
}}
.key-modal-title {{
    font-size: 1.4rem; font-weight: 700; letter-spacing: -0.02em; color: var(--text); margin-bottom: 0.5rem;
}}
.key-modal-sub {{ font-size: 0.9rem; color: var(--muted); margin-bottom: 1.5rem; line-height: 1.6; }}

[data-testid="stVerticalBlock"] > [data-testid="element-container"] {{
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
    margin-top: 0 !important;
    padding-top: 0 !important;
}}
[data-testid="stVerticalBlock"] > [data-testid="element-container"] + [data-testid="element-container"] {{
    margin-top: 8px !important;
}}
[data-testid="stVerticalBlock"] > [data-testid="element-container"]:has(.panel-header) + * {{
    margin-top: 4px !important;
}}
[data-testid="stCustomComponentV1"] {{
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}}
[data-testid="stVerticalBlock"] > [data-testid="stCustomComponentV1"] + [data-testid="element-container"] {{
    margin-top: 8px !important;
}}
[data-testid="stVerticalBlockBorderWrapper"] {{
    padding: 0 !important;
    margin: 0 !important;
}}

/* Tab styling */
[data-testid="stTabs"] [role="tablist"] {{
    gap: 4px;
    border-bottom: 1px solid var(--border);
}}
[data-testid="stTabs"] [role="tab"] {{
    background: transparent !important;
    color: var(--muted) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    font-family: var(--font) !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}}
[data-testid="stTabs"] [role="tab"]:hover {{
    color: var(--text) !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: var(--primary) !important;
    border-bottom-color: var(--primary) !important;
}}
[data-testid="stTabContent"] {{
    padding-top: 16px !important;
}}

hr {{ border-color: var(--border) !important; }}
div[data-testid="stSpinner"] {{
    display: flex !important;
    justify-content: center !important;
    padding: 8px 0 !important;
}}
div[data-testid="stSpinner"] p {{
    display: none !important;
}}
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

[data-testid="stIconMaterial"] {{
    font-size: 0 !important;
    color: transparent !important;
    width: 0 !important;
    overflow: hidden !important;
}}
[data-testid="stBaseButton-headerNoPadding"] {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    width: 2rem !important;
    height: 2rem !important;
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

function fixSidebarButtons() {{
    var p = window.parent.document;
    var collapseBtn = p.querySelector('[data-testid="stSidebarCollapseButton"] button');
    if (collapseBtn && !collapseBtn.dataset.fixed) {{
        collapseBtn.style.cssText = 'display:flex!important;align-items:center!important;justify-content:center!important;background:rgba(30,41,59,0.8)!important;border:1px solid rgba(255,255,255,0.06)!important;border-radius:8px!important;width:32px!important;height:32px!important;cursor:pointer!important;';
        collapseBtn.innerHTML = '<span style="color:#f8fafc;font-size:16px;line-height:1;">«</span>';
        collapseBtn.dataset.fixed = 'true';
    }}
    var expandBtn = p.querySelector('[data-testid="stExpandSidebarButton"]');
    if (expandBtn && !expandBtn.dataset.fixed) {{
        expandBtn.style.cssText = 'display:flex!important;align-items:center!important;justify-content:center!important;background:rgba(30,41,59,0.8)!important;border:1px solid rgba(255,255,255,0.06)!important;border-radius:8px!important;width:32px!important;height:32px!important;cursor:pointer!important;';
        expandBtn.innerHTML = '<span style="color:#f8fafc;font-size:16px;line-height:1;">»</span>';
        expandBtn.dataset.fixed = 'true';
    }}
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
fixSidebarButtons();
openSidebar();
setInterval(fixSidebarButtons, 500);
new MutationObserver(function() {{ fixSliders(); fixSidebarButtons(); }}).observe(window.parent.document.body, {{childList: true, subtree: true, attributes: true}});
</script>
""", height=0)

@st.cache_resource
def load_classifier():
    return HierarchicalComplaintClassifier()

def render_chat(height="420px"):
    if not st.session_state.chat_messages:
        st.markdown(f"""
<div style="background:var(--surface);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
            border:1px solid var(--border);border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.2);
            padding:2.5rem;text-align:center;color:var(--muted);height:{height};
            display:flex;align-items:center;justify-content:center;flex-direction:column;gap:10px">
  <div style="width:10px;height:10px;border-radius:50%;background:var(--primary);
              animation:pulseDot 1.8s ease-in-out infinite;"></div>
  <div style="font-size:0.9rem; font-weight:500; letter-spacing:-0.02em;">System Idle</div>
  <div style="font-size:0.8rem; opacity:0.7;">Hit <b>Generate & Classify</b> to start the pipeline.</div>
</div>
<style>
@keyframes pulseDot {{
  0%, 100% {{ opacity: 0.35; transform: scale(0.85); }}
  50% {{ opacity: 1; transform: scale(1.15); }}
}}
</style>
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
<div class="chat-container" style="height:{height};overflow-y:auto;justify-content:flex-end">{bubbles}</div>
""", unsafe_allow_html=True)


def render_dashboard(results: pd.DataFrame, dark_mode: bool):
    df = results.copy()

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


# ── LOGO BADGE ────────────────────────────────────────────────────────────────
# Replaces the previous generated SVG checkmark with the supplied artwork.
# The source PNG was cropped to its visible content (it shipped with large
# transparent margins) and downscaled/recompressed from ~290KB to ~89KB so it
# doesn't bloat this file too much as an inline base64 string.
LOGO_B64 = "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"

def _badge_html() -> str:
    return f'<div style="text-align:center;padding:2.5rem 0 1rem 0"><img src="data:image/png;base64,{LOGO_B64}" style="max-width:360px;width:100%;height:auto"/></div>'

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

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_log, tab_results, tab_review = st.tabs(["Activity Log", "Results & Analysis", "Human Review Queue"])

with tab_log:
    chat_col, ctrl_col = st.columns([5, 1])

    with ctrl_col:
        st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
        n_complaints = st.slider("Number of complaints", min_value=10, max_value=25, value=10, step=1)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        threshold = st.slider("Joint Confidence Threshold", 0.0, 1.0, DEFAULT_REJECTION_THRESHOLD, 0.01)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        generate_clicked = st.button("Generate & Classify", type="primary", use_container_width=True)

    with chat_col:
        render_chat("400px")

        if generate_clicked:
            allowed, secs_left = check_rate_limit()
            if not allowed:
                mins = secs_left // 60
                st.error(f"Rate limit reached. Please wait ~{mins} minute(s).")
                st.stop()

            user_msg = f"Classify {n_complaints} student loan complaints — threshold {threshold:.2f}"
            st.session_state.chat_messages = [{"role": "user", "content": user_msg}, {"role": "assistant", "content": "Processing pipeline..."}]
            st.session_state.review_decisions = {}; st.session_state.review_finalised = False; st.session_state.results_with_review = None

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
                    if "needs_review" not in results.columns:
                        results["needs_review"] = False
                    results["needs_review"] = results["needs_review"] | (~results["issue_correct"])
                    st.session_state.results_df = results
                    st.session_state.chat_messages[-1] = {"role": "assistant", "content": f"Done! **{len(results)}** complaints classified. Check the results below."}
                except Exception as e:
                    st.session_state.chat_messages[-1] = {"role": "assistant", "content": f"Pipeline failed: {e}"}
            st.rerun()

with tab_results:
    if st.session_state.results_df is not None:
        results = st.session_state.results_df

        if st.session_state.review_finalised and st.session_state.results_with_review is not None:
            display_results = st.session_state.results_with_review.copy()
            display_results["predicted_issue_broad"] = display_results["reviewed_issue"]
            display_results["predicted_subissue"]    = display_results["reviewed_subissue"]
        else:
            display_results = results

        render_dashboard(display_results, st.session_state.dark_mode)
    else:
        st.markdown("""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;
            padding:2.5rem;text-align:center;color:var(--muted);
            display:flex;align-items:center;justify-content:center;flex-direction:column;gap:8px">
  <div style="font-size:0.95rem">No results yet. Generate complaints to see the analysis here.</div>
</div>
""", unsafe_allow_html=True)

with tab_review:
    if st.session_state.results_df is not None:
        render_review_queue(st.session_state.results_df, st.session_state.dark_mode)
    else:
        st.markdown("""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;
            padding:2.5rem;text-align:center;color:var(--muted);
            display:flex;align-items:center;justify-content:center;flex-direction:column;gap:8px">
  <div style="font-size:0.95rem">No results yet. Run the pipeline first to review complaints.</div>
</div>
""", unsafe_allow_html=True)