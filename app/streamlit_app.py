import os
import time
import re
import json
import base64
from pathlib import Path
import streamlit as st
import pandas as pd

from real_complaint_loader import load_real_complaints
from model_pipeline import (
    HierarchicalComplaintClassifier,
    DEFAULT_REJECTION_THRESHOLD,
    apply_eval_review_override,
)
from human_review_section import render_review_queue

st.set_page_config(page_title="ComplaintFlow", layout="wide")

HOLDOUT_DATA_PATH = os.path.join("data", "cfpb_2021-2022_holdout.csv")

# ── SESSION STATE DEFAULTS ────────────────────────────────────────────────────
defaults = {
    "dark_mode":           True,
    "results_df":          None,
    "chat_messages":       [],
    "results_open":        True,
    "chat_open":           True,
    "chat_fullscreen":     False,
    "results_fullscreen":  False,
    "review_decisions":    {},
    "review_finalised":    False,
    "results_with_review": None,
    "nav_section":         "About",
    "active_app_tab":      "Activity Log",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

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

[data-testid="stSidebar"] div.stButton > button {{
    display: flex !important;
    justify-content: flex-start !important;
    text-align: left !important;
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01)) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 2px 10px rgba(0,0,0,0.18);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    letter-spacing: 0.01em;
}}
[data-testid="stSidebar"] div.stButton > button:hover {{
    transform: translateY(-1px);
    background: linear-gradient(135deg, rgba(129,140,248,0.14), rgba(255,255,255,0.02)) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.07), 0 6px 16px rgba(0,0,0,0.28);
}}
[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, var(--primary), var(--primary-hover)) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.22), 0 6px 20px rgba(129,140,248,0.35) !important;
}}
[data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {{
    transform: translateY(-1px);
    background: linear-gradient(135deg, var(--primary-hover), var(--primary)) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.25), 0 8px 24px rgba(129,140,248,0.45) !important;
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

div[data-testid="stExpander"] {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
    overflow: hidden;
}}
div[data-testid="stExpander"] summary {{
    padding: 10px 14px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: var(--text) !important;
}}
div[data-testid="stExpander"] summary:hover {{
    color: var(--primary) !important;
}}
div[data-testid="stExpander"] summary svg {{
    fill: var(--muted) !important;
}}
div[data-testid="stExpanderDetails"] {{
    padding: 4px 14px 16px 14px !important;
}}

div[data-testid="stToggle"] label div[data-checked="true"] {{
    background-color: var(--primary) !important;
}}

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
  <div style="font-size:0.8rem; opacity:0.7;">Hit <b>Load & Classify</b> to sample real held-out complaints.</div>
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
    if "joint_perplexity" not in df.columns:
        df["joint_perplexity"] = 1.0
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
        "subissue_correct", "joint_confidence", "joint_perplexity", "needs_review", "review_source",
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
  text-align:center;padding:12px 8px;color:var(--muted);background:var(--surface);
  border-bottom:1px solid var(--border);position:sticky;top:0;cursor:pointer;font-weight:600;
  white-space:nowrap
}}
.dtbl th:hover{{color:var(--text)}}
.dtbl td{{padding:10px 8px;border-bottom:1px solid var(--border);vertical-align:top;text-align:center}}
.dtbl tr:hover{{background:rgba(255,255,255,0.02)}}

.badge{{padding:3px 9px;border-radius:4px;font-size:12px;font-weight:600}}
.b-auto{{background:rgba(16,185,129,0.1);color:var(--accent)}}
.b-review{{background:rgba(239,68,68,0.1);color:var(--danger)}}
.b-human{{background:rgba(99,102,241,0.15);color:var(--primary)}}
.match-y{{color:var(--accent);font-weight:700}}.match-n{{color:var(--danger);font-weight:700}}
.conf-chip{{padding:3px 7px;border-radius:4px;font-weight:600;font-size:12px;display:inline-block;min-width:44px;text-align:center}}
.col-center{{text-align:center !important}}
.col-right{{text-align:right !important}}
.c-hi{{background:rgba(16,185,129,0.15);color:var(--accent)}}
.c-mid{{background:rgba(99,102,241,0.15);color:var(--primary)}}
.c-lo{{background:rgba(239,68,68,0.15);color:var(--danger)}}
.p-hi{{background:rgba(45,212,191,0.15);color:#2dd4bf}}
.p-mid{{background:rgba(251,191,36,0.15);color:#fbbf24}}
.p-lo{{background:rgba(251,113,133,0.15);color:#fb7185}}
.si{{font-size:11px;margin-left:4px;opacity:0.3}}.si.on{{opacity:1;color:var(--primary)}}
</style>
</head>
<body>
  <div class="metrics">
    <div class="metric"><div class="metric-label">Total</div><div class="metric-value" id="m-total">0</div><div class="metric-sub" id="m-total-sub"></div></div>
    <div class="metric"><div class="metric-label">Auto-Labelled</div><div class="metric-value" id="m-auto">0</div><div class="metric-sub" id="m-auto-pct"></div></div>
    <div class="metric"><div class="metric-label">Needs Review</div><div class="metric-value" id="m-review">0</div><div class="metric-sub" id="m-review-pct"></div></div>
    <div class="metric"><div class="metric-label">Level 1 Accuracy</div><div class="metric-value" id="m-l1">0%</div><div class="metric-sub">all complaints</div></div>
    <div class="metric"><div class="metric-label">Level 2 Accuracy</div><div class="metric-value" id="m-l2">0%</div><div class="metric-sub">all complaints</div></div>
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
      <div class="card"><div class="card-title">Level 1 accuracy by broad issue</div><div id="l1-cat"></div></div>
      <div class="card"><div class="card-title">Level 2 accuracy by sub-issue</div><div id="l2-cat"></div></div>
    </div>
  </div>

  <div id="tab-breakdown" class="tab-panel">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
      <div class="card"><div class="card-title">Broad issue — truth vs predicted</div><table class="btbl" id="tbl-l1"><thead><tr><th>Category</th><th class="col-right">Truth</th><th class="col-right">Predicted</th><th class="col-right">Diff</th></tr></thead><tbody></tbody></table></div>
      <div class="card"><div class="card-title">Sub-issue — truth vs predicted</div><table class="btbl" id="tbl-l2"><thead><tr><th>Category</th><th class="col-right">Truth</th><th class="col-right">Predicted</th><th class="col-right">Diff</th></tr></thead><tbody></tbody></table></div>
    </div>
  </div>

  <div id="tab-detail" class="tab-panel">
    <div class="dtbl-wrap" style="margin-top:4px">
      <table class="dtbl" id="dtbl">
        <thead>
          <tr><th class="col-center" style="width:36px" data-col="idx"># <span class="si" id="si-idx"></span></th><th style="width:200px;text-align:left">Complaint</th><th class="col-center" style="width:70px">Status</th><th data-col="true_issue">True issue <span class="si" id="si-true_issue"></span></th><th data-col="predicted_issue_broad">Pred issue <span class="si" id="si-predicted_issue_broad"></span></th><th class="col-center" style="width:64px" data-col="issue_correct">Level 1 <span class="si" id="si-issue_correct"></span></th><th data-col="true_subissue">True sub-issue <span class="si" id="si-true_subissue"></span></th><th data-col="predicted_subissue">Pred sub-issue <span class="si" id="si-predicted_subissue"></span></th><th class="col-center" style="width:64px" data-col="subissue_correct">Level 2 <span class="si" id="si-subissue_correct"></span></th><th class="col-center" style="width:62px" data-col="joint_confidence">Conf <span class="si" id="si-joint_confidence"></span></th><th class="col-center" style="width:80px" data-col="joint_perplexity">Perplexity <span class="si" id="si-joint_perplexity"></span></th></tr>
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
    barRow('Level 1 correct', l1Acc, 'var(--primary)') + 
    barRow('Level 1 incorrect', 1 - l1Acc, 'var(--danger)') +
    barRow('Level 2 correct', l2Acc, 'var(--primary)') +
    barRow('Level 2 incorrect', 1 - l2Acc, 'var(--danger)');

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
    return `<tr><td>${{i}}</td><td class="col-right">${{gt}}</td><td class="col-right">${{pr}}</td><td class="col-right" style="color:${{col}};font-weight:600">${{d > 0 ? '+' : ''}}${{d}}</td></tr>`;
  }}).join('') || '<tr><td colspan="4" style="color:var(--muted);padding:10px">No data.</td></tr>';

  const allSub = unique([...data.map(r => r.true_subissue), ...data.map(r => r.predicted_subissue)]);
  document.querySelector('#tbl-l2 tbody').innerHTML = allSub.map(s => {{
    const gt = data.filter(r => r.true_subissue === s).length;
    const pr = data.filter(r => r.predicted_subissue === s).length;
    const d  = pr - gt;
    const col = d === 0 ? 'var(--muted)' : d > 0 ? 'var(--accent)' : 'var(--danger)';
    return `<tr><td>${{s}}</td><td class="col-right">${{gt}}</td><td class="col-right">${{pr}}</td><td class="col-right" style="color:${{col}};font-weight:600">${{d > 0 ? '+' : ''}}${{d}}</td></tr>`;
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
    const cls = conf >= 80 ? 'p-hi' : conf >= 60 ? 'p-mid' : 'p-lo';
    const perp = r.joint_perplexity.toFixed(2);
    const perpCls = r.joint_perplexity < 2.0 ? 'p-hi' : r.joint_perplexity < 3.0 ? 'p-mid' : 'p-lo';
    return `<tr>
      <td class="col-center" style="font-weight:600">${{r.idx + 1}}</td>
      <td title="${{r.complaint_text}}" style="font-size:12px;text-align:left">${{snip}}</td>
      <td class="col-center">${{badge}}</td>
      <td style="font-size:12px">${{r.true_issue}}</td>
      <td style="font-size:12px">${{r.predicted_issue_broad}}</td>
      <td class="col-center">${{ic}}</td>
      <td style="font-size:12px">${{r.true_subissue}}</td>
      <td style="font-size:12px">${{r.predicted_subissue}}</td>
      <td class="col-center">${{sc}}</td>
      <td class="col-center"><span class="conf-chip ${{cls}}">${{conf}}%</span></td>
      <td class="col-center"><span class="conf-chip ${{perpCls}}" title="~1.0 certain, ~2.0 moderate, ≥3.0 high uncertainty">${{perp}}</span></td>
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
def _badge_html() -> str:
    logo_path = Path(__file__).parent / "logo.png"
    if not logo_path.exists():
        return ""
    data = base64.b64encode(logo_path.read_bytes()).decode()
    return (
        '<div style="text-align:center;padding:2.5rem 0 1rem 0">'
        f'<img src="data:image/png;base64,{data}" '
        'style="max-width:360px;width:100%;height:auto"/></div>'
    )


# ── FULL APP ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="display:flex;align-items:center;gap:10px;padding:2px 2px 20px 2px">
  <div style="width:34px;height:34px;border-radius:9px;flex-shrink:0;
              background:linear-gradient(135deg,var(--primary),var(--primary-hover));
              display:flex;align-items:center;justify-content:center;font-weight:800;
              color:#fff;font-size:14px;box-shadow:0 4px 14px rgba(129,140,248,0.35)">CF</div>
  <div>
    <div style="font-weight:700;font-size:0.92rem;letter-spacing:-0.01em">ComplaintFlow</div>
    <div style="font-size:0.68rem;color:var(--muted)">NLP Complaint Classifier</div>
  </div>
</div>
""", unsafe_allow_html=True)

    icons = {"About": "◇", "Demo": "▷", "App": "⚙"}
    for label in ["About", "Demo", "App"]:
        is_active = st.session_state.nav_section == label
        if st.button(
            f"{icons[label]}   {label}",
            key=f"nav_{label}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state.nav_section = label
            st.rerun()

    st.divider()
    st.markdown("**System Status**")
    if os.path.exists(HOLDOUT_DATA_PATH):
        st.success("Held-out data found")
    else:
        st.error(f"Held-out data not found at `{HOLDOUT_DATA_PATH}`. Place the CFPB 2021-2022 holdout CSV there.")

    st.divider()
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

    st.markdown("""
<div style="margin-top:3rem;padding-top:14px;border-top:1px solid var(--border);
            text-align:center;color:var(--muted);font-size:0.7rem;letter-spacing:0.03em">
  ComplaintFlow · v1.0
</div>
""", unsafe_allow_html=True)

def render_app_tabs():
    st.markdown(_badge_html(), unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────────────────
    tab_names = ["Activity Log", "Results & Analysis", "Human Review Queue"]
    tab_cols = st.columns(3)
    for col, name in zip(tab_cols, tab_names):
        with col:
            is_active = st.session_state.active_app_tab == name
            if st.button(
                name,
                key=f"apptab_{name}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state.active_app_tab = name
                st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if st.session_state.active_app_tab == "Activity Log":
        chat_col, ctrl_col = st.columns([5, 1])

        with ctrl_col:
            st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
            n_complaints = st.slider("Number of complaints", min_value=10, max_value=30, value=20, step=1)
            threshold = st.slider("Joint Confidence Threshold", 0.0, 1.0, DEFAULT_REJECTION_THRESHOLD, 0.01)
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            generate_clicked = st.button("Load & Classify", type="primary", use_container_width=True)

        with chat_col:
            render_chat("400px")

            if generate_clicked:
                user_msg = f"Classify {n_complaints} student loan complaints, threshold {threshold:.2f}"
                st.session_state.chat_messages = [{"role": "user", "content": user_msg}, {"role": "assistant", "content": "Processing pipeline..."}]
                st.session_state.review_decisions = {}; st.session_state.review_finalised = False; st.session_state.results_with_review = None

                with st.spinner("Running pipeline..."):
                    try:
                        records = load_real_complaints(HOLDOUT_DATA_PATH, n=n_complaints, random_state=None)

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
                        results = apply_eval_review_override(results)
                        # Also flag rows for review when the sub-issue (Level 2) prediction is
                        # wrong, even if Level 1 was correct. apply_eval_review_override only
                        # forces review on Level 1 mistakes by design (see model_pipeline.py);
                        # this extends that behaviour without touching that file.
                        results["needs_review"] = results["needs_review"] | (~results["subissue_correct"])
                        st.session_state.results_df = results
                        st.session_state.chat_messages[-1] = {"role": "assistant", "content": f"Loaded {n_complaints} real held-out complaints. **{len(results)}** classified. Check the results below."}
                        st.session_state.active_app_tab = "Results & Analysis"
                    except (FileNotFoundError, ValueError) as e:
                        st.session_state.chat_messages[-1] = {"role": "assistant", "content": f"Could not load held-out data: {e}"}
                    except Exception as e:
                        st.session_state.chat_messages[-1] = {"role": "assistant", "content": f"Pipeline failed: {e}"}
                st.rerun()

    elif st.session_state.active_app_tab == "Results & Analysis":
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
      <div style="font-size:0.95rem">No results yet. Load real complaints to see the analysis here.</div>
    </div>
    """, unsafe_allow_html=True)

    elif st.session_state.active_app_tab == "Human Review Queue":
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


# ── NAV ROUTING ───────────────────────────────────────────────────────────────
if st.session_state.nav_section == "About":
    CONTRIBUTORS = [
        {
            "name": "antonisraf",
            "bio": "Into tech and business. Just learning a bit of everything as long as it's cool",
            "avatar": "https://avatars.githubusercontent.com/u/236162190?v=4",
            "url": "https://github.com/antonisraf",
        },
        {
            "name": "angelosdav",
            "bio": "Data enthusiast curious about how systems work.",
            "avatar": "https://avatars.githubusercontent.com/u/270142810?v=4",
            "url": "https://github.com/angelosdav",
        },
        {
            "name": "Dimitrispgt",
            "bio": "",
            "avatar": "https://avatars.githubusercontent.com/u/290385305?v=4",
            "url": "https://github.com/Dimitrispgt",
        },
    ]

    cards = ""
    for c in CONTRIBUTORS:
        bio_text = c["bio"] if c["bio"] else "No bio yet"
        cards += f"""
<a href="{c['url']}" target="_blank" class="contributor-card" style="text-decoration:none;display:block">
  <div style="background:rgba(255,255,255,0.02);border:1px solid var(--border);border-radius:12px;
              padding:1.4rem;display:flex;align-items:center;gap:16px;min-height:96px;
              transition:transform .2s ease, border-color .2s ease, box-shadow .2s ease">
    <img src="{c['avatar']}" class="contributor-avatar"
         style="width:60px;height:60px;min-width:60px;border-radius:50%;object-fit:cover;
                border:2px solid var(--border);flex-shrink:0;
                transition:border-color .2s ease, box-shadow .2s ease"/>
    <div style="text-align:left">
      <div style="font-weight:700;font-size:1.02rem;color:var(--text);
                  display:flex;align-items:center;gap:6px">
        {c['name']}
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" style="color:var(--muted)">
          <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                   0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                   -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07
                   -1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82
                   .64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12
                   .51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48
                   0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/>
        </svg>
      </div>
      <div style="font-size:0.9rem;color:var(--muted);line-height:1.55;margin-top:2px">{bio_text}</div>
    </div>
  </div>
</a>
"""

    st.markdown("""
<style>
.contributor-card:hover > div {
  transform:translateX(3px);
  border-color:rgba(129,140,248,0.4) !important;
  box-shadow:0 4px 16px rgba(129,140,248,0.12);
}
.contributor-card:hover .contributor-avatar {
  border-color:#818cf8 !important;
  box-shadow:0 0 0 4px rgba(129,140,248,0.15);
}
@keyframes titleShimmer {
  0%   { background-position: 0% 50%; }
  100% { background-position: 300% 50%; }
}
.contributors-title {
  animation: titleShimmer 4s linear infinite;
  background: linear-gradient(90deg,#f8fafc,#818cf8,#6366f1,#818cf8,#f8fafc) !important;
  background-size: 300% 100% !important;
  -webkit-background-clip: text !important;
  background-clip: text !important;
  color: transparent !important;
  -webkit-text-fill-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

    st.markdown('<div style="max-width:1180px;margin:2.5rem auto;padding:0 1rem">', unsafe_allow_html=True)

    # ---- Top: About (full width) ----
    st.markdown("""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:14px;
            padding:2.2rem;box-shadow:0 4px 20px rgba(0,0,0,0.2);margin-bottom:20px">
  <h2 style="margin-bottom:0.9rem;font-size:1.55rem;letter-spacing:-0.02em">About ComplaintFlow</h2>
  <p style="color:var(--muted);line-height:1.7;font-size:0.98rem">
    ComplaintFlow is an Hierarchical NLP pipeline that classifies CFPB student loan complaints
    into structured Issue and Sub-issue labels, with confidence-based routing to a human review app.
  </p>
</div>
""", unsafe_allow_html=True)

    # ---- Bottom row: Contributors (left) / Contribute (right) ----
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown(f"""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:14px;
            padding:2.2rem;box-shadow:0 4px 20px rgba(0,0,0,0.2)">
  <h3 class="contributors-title" style="margin:0 0 1.4rem 0;font-size:1.4rem;font-weight:800;letter-spacing:-0.02em">
    Contributors
  </h3>
  <div style="display:flex;flex-direction:column;gap:12px">
    {cards}
  </div>
</div>
""", unsafe_allow_html=True)

    with right:
        st.markdown("""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:14px;
            padding:2.2rem;box-shadow:0 4px 20px rgba(0,0,0,0.2)">
  <h3 style="margin:0 0 0.6rem 0;font-size:1.15rem;font-weight:700;letter-spacing:-0.01em;color:var(--text)">
    Want to contribute?
  </h3>
  <p style="color:var(--muted);font-size:0.92rem;line-height:1.65;margin-bottom:1.3rem">
    Bug fixes, new features, or improvements to the classifier itself, here's how to get started.
  </p>

  <div style="display:flex;flex-direction:column;gap:12px;margin-bottom:1.4rem">
    <div style="display:flex;gap:12px">
      <div style="width:24px;height:24px;border-radius:50%;background:rgba(129,140,248,0.12);
                  border:1px solid rgba(129,140,248,0.3);display:flex;align-items:center;
                  justify-content:center;font-size:12px;font-weight:700;color:#818cf8;flex-shrink:0">1</div>
      <div style="font-size:0.92rem;color:var(--muted);line-height:1.55">
        <b style="color:var(--text)">Fork the repo</b> and clone it locally
      </div>
    </div>
    <div style="display:flex;gap:12px">
      <div style="width:24px;height:24px;border-radius:50%;background:rgba(129,140,248,0.12);
                  border:1px solid rgba(129,140,248,0.3);display:flex;align-items:center;
                  justify-content:center;font-size:12px;font-weight:700;color:#818cf8;flex-shrink:0">2</div>
      <div style="font-size:0.92rem;color:var(--muted);line-height:1.55">
        <b style="color:var(--text)">Create a branch</b> and commit your work
      </div>
    </div>
    <div style="display:flex;gap:12px">
      <div style="width:24px;height:24px;border-radius:50%;background:rgba(129,140,248,0.12);
                  border:1px solid rgba(129,140,248,0.3);display:flex;align-items:center;
                  justify-content:center;font-size:12px;font-weight:700;color:#818cf8;flex-shrink:0">3</div>
      <div style="font-size:0.92rem;color:var(--muted);line-height:1.55">
        <b style="color:var(--text)">Open a pull request</b> describing your change
      </div>
    </div>
  </div>

  <a href="https://github.com/antonisraf/NLP-Tagging-Consumer-Complaints" target="_blank" style="text-decoration:none">
    <div style="display:inline-flex;align-items:center;gap:8px;background:#818cf8;color:white;
                font-weight:600;font-size:0.92rem;padding:0.65rem 1.4rem;border-radius:8px">
      View the repo on GitHub
    </div>
  </a>
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
elif st.session_state.nav_section == "Demo":
    import hashlib

    if "demo_ran_for" not in st.session_state:
        st.session_state.demo_ran_for = None

    DEMO_EXAMPLES = {
        "Loan Servicing & Payments — Loan Information & Servicing": {
            "text": (
                "My loan servicer keeps sending me incorrect account statements. "
                "The balance shown online does not match what they told me over the phone, "
                "and I have been unable to get a clear breakdown of my remaining principal, "
                "interest, and fees despite multiple requests over the past three months."
            ),
            "true_issue": "Loan Servicing & Payments",
            "true_subissue": "Loan Information & Servicing",
        },
        "Loan Servicing & Payments — Payment & Repayment Issues": {
            "text": (
                "I enrolled in an income-driven repayment plan over six months ago but my "
                "servicer continues to charge me the old standard monthly amount. Two payments "
                "have already been debited incorrectly and I have not received any refund or "
                "confirmation that the plan change was actually processed."
            ),
            "true_issue": "Loan Servicing & Payments",
            "true_subissue": "Payment & Repayment Issues",
        },
        "Non-Servicing Issues — Credit Reporting Issues": {
            "text": (
                "My student loan servicer reported a late payment to all three credit bureaus "
                "even though I had an approved deferment in place at the time. My credit score "
                "dropped significantly as a result and the servicer has refused to submit a "
                "correction despite my repeated written disputes."
            ),
            "true_issue": "Non-Servicing Issues",
            "true_subissue": "Credit Reporting Issues",
        },
        "Non-Servicing Issues — Loan Acquisition & Eligibility": {
            "text": (
                "I applied for a federal student loan to cover my upcoming semester but was "
                "told I am ineligible due to a prior default that was already resolved and "
                "removed from my record two years ago. The financial aid office cannot explain "
                "why the system still shows me as ineligible and I am at risk of losing my "
                "enrollment for the semester."
            ),
            "true_issue": "Non-Servicing Issues",
            "true_subissue": "Loan Acquisition & Eligibility",
        },
    }

    # Only structural CSS here, no background colors, no new palette.
    # Everything below reuses var(--surface) / var(--border) / var(--muted) / var(--text)
    # so this section matches the rest of the app instead of introducing its own theme.
    st.markdown("""
<style>
@keyframes bar-fill { from { width:0%; } }
.bar-fill { animation: bar-fill 1s cubic-bezier(.16,1,.3,1); }
</style>
""", unsafe_allow_html=True)

    # ── Section header ──────────────────────────────────────────────────────
    st.markdown("""
<div style="max-width:1180px;margin:2.5rem auto 1.8rem auto;padding:0 1rem;text-align:left">
  <div style="font-size:11px;font-weight:700;letter-spacing:0.14em;color:#818cf8;
              text-transform:uppercase;margin-bottom:0.5rem;text-align:left">Live Demo · 02</div>
  <h2 style="margin:0;font-size:1.7rem;letter-spacing:-0.03em;max-width:640px;
             color:var(--text);text-align:left">
    Run a real complaint through the classifier
  </h2>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div style="max-width:1180px;margin:0 auto;padding:0 1rem">', unsafe_allow_html=True)
    rail, workspace = st.columns([1, 2], gap="large")

    with rail:
        st.markdown("""
<div style="position:sticky;top:1.5rem">
  <p style="color:var(--muted);line-height:1.75;font-size:0.86rem;margin-bottom:1.8rem">
    ComplaintFlow is a hierarchical classifier trained on CFPB student loan complaints.
    A two-stage ensemble of Logistic Regression and LinearSVC models on TF-IDF features
    assigns a <b style="color:var(--text)">broad issue</b> and a <b style="color:var(--text)">sub-issue</b>,
    each backed by a joint confidence and joint perplexity score. Anything uncertain gets
    routed to the <b style="color:var(--text)">Human Review Queue</b> instead of guessed at.
  </p>

  <div style="border-left:1px solid var(--border);margin-left:3px">
    <div style="display:flex;gap:16px;padding:0 0 24px 0;position:relative">
      <div style="width:8px;height:8px;border-radius:50%;background:#818cf8;
                  margin-left:-5px;margin-top:4px;flex-shrink:0"></div>
      <div style="font-size:0.82rem;color:var(--muted);line-height:1.6">
        <b style="color:var(--text)">Pick a category</b> from the dropdown
      </div>
    </div>
    <div style="display:flex;gap:16px;padding:0 0 24px 0;position:relative">
      <div style="width:8px;height:8px;border-radius:50%;background:#818cf8;
                  margin-left:-5px;margin-top:4px;flex-shrink:0"></div>
      <div style="font-size:0.82rem;color:var(--muted);line-height:1.6">
        <b style="color:var(--text)">Read the complaint</b> that gets sent in
      </div>
    </div>
    <div style="display:flex;gap:16px;position:relative">
      <div style="width:8px;height:8px;border-radius:50%;background:#818cf8;
                  margin-left:-5px;margin-top:4px;flex-shrink:0"></div>
      <div style="font-size:0.82rem;color:var(--muted);line-height:1.6">
        <b style="color:var(--text)">Hit Classify</b> and see if the model gets it right
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    with workspace:
        selected = st.selectbox(
            "Category",
            options=list(DEMO_EXAMPLES.keys()),
            label_visibility="collapsed",
            key="demo_select",
        )

        example        = DEMO_EXAMPLES[selected]
        complaint_text = example["text"]
        true_issue     = example["true_issue"]
        true_subissue  = example["true_subissue"]
        docket_id      = "#" + hashlib.md5(selected.encode()).hexdigest()[:5].upper()

        st.markdown(f"""
<div style="margin:8px 0 14px 0;padding:0;
            background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden">
  <div style="display:flex;justify-content:space-between;align-items:center;
              padding:9px 18px;background:rgba(255,255,255,0.02);
              border-bottom:1px solid var(--border);font-size:10.5px;
              letter-spacing:0.06em;color:var(--muted);font-family:monospace">
    <span>CASE {docket_id}</span>
    <span>READY</span>
  </div>
  <div style="padding:18px 20px;font-size:0.87rem;line-height:1.8;color:var(--muted)">
    {complaint_text}
  </div>
</div>
""", unsafe_allow_html=True)

        run_demo = st.button("Classify", type="primary", use_container_width=True, key="demo_run")
        if run_demo:
            st.session_state.demo_ran_for = selected

        # Persisted across reruns, unlike `run_demo`, this stays true even after
        # a later rerun (e.g. clicking "Try the full app"), so buttons rendered
        # below inside this block still get instantiated and can receive clicks.
        show_result = st.session_state.get("demo_ran_for") == selected

        # ── Result ────────────────────────────────────────────────────────────
        if show_result:
            with st.spinner("Running classifier..."):
                try:
                    clf = load_classifier()
                    demo_result = clf.predict([complaint_text])
                    row = demo_result.iloc[0]

                    issue     = row["predicted_issue_broad"]
                    subissue  = row["predicted_subissue"]
                    conf      = row["joint_confidence"]
                    perp      = row["joint_perplexity"]

                    def score_color(pct):
                        # one shared scale used for every progress bar in this card
                        if pct >= 70:
                            return "#10b981"   # green — strong
                        elif pct >= 40:
                            return "#818cf8"   # indigo — moderate
                        else:
                            return "#dc2626"   # red — weak

                    conf_pct    = int(conf * 100)
                    clarity_pct = max(0, min(100, int(100 - (perp / 5.0) * 100)))

                    conf_color    = score_color(conf_pct)
                    clarity_color = score_color(clarity_pct)

                    l1_correct   = issue == true_issue
                    l2_correct   = subissue == true_subissue
                    needs_review = (not l1_correct) or (conf < DEFAULT_REJECTION_THRESHOLD)

                    if l1_correct and l2_correct:
                        verdict_badge = (
                            "<span style='background:rgba(16,185,129,0.1);color:#34d399;"
                            "padding:4px 12px;border-radius:6px;font-size:11px;font-weight:700;"
                            "letter-spacing:0.04em;text-transform:uppercase'>✓ Correct</span>"
                        )
                    elif l1_correct and not l2_correct:
                        verdict_badge = (
                            "<span style='background:rgba(251,191,36,0.1);color:#fbbf24;"
                            "padding:4px 12px;border-radius:6px;font-size:11px;font-weight:700;"
                            "letter-spacing:0.04em;text-transform:uppercase'>~ Partial</span>"
                        )
                    else:
                        verdict_badge = (
                            "<span style='background:rgba(239,68,68,0.1);color:#f87171;"
                            "padding:4px 12px;border-radius:6px;font-size:11px;font-weight:700;"
                            "letter-spacing:0.04em;text-transform:uppercase'>✗ Incorrect</span>"
                        )

                    issue_border    = "rgba(16,185,129,0.25)" if l1_correct else "rgba(239,68,68,0.25)"
                    issue_bg        = "rgba(16,185,129,0.06)" if l1_correct else "rgba(239,68,68,0.06)"
                    subissue_border = "rgba(16,185,129,0.25)" if l2_correct else "rgba(239,68,68,0.25)"
                    subissue_bg     = "rgba(16,185,129,0.06)" if l2_correct else "rgba(239,68,68,0.06)"

                    if needs_review:
                        reason = (
                            f"Wrong Level 1 prediction and low confidence ({conf_pct}%)"
                            if not l1_correct
                            else f"Low confidence ({conf_pct}%) and high perplexity ({perp:.2f})"
                        )
                        st.markdown(f"""
<div style="margin-top:1.4rem;padding:12px 18px;
            background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.2);
            border-radius:10px;display:flex;align-items:center;gap:12px">
  <span style="font-size:1.1rem">⚠</span>
  <span style="font-size:0.82rem;color:#fbbf24;line-height:1.6">
    {reason} — in the full pipeline this complaint would be routed to the
    <b>Human Review Queue</b> for manual labelling.
  </span>
</div>
""", unsafe_allow_html=True)

                    st.markdown(f"""
<div style="margin-top:1rem;padding:1.8rem;
            background:var(--surface);border:1px solid var(--border);
            border-radius:14px;box-shadow:0 4px 24px rgba(0,0,0,0.25)">
  <div style="display:flex;justify-content:space-between;align-items:center;
              margin-bottom:1.4rem;padding-bottom:1rem;border-bottom:1px solid var(--border)">
    <span style="font-weight:700;font-size:0.95rem;letter-spacing:-0.01em;color:var(--text)">Classification Result</span>
    {verdict_badge}
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px">
    <div style="padding:14px 16px;background:{issue_bg};border:1px solid {issue_border};border-radius:10px">
      <div style="font-size:9px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">Broad Issue · Level 1</div>
      <div style="font-size:0.88rem;font-weight:600;color:var(--text)">{issue}</div>
      <div style="font-size:10px;color:var(--muted);margin-top:4px">Expected: {true_issue}</div>
    </div>
    <div style="padding:14px 16px;background:{subissue_bg};border:1px solid {subissue_border};border-radius:10px">
      <div style="font-size:9px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">Sub-Issue · Level 2</div>
      <div style="font-size:0.88rem;font-weight:600;color:var(--text)">{subissue}</div>
      <div style="font-size:10px;color:var(--muted);margin-top:4px">Expected: {true_subissue}</div>
    </div>
  </div>

  <div style="display:flex;flex-direction:column;gap:14px">
    <div>
      <div style="display:flex;justify-content:space-between;margin-bottom:6px">
        <span style="font-size:9px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em">Joint Confidence</span>
        <span style="font-size:12px;font-weight:700;color:{conf_color}">{conf_pct}%</span>
      </div>
      <div style="height:8px;border-radius:999px;background:rgba(255,255,255,0.06);overflow:hidden">
        <div class="bar-fill" style="height:100%;width:{conf_pct}%;border-radius:999px;background:{conf_color}"></div>
      </div>
    </div>
    <div>
      <div style="display:flex;justify-content:space-between;margin-bottom:6px">
        <span style="font-size:9px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em">Joint Perplexity ({perp:.2f})</span>
        <span style="font-size:12px;font-weight:700;color:{clarity_color}">{clarity_pct}% clarity</span>
      </div>
      <div style="height:8px;border-radius:999px;background:rgba(255,255,255,0.06);overflow:hidden">
        <div class="bar-fill" style="height:100%;width:{clarity_pct}%;border-radius:999px;background:{clarity_color}"></div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

                    st.markdown("""
<div style="margin-top:1rem;padding:1.2rem 1.6rem;
            background:var(--surface);border:1px solid var(--border);border-radius:12px;
            display:flex;justify-content:space-between;align-items:center;gap:1.2rem;flex-wrap:wrap">
  <div style="font-size:0.85rem;color:var(--muted)">
    Curious how it holds up on real, unseen complaints?
  </div>
</div>
""", unsafe_allow_html=True)
                    if st.button("Try the full app →", key="goto_app_cta"):
                        st.session_state.nav_section = "App"
                        st.rerun()

                except Exception as e:
                    st.error(f"Classification failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.nav_section == "App":
    render_app_tabs()