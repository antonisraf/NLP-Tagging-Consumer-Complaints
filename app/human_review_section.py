"""
Human Review Queue for the hierarchical complaint classifier.

Renders a per-complaint review interface for any row where needs_review=True.
Saves decisions to st.session_state and applies them back to the results
DataFrame when the user finalises, updating accuracy metrics accordingly.

Public API:
    render_review_queue(results_df, dark_mode)
    apply_review_decisions(results_df)   -> pd.DataFrame
"""

import pandas as pd
import streamlit as st

# ── THEME (mirrors app.py's Slate/Indigo palette) ─────────────────────────────
DARK_SURFACE  = "#1e293b"  # Slate 800
LIGHT_SURFACE = "#ffffff"

PRIMARY       = "#6366f1"  # Indigo 500
PRIMARY_LIGHT = "#818cf8"  # Indigo 400
ACCENT        = "#10b981"  # Emerald 500 (Success)
DANGER        = "#ef4444"  # Red 500


# The canonical taxonomy used by the classifier. Confirmed to match the
# GROUPING dict in train_and_save_models.py and the taxonomy used by
# complaint_generator.py: 2 broad issues, each with exactly 2 sub-issues
# (4 sub-issues total, no overlap between groups).
ISSUE_SUBISSUE_MAP: dict[str, list[str]] = {
    "Loan Servicing & Payments": [
        "Loan Information & Servicing",
        "Payment & Repayment Issues",
    ],
    "Non-Servicing Issues": [
        "Credit Reporting Issues",
        "Loan Acquisition & Eligibility",
    ],
}

ALL_ISSUES = list(ISSUE_SUBISSUE_MAP.keys())

# Flattened list of all 4 sub-issues, plus the reverse lookup that derives
# the broad issue from a chosen sub-issue. Since each sub-issue belongs to
# exactly one broad issue (strict hierarchy, no overlap), the broad issue
# never needs to be picked independently — it's fully determined by the
# sub-issue choice. This lets the review UI show a single dropdown with all
# 4 sub-issues at once, instead of two cascading dropdowns where picking a
# broad issue first would only reveal the 2 sub-issues under it.
ALL_SUBISSUES = [sub for subs in ISSUE_SUBISSUE_MAP.values() for sub in subs]
SUBISSUE_TO_ISSUE = {
    sub: issue for issue, subs in ISSUE_SUBISSUE_MAP.items() for sub in subs
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_review_state():
    """Initialise review-related session state keys if they don't exist yet."""
    if "review_decisions" not in st.session_state:
        st.session_state.review_decisions = {}
    if "review_finalised" not in st.session_state:
        st.session_state.review_finalised = False
    if "results_with_review" not in st.session_state:
        st.session_state.results_with_review = None


def _reset_review():
    st.session_state.review_decisions = {}
    st.session_state.review_finalised = False
    st.session_state.results_with_review = None


def _progress_bar_html(done: int, total: int, dark_mode: bool) -> str:
    pct      = int(done / total * 100) if total else 0
    track_bg = "rgba(255,255,255,0.08)" if dark_mode else "rgba(0,0,0,0.08)"
    muted    = "#94a3b8" if dark_mode else "#64748b"
    text     = "#f1f5f9" if dark_mode else "#1e293b"
    return f"""
<div style="margin-bottom:12px">
  <div style="display:flex;justify-content:space-between;
              font-size:11px;color:{muted};margin-bottom:5px">
    <span>Progress</span>
    <span><b style="color:{text}">{done}</b> / {total} reviewed</span>
  </div>
  <div style="height:6px;border-radius:3px;background:{track_bg};overflow:hidden">
    <div style="width:{pct}%;height:100%;
                background:{PRIMARY};
                border-radius:3px;transition:width .3s ease"></div>
  </div>
</div>"""


def _complaint_card_html(
    idx: int,
    complaint_text: str,
    model_issue: str,
    model_subissue: str,
    model_confidence: float,
    is_decided: bool,
    dark_mode: bool,
) -> str:
    surface  = DARK_SURFACE if dark_mode else LIGHT_SURFACE
    border   = "rgba(255,255,255,0.08)" if dark_mode else "rgba(0,0,0,0.08)"
    muted    = "#94a3b8" if dark_mode else "#64748b"
    text     = "#f1f5f9" if dark_mode else "#1e293b"
    inner_bg = "rgba(255,255,255,0.03)" if dark_mode else "rgba(0,0,0,0.03)"
    conf_pct = int(model_confidence * 100)

    status_dot = (
        f"<span style='color:{PRIMARY};font-size:12px'>✓ Decided</span>"
        if is_decided
        else f"<span style='color:{DANGER};font-size:12px'>⬤ Pending</span>"
    )

    return f"""
<div style="background:{surface};border:1px solid {border};border-radius:8px;
            padding:14px 16px;margin-bottom:8px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
    <span style="font-size:11px;font-weight:700;color:{muted};
                 text-transform:uppercase;letter-spacing:.06em">
      Complaint #{idx + 1}
    </span>
    {status_dot}
  </div>
  <div style="font-size:12px;line-height:1.6;color:{text};
              padding:10px 12px;background:{inner_bg};
              border-radius:6px;margin-bottom:12px;
              max-height:100px;overflow-y:auto">
    {complaint_text}
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap">
    <div style="font-size:10px;color:{muted}">
      Model guess:
      <span style="background:rgba(99,102,241,0.12);color:{PRIMARY_LIGHT};
                   padding:2px 8px;border-radius:4px;font-weight:600;margin-left:4px">
        {model_subissue}
      </span>
    </div>
    <div style="font-size:10px;color:{muted}">
      Confidence:
      <span style="background:rgba(99,102,241,0.12);color:{PRIMARY_LIGHT};
                   padding:2px 8px;border-radius:4px;font-weight:600;margin-left:4px">
        {conf_pct}%
      </span>
    </div>
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Public: apply decisions back to the results DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def apply_review_decisions(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of results_df with three new columns:
      - reviewed_issue     : human label (or model label if auto-classified)
      - reviewed_subissue  : human label (or model label if auto-classified)
      - review_source      : "human" | "model"

    Rows fixed by a human are accepted as-is — the human takes responsibility
    for them, so we don't score them against the synthetic ground-truth
    labels. Only rows still resolved by the model keep their original
    accuracy verdict.
    """
    df = results_df.copy()
    decisions = st.session_state.get("review_decisions", {})

    reviewed_issue    = []
    reviewed_subissue = []
    review_source     = []

    for i, row in df.iterrows():
        if row["needs_review"] and i in decisions:
            reviewed_issue.append(decisions[i]["issue"])
            reviewed_subissue.append(decisions[i]["subissue"])
            review_source.append("human")
        else:
            reviewed_issue.append(row["predicted_issue_broad"])
            reviewed_subissue.append(row["predicted_subissue"])
            review_source.append("model")

    df["reviewed_issue"]    = reviewed_issue
    df["reviewed_subissue"] = reviewed_subissue
    df["review_source"]     = review_source

    is_human = df["review_source"] == "human"

    # Human-reviewed rows: trust the human, no need to verify against truth.
    df.loc[is_human, "issue_correct"]    = True
    df.loc[is_human, "subissue_correct"] = True

    # Model-only rows: keep scoring against the ground truth as before.
    df.loc[~is_human, "issue_correct"]    = df.loc[~is_human, "reviewed_issue"]    == df.loc[~is_human, "true_issue"]
    df.loc[~is_human, "subissue_correct"] = df.loc[~is_human, "reviewed_subissue"] == df.loc[~is_human, "true_subissue"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public: render the full review queue section
# ─────────────────────────────────────────────────────────────────────────────

def render_review_queue(results_df: pd.DataFrame, dark_mode: bool):
    """
    Renders the Human Review Queue section.

    Call this after the Results & Analysis section in app.py, passing the
    same `results` DataFrame that render_dashboard() receives.
    """
    _ensure_review_state()

    review_rows = results_df[results_df["needs_review"]].copy()

    if review_rows.empty:
        surface = DARK_SURFACE if dark_mode else LIGHT_SURFACE
        border  = "rgba(255,255,255,0.08)" if dark_mode else "rgba(0,0,0,0.08)"
        muted   = "#94a3b8" if dark_mode else "#64748b"
        st.markdown(f"""
<div style="background:{surface};border:1px solid {border};border-radius:8px;
            padding:2rem;text-align:center;color:{muted}">
  <div style="font-size:1.5rem;margin-bottom:8px"> </div>
  <div style="font-size:0.9rem">
    All complaints were auto-classified with confidence above the threshold.
    No human review needed.
  </div>
</div>""", unsafe_allow_html=True)
        return

    # ── If already finalised, show summary and re-review option ──────────────
    if st.session_state.review_finalised:
        _render_finalised_summary(results_df, review_rows, dark_mode)
        return

    # ── Active review queue ───────────────────────────────────────────────────
    decisions  = st.session_state.review_decisions
    total      = len(review_rows)
    done_count = sum(1 for idx in review_rows.index if idx in decisions)

    st.markdown(
        _progress_bar_html(done_count, total, dark_mode),
        unsafe_allow_html=True,
    )

    for row_idx, row in review_rows.iterrows():
        is_decided = row_idx in decisions
        current_issue = (
            decisions[row_idx]["issue"]
            if is_decided
            else row["predicted_issue_broad"]
        )
        current_subissue = (
            decisions[row_idx]["subissue"]
            if is_decided
            else row["predicted_subissue"]
        )

        st.markdown(
            _complaint_card_html(
                idx=row_idx,
                complaint_text=row["complaint_text"],
                model_issue=row["predicted_issue_broad"],
                model_subissue=row["predicted_subissue"],
                model_confidence=row["joint_confidence"],
                is_decided=is_decided,
                dark_mode=dark_mode,
            ),
            unsafe_allow_html=True,
        )

        col_issue, col_sub, col_btn = st.columns([2, 2, 1])

        # Both dropdowns are fully independent and manually selectable — the
        # broad issue dropdown shows all 2 options and the sub-issue dropdown
        # shows all 4 options, with no filtering/auto-derivation between them.
        with col_issue:
            chosen_issue = st.selectbox(
                "Broad issue",
                options=ALL_ISSUES,
                index=ALL_ISSUES.index(current_issue) if current_issue in ALL_ISSUES else 0,
                key=f"review_issue_{row_idx}",
                label_visibility="collapsed",
            )

        with col_sub:
            chosen_sub = st.selectbox(
                "Sub-issue",
                options=ALL_SUBISSUES,
                index=ALL_SUBISSUES.index(current_subissue) if current_subissue in ALL_SUBISSUES else 0,
                key=f"review_sub_{row_idx}",
                label_visibility="collapsed",
            )

        with col_btn:
            btn_label = "Update" if is_decided else "Submit"
            if st.button(btn_label, key=f"review_submit_{row_idx}", use_container_width=True):
                st.session_state.review_decisions[row_idx] = {
                    "issue":    chosen_issue,
                    "subissue": chosen_sub,
                }
                st.rerun()

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Finalise button ────────────────────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    all_done = done_count == total
    if not all_done:
        remaining = total - done_count
        st.caption(f"{remaining} complaint(s) still pending review before you can finalise.")

    col_fin, col_clear = st.columns([3, 1])
    with col_fin:
        if st.button(
            "Finalise Reviews",
            type="primary",
            use_container_width=True,
            disabled=not all_done,
            key="review_finalise_btn",
        ):
            df_with_review = apply_review_decisions(results_df)
            st.session_state.results_with_review = df_with_review
            st.session_state.review_finalised    = True
            st.rerun()

    with col_clear:
        if st.button("↺ Clear All", use_container_width=True, key="review_clear_btn"):
            _reset_review()
            st.rerun()


def _render_finalised_summary(
    results_df: pd.DataFrame,
    review_rows: pd.DataFrame,
    dark_mode: bool,
):
    """Shown after the user clicks Finalise Reviews.

    No accuracy stats here on purpose — once a human has reviewed a
    complaint, they own that call. This shows the FULL final list: every
    complaint's final issue/sub-issue, whether it came straight from the
    model or was corrected by a human, with a badge marking the source and
    the model's original guess shown alongside wherever a human changed it.
    """
    df_rev    = st.session_state.results_with_review
    decisions = st.session_state.review_decisions

    surface = DARK_SURFACE if dark_mode else LIGHT_SURFACE
    border  = "rgba(255,255,255,0.08)" if dark_mode else "rgba(0,0,0,0.08)"
    text    = "#f1f5f9" if dark_mode else "#1e293b"
    muted   = "#94a3b8" if dark_mode else "#64748b"

    total_reviewed = len(decisions)
    total_rows     = len(df_rev)

    st.markdown(f"""
<div style="background:{surface};border:1px solid {border};border-radius:8px;
            padding:14px 18px;margin-bottom:12px;
            display:flex;justify-content:space-between;align-items:center">
  <span style="font-weight:700;font-size:0.95rem;color:{PRIMARY}">Review complete — final list</span>
  <span style="font-size:11px;color:{PRIMARY};font-weight:600">
    ✓ {total_reviewed} of {total_rows} updated by a human reviewer
  </span>
</div>""", unsafe_allow_html=True)

    # Full final list: every complaint's final issue/sub-issue — the
    # model's own answer where it was trusted, the human's correction
    # where it wasn't.
    rows_html = ""
    for row_idx, row in df_rev.iterrows():
        snip = row["complaint_text"]
        snip = snip[:60] + "…" if len(snip) > 60 else snip

        if row["review_source"] == "human":
            model_issue    = row["predicted_issue_broad"]
            model_subissue = row["predicted_subissue"]
            human_issue    = row["reviewed_issue"]
            human_subissue = row["reviewed_subissue"]

            issue_cell = (
                f"<span style='color:{muted};text-decoration:line-through'>{model_issue}</span> "
                f"<span style='color:{muted}'>&rarr;</span> "
                f"<span style='color:{ACCENT};font-weight:600'>{human_issue}</span>"
                if human_issue != model_issue
                else f"<span style='color:{text}'>{human_issue}</span>"
            )
            subissue_cell = (
                f"<span style='color:{muted};text-decoration:line-through'>{model_subissue}</span> "
                f"<span style='color:{muted}'>&rarr;</span> "
                f"<span style='color:{ACCENT};font-weight:600'>{human_subissue}</span>"
                if human_subissue != model_subissue
                else f"<span style='color:{text}'>{human_subissue}</span>"
            )
            badge = (
                f"<span style='background:rgba(99,102,241,0.15);color:{PRIMARY};"
                f"padding:2px 8px;border-radius:4px;font-size:10px;font-weight:600'>Human</span>"
            )
        else:
            issue_cell    = f"<span style='color:{text}'>{row['reviewed_issue']}</span>"
            subissue_cell = f"<span style='color:{text}'>{row['reviewed_subissue']}</span>"
            badge = (
                f"<span style='background:rgba(16,185,129,0.12);color:{ACCENT};"
                f"padding:2px 8px;border-radius:4px;font-size:10px;font-weight:600'>Model</span>"
            )

        rows_html += f"""
<tr>
  <td style="padding:7px 10px;font-size:11px;color:{muted}">{snip}</td>
  <td style="padding:7px 10px;font-size:11px">{issue_cell}</td>
  <td style="padding:7px 10px;font-size:11px">{subissue_cell}</td>
  <td style="padding:7px 10px;text-align:center">{badge}</td>
</tr>"""

    st.markdown(f"""
<div style="overflow:auto;max-height:480px;border-radius:8px;border:1px solid {border};margin-bottom:14px">
  <table style="width:100%;border-collapse:collapse;font-size:11px">
    <thead>
      <tr style="background:{surface};position:sticky;top:0">
        <th style="padding:8px 10px;text-align:left;font-size:10px;
                   text-transform:uppercase;letter-spacing:.05em;color:{muted}">Complaint</th>
        <th style="padding:8px 10px;text-align:left;font-size:10px;
                   text-transform:uppercase;letter-spacing:.05em;color:{muted}">Final issue</th>
        <th style="padding:8px 10px;text-align:left;font-size:10px;
                   text-transform:uppercase;letter-spacing:.05em;color:{muted}">Final sub-issue</th>
        <th style="padding:8px 10px;text-align:center;font-size:10px;
                   text-transform:uppercase;letter-spacing:.05em;color:{muted}">Source</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>""", unsafe_allow_html=True)

    # Export + re-review
    col_dl, col_re = st.columns([3, 1])
    with col_dl:
        csv = df_rev.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV (final list)",
            data=csv,
            file_name="classifier_evaluation_reviewed.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
            key="review_download_btn",
        )
    with col_re:
        if st.button("Re-review", use_container_width=True, key="re_review_btn"):
            _reset_review()
            st.rerun()