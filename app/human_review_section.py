import pandas as pd
import streamlit as st

DARK_SURFACE  = "#1e293b"
LIGHT_SURFACE = "#ffffff"

PRIMARY       = "#818cf8"
PRIMARY_LIGHT = "#6366f1"
ACCENT        = "#10b981"
DANGER        = "#dc2626"

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
ALL_SUBISSUES = [sub for subs in ISSUE_SUBISSUE_MAP.values() for sub in subs]
SUBISSUE_TO_ISSUE = {
    sub: issue for issue, subs in ISSUE_SUBISSUE_MAP.items() for sub in subs
}


def _ensure_review_state():
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
      <span style="background:rgba(129,140,248,0.12);color:{PRIMARY};
                   padding:2px 8px;border-radius:4px;font-weight:600;margin-left:4px">
        {model_subissue}
      </span>
    </div>
    <div style="font-size:10px;color:{muted}">
      Confidence:
      <span style="background:rgba(129,140,248,0.12);color:{PRIMARY};
                   padding:2px 8px;border-radius:4px;font-weight:600;margin-left:4px">
        {conf_pct}%
      </span>
    </div>
  </div>
</div>"""


def apply_review_decisions(results_df: pd.DataFrame) -> pd.DataFrame:
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

    df.loc[is_human, "issue_correct"]    = True
    df.loc[is_human, "subissue_correct"] = True

    df.loc[~is_human, "issue_correct"]    = df.loc[~is_human, "reviewed_issue"]    == df.loc[~is_human, "true_issue"]
    df.loc[~is_human, "subissue_correct"] = df.loc[~is_human, "reviewed_subissue"] == df.loc[~is_human, "true_subissue"]

    return df


def render_review_queue(results_df: pd.DataFrame, dark_mode: bool):
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

    if st.session_state.review_finalised:
        _render_finalised_summary(results_df, review_rows, dark_mode)
        return

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

        with col_issue:
            chosen_issue = st.selectbox(
                "Broad issue",
                options=ALL_ISSUES,
                index=ALL_ISSUES.index(current_issue) if current_issue in ALL_ISSUES else 0,
                key=f"review_issue_{row_idx}",
                label_visibility="collapsed",
            )

        available_subissues = ISSUE_SUBISSUE_MAP[chosen_issue]

        with col_sub:
            sub_index = 0
            if current_subissue in available_subissues:
                sub_index = available_subissues.index(current_subissue)
            
            chosen_sub = st.selectbox(
                "Sub-issue",
                options=available_subissues,
                index=sub_index,
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
                f"<span style='background:rgba(129,140,248,0.15);color:{PRIMARY};"
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