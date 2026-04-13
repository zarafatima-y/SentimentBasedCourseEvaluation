import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render_rq1_tab():
    st.markdown("### 🔗 RQ1: How Well Does Whole-Text Sentiment Agree With Aspect Sentiment?")

    st.markdown(
        "**Research Question:** To what extent does the overall (whole-text) sentiment of a "
        "student review agree with the sentiment expressed at the individual aspect level? "
        "Following Nashihin et al. (2025), disagreement is measured via the **Overall "
        "Disagreement Rate (ODR)** and a **confusion matrix** of whole-text vs aspect sentiment."
    )

    st.caption(
        "This analysis always runs on the **full uploaded dataset** — it is independent of "
        "whichever course / section / year comparison you selected in Stage 3."
    )

    st.divider()

    # ── Pull global data ─────────────────────────────────────────────────────
    main_global   = st.session_state.get('df_full')
    aspect_global = st.session_state.get('aspect_df_full')

    if main_global is None or len(main_global) == 0:
        st.warning("No global sentiment data found. Please run sentiment analysis first.")
        return
    if aspect_global is None or len(aspect_global) == 0:
        st.warning("No global aspect data found. Please run aspect analysis first.")
        return
    if 'sentiment' not in main_global.columns:
        st.warning("Sentiment column not found in global data.")
        return

    # ── Resolve aspect sentiment column ──────────────────────────────────────
    asp_sent_col = (
        'aspect_sentiment'
        if 'aspect_sentiment' in aspect_global.columns
        else 'sentiment'
    )

    # ── Build cor_aspect_df exactly as the notebook does ────────────────────
    # main_global  : one row per review, has 'review' + 'sentiment'
    # aspect_global: one row per aspect mention, has 'review' + asp_sent_col
    # We merge so each aspect row carries the whole-text sentiment,
    # then normalise both columns to Title Case so labels always match.

    main_slim = (
        main_global[['review', 'sentiment']]
        .drop_duplicates('review')
        .copy()
        .rename(columns={'sentiment': 'whole_sentiment'})
    )
    main_slim['whole_sentiment'] = main_slim['whole_sentiment'].str.strip().str.title()

    aspect_slim = aspect_global[['review', asp_sent_col]].copy()
    aspect_slim = aspect_slim.rename(columns={asp_sent_col: 'aspect_sentiment'})
    aspect_slim['aspect_sentiment'] = aspect_slim['aspect_sentiment'].str.strip().str.title()

    cor_aspect_df = aspect_slim.merge(main_slim, on='review', how='left')
    cor_aspect_df = cor_aspect_df.dropna(subset=['whole_sentiment', 'aspect_sentiment'])

    # Assign a stable integer review_id (one id per unique review text)
    review_ids = (
        cor_aspect_df[['review']]
        .drop_duplicates()
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={'index': 'review_id'})
    )
    cor_aspect_df = cor_aspect_df.merge(review_ids, on='review', how='left')

    if len(cor_aspect_df) == 0:
        st.warning("Could not match aspect data to whole-text sentiment. Check that both analyses ran correctly.")
        return

    total_reviews         = cor_aspect_df['review_id'].nunique()
    total_aspect_mentions = len(cor_aspect_df)

    # ── Top metrics ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Reviews (global)", total_reviews)
    col2.metric("Aspect mentions",  total_aspect_mentions)
    col3.metric(
        "Avg aspects / review",
        f"{total_aspect_mentions / total_reviews:.2f}" if total_reviews > 0 else "—"
    )

    st.divider()

    # ── ODR — exact notebook logic ───────────────────────────────────────────
    st.markdown("## 📐 Overall Disagreement Rate (ODR)")
    st.caption(
        "ODR = proportion of reviews where **at least one** aspect sentiment differs from "
        "the whole-text sentiment (Nashihin et al. 2025). A high ODR means the document-level "
        "label misses important nuance captured at the aspect level."
    )

    def has_disagreement(group):
        return any(group['aspect_sentiment'] != group['whole_sentiment'].iloc[0])

    review_disagreement = (
        cor_aspect_df
        .groupby('review_id', group_keys=False)[['aspect_sentiment', 'whole_sentiment']]
        .apply(has_disagreement)
        .reset_index()
    )
    review_disagreement.columns = ['review_id', 'has_disagreement']

    n_disagree = int(review_disagreement['has_disagreement'].sum())
    odr        = n_disagree / total_reviews if total_reviews > 0 else 0

    # Thresholds match the notebook exactly
    if odr > 0.70:
        nuance_level = "High"
        nuance_desc  = "Method captures significant additional nuance."
        alert_fn     = st.error
        alert_icon   = "🔴"
    elif odr > 0.40:
        nuance_level = "Moderate"
        nuance_desc  = "Method adds meaningful nuance beyond whole-text sentiment."
        alert_fn     = st.warning
        alert_icon   = "⚠️"
    elif odr > 0.20:
        nuance_level = "Low"
        nuance_desc  = "Method adds some nuance."
        alert_fn     = st.info
        alert_icon   = "🔵"
    else:
        nuance_level = "Minimal"
        nuance_desc  = "Method largely reproduces whole-text sentiment."
        alert_fn     = st.success
        alert_icon   = "✅"

    odr_col1, odr_col2 = st.columns([1, 2])

    with odr_col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(odr * 100, 1),
            number={'suffix': '%', 'font': {'size': 36}},
            title={'text': "ODR", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': '#F44336'},
                'steps': [
                    {'range': [0,  20], 'color': '#C8E6C9'},
                    {'range': [20, 40], 'color': '#DCEDC8'},
                    {'range': [40, 70], 'color': '#FFF9C4'},
                    {'range': [70,100], 'color': '#FFCDD2'},
                ],
                'threshold': {
                    'line':      {'color': '#B71C1C', 'width': 4},
                    'thickness': 0.75,
                    'value':     round(odr * 100, 1),
                },
            },
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with odr_col2:
        st.markdown(f"### {odr*100:.1f}% of reviews show disagreement")
        st.markdown(
            f"Out of **{total_reviews}** reviews, **{n_disagree}** have at least one aspect "
            f"whose sentiment does not match the overall review sentiment."
        )
        alert_fn(f"{alert_icon} **{nuance_level} disagreement** — {nuance_desc}")
        st.markdown(
            "_Following Nashihin et al. (2025): this quantifies the additional nuance "
            "captured by aspect-based analysis._"
        )

    st.divider()

    # ── Confusion matrix — exact notebook logic ──────────────────────────────
    st.markdown("## 🗂️ Confusion Matrix — Whole-Text vs Aspect Sentiment")
    st.caption(
        "Each cell shows what % of aspect mentions belonging to a given whole-text sentiment "
        "were labelled with a given aspect sentiment. "
        "Diagonal = agreement; off-diagonal = disagreement."
    )

    conf_pct = pd.crosstab(
        cor_aspect_df['whole_sentiment'],
        cor_aspect_df['aspect_sentiment'],
        normalize='index'
    ) * 100

    conf_counts = pd.crosstab(
        cor_aspect_df['whole_sentiment'],
        cor_aspect_df['aspect_sentiment'],
    )

    # Enforce Negative / Neutral / Positive ordering to match notebook
    order    = [s for s in ['Negative', 'Neutral', 'Positive']
                if s in conf_pct.index or s in conf_pct.columns]
    conf_pct    = conf_pct.reindex(index=order, columns=order, fill_value=0).round(1)
    conf_counts = conf_counts.reindex(index=order, columns=order, fill_value=0)

    # RdYlGn_r palette — dark green = agreement, dark red = disagreement (notebook cmap)
    colorscale = [
        [0.0,  '#006837'],
        [0.25, '#31a354'],
        [0.50, '#FFFFBF'],
        [0.75, '#d7301f'],
        [1.0,  '#7f0000'],
    ]

    annotations = []
    for i, row_label in enumerate(conf_pct.index):
        for j, col_label in enumerate(conf_pct.columns):
            val = conf_pct.loc[row_label, col_label]
            cnt = (
                conf_counts.loc[row_label, col_label]
                if row_label in conf_counts.index and col_label in conf_counts.columns
                else 0
            )
            annotations.append(dict(
                x=j, y=i,
                text=f"<b>{val:.1f}%</b><br><span style='font-size:10px'>n={cnt}</span>",
                showarrow=False,
                font=dict(size=13, color='black'),
            ))

    fig_hm = go.Figure(go.Heatmap(
        z=conf_pct.values,
        x=[f"Aspect: {c}" for c in conf_pct.columns],
        y=[f"Whole: {r}"  for r in conf_pct.index],
        colorscale=colorscale,
        zmin=0,
        zmid=50,
        zmax=100,
        showscale=True,
        colorbar=dict(title='%', ticksuffix='%'),
        hovertemplate=(
            "Whole-text: %{y}<br>"
            "Aspect: %{x}<br>"
            "Percentage: %{z:.1f}%<extra></extra>"
        ),
    ))
    fig_hm.update_layout(
        annotations=annotations,
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(side='top'),
        plot_bgcolor='white',
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.caption("Darker green = strong agreement (diagonal). Darker red = disagreement (off-diagonal).")

    with st.expander("📋 View raw counts table"):
        st.dataframe(conf_counts, use_container_width=True)

    st.divider()

    # ── Key finding — mirrors notebook summary paragraph ─────────────────────
    st.markdown("## 🎯 Key Finding")

    diag_vals     = [conf_pct.loc[s, s] for s in order if s in conf_pct.index and s in conf_pct.columns]
    avg_agreement = sum(diag_vals) / len(diag_vals) if diag_vals else 0

    # Neutral-row nuance check — mirrors notebook's final paragraph
    neutral_nuance = ""
    if 'Neutral' in conf_pct.index:
        off_diag_neutral = conf_pct.loc['Neutral'].drop('Neutral', errors='ignore').sum()
        if off_diag_neutral > 40:
            neutral_nuance = (
                " Most notably, when whole-text sentiment is Neutral, aspects frequently "
                "register as Positive or Negative — confirming that students express mixed "
                "sentiments that average to neutrality."
            )

    finding = (
        f"Following the disagreement analysis framework of Nashihin et al. (2025), "
        f"we calculated an Overall Disagreement Rate (ODR) of **{odr*100:.1f}%** "
        f"between whole-text and aspect-level sentiment across {total_reviews} reviews "
        f"({total_aspect_mentions} aspect mentions). "
        f"This indicates that in {odr*100:.1f}% of reviews, at least one aspect sentiment "
        f"diverges from the global classification — revealing nuanced, multi-dimensional "
        f"feedback that whole-text analysis alone would mask. "
        f"On average, **{avg_agreement:.1f}%** of aspect mentions within each sentiment "
        f"category agree with the document-level label."
        f"{neutral_nuance}"
    )

    st.info(finding)

    # Persist for download tab
    st.session_state['rq1_results'] = {
        'odr':              odr,
        'n_disagree':       n_disagree,
        'total_reviews':    total_reviews,
        'confusion_pct':    conf_pct,
        'confusion_counts': conf_counts,
        'finding':          finding,
    }