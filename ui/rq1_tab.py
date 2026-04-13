import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from analysis.correlation import CorrelationAnalyzer


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

    # ── Pull global data from session state ─────────────────────────────────
    main_global   = st.session_state.get('df_full')
    aspect_global = st.session_state.get('aspect_df_full')

    if main_global is None or len(main_global) == 0:
        st.warning("No global sentiment data found. Please run sentiment analysis first.")
        return

    if aspect_global is None or len(aspect_global) == 0:
        st.warning("No global aspect data found. Please run aspect analysis first.")
        return

    # Resolve sentiment column name
    if 'sentiment' not in main_global.columns:
        st.warning("Sentiment column not found in global data.")
        return

    # Resolve aspect sentiment column name
    asp_sent_col = (
        'aspect_sentiment'
        if 'aspect_sentiment' in aspect_global.columns
        else 'sentiment'
    )

    # ── Build the long-format dataframe CorrelationAnalyzer expects ─────────
    # Merge aspect rows onto the main df so each row has both whole and aspect sentiment.
    # CorrelationAnalyzer.create_long_format() expects an 'aspects_found' + 'aspect_data'
    # structure (LLM pipeline), so we build the long df directly here from the already-
    # exploded aspect_df instead.

    main_slim = main_global[['review', 'sentiment']].drop_duplicates('review').copy()
    main_slim = main_slim.rename(columns={'sentiment': 'whole_sentiment'})

    aspect_slim = aspect_global[['review', asp_sent_col]].copy()
    aspect_slim = aspect_slim.rename(columns={asp_sent_col: 'aspect_sentiment'})

    long_df = aspect_slim.merge(main_slim, on='review', how='left')
    long_df = long_df.dropna(subset=['whole_sentiment', 'aspect_sentiment'])

    if len(long_df) == 0:
        st.warning("Could not match aspect data to whole-text sentiment. Check that both analyses ran correctly.")
        return

    total_reviews = long_df['review'].nunique()
    total_aspect_mentions = len(long_df)

    # ── Metric row ───────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Reviews (global)", total_reviews)
    col2.metric("Aspect mentions", total_aspect_mentions)
    col3.metric(
        "Avg aspects / review",
        f"{total_aspect_mentions / total_reviews:.2f}" if total_reviews > 0 else "—"
    )

    st.divider()

    # ── ODR ──────────────────────────────────────────────────────────────────
    st.markdown("## 📐 Overall Disagreement Rate (ODR)")
    st.caption(
        "ODR = proportion of reviews where **at least one** aspect sentiment differs from "
        "the whole-text sentiment. A high ODR means the document-level label misses "
        "important nuance captured at the aspect level."
    )

    # Inline ODR calculation (mirrors CorrelationAnalyzer.calculate_disagreement_rate)
    long_df = long_df.reset_index(drop=True)
    long_df['review_id'] = long_df.groupby('review').ngroup()

    def has_disagreement(group):
        whole = group['whole_sentiment'].iloc[0]
        return any(group['aspect_sentiment'] != whole)

    review_disagree = (
        long_df.groupby('review_id')
        .apply(has_disagreement)
        .reset_index()
    )
    review_disagree.columns = ['review_id', 'has_disagreement']

    n_disagree = int(review_disagree['has_disagreement'].sum())
    odr        = n_disagree / total_reviews if total_reviews > 0 else 0

    odr_col1, odr_col2 = st.columns([1, 2])

    with odr_col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(odr * 100, 1),
            number={'suffix': '%'},
            title={'text': "ODR"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': '#F44336'},
                'steps': [
                    {'range': [0,  30], 'color': '#C8E6C9'},
                    {'range': [30, 60], 'color': '#FFF9C4'},
                    {'range': [60,100], 'color': '#FFCDD2'},
                ],
                'threshold': {
                    'line':  {'color': '#B71C1C', 'width': 4},
                    'thickness': 0.75,
                    'value': round(odr * 100, 1),
                },
            },
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with odr_col2:
        st.markdown(f"### {odr*100:.1f}% of reviews show disagreement")
        st.markdown(
            f"Out of **{total_reviews}** reviews, **{n_disagree}** have at least one aspect "
            f"whose sentiment does not match the overall review sentiment."
        )

        if odr < 0.30:
            st.success(
                "✅ **Low disagreement** — whole-text sentiment is a reliable proxy for "
                "aspect-level sentiment in this dataset."
            )
        elif odr < 0.60:
            st.warning(
                "⚠️ **Moderate disagreement** — aspect-level analysis reveals meaningful "
                "nuance not captured by document-level sentiment alone."
            )
        else:
            st.error(
                "🔴 **High disagreement** — whole-text sentiment frequently masks conflicting "
                "aspect sentiments. Aspect-level analysis is essential for this dataset."
            )

    st.divider()

    # ── Confusion matrix ─────────────────────────────────────────────────────
    st.markdown("## 🗂️ Confusion Matrix — Whole-Text vs Aspect Sentiment")
    st.caption(
        "Each cell shows what percentage of aspect mentions belonging to a given whole-text "
        "sentiment category were labelled with a given aspect sentiment. A perfect agreement "
        "would show 100% on the diagonal."
    )

    # Inline confusion matrix (mirrors CorrelationAnalyzer.create_confusion_matrix)
    conf = pd.crosstab(
        long_df['whole_sentiment'],
        long_df['aspect_sentiment'],
        normalize='index'
    ) * 100
    conf = conf.round(1)

    # Ensure consistent column/row ordering
    sentiment_order = [s for s in ['Positive', 'Neutral', 'Negative'] if s in conf.columns or s in conf.index]
    conf = conf.reindex(
        index=[s for s in sentiment_order if s in conf.index],
        columns=[s for s in sentiment_order if s in conf.columns],
        fill_value=0
    )

    # Heatmap
    color_scale = [
        [0.0, '#FFFFFF'],
        [0.5, '#FFCDD2'],
        [1.0, '#B71C1C'],
    ]

    fig_hm = go.Figure(go.Heatmap(
        z=conf.values,
        x=[f"Aspect: {c}" for c in conf.columns],
        y=[f"Whole: {r}" for r in conf.index],
        colorscale=color_scale,
        zmin=0,
        zmax=100,
        text=[[f"{v:.1f}%" for v in row] for row in conf.values],
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title='%', ticksuffix='%'),
    ))
    fig_hm.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(side='top'),
        plot_bgcolor='white',
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # Raw counts table alongside
    with st.expander("📋 View raw counts"):
        conf_counts = pd.crosstab(
            long_df['whole_sentiment'],
            long_df['aspect_sentiment']
        )
        conf_counts = conf_counts.reindex(
            index=[s for s in sentiment_order if s in conf_counts.index],
            columns=[s for s in sentiment_order if s in conf_counts.columns],
            fill_value=0
        )
        st.dataframe(conf_counts, use_container_width=True)

    st.divider()

    # ── Key finding ──────────────────────────────────────────────────────────
    st.markdown("## 🎯 Key Finding")

    # Pull diagonal agreement rate for the narrative
    diag_vals = [conf.loc[s, s] for s in sentiment_order if s in conf.index and s in conf.columns]
    avg_agreement = sum(diag_vals) / len(diag_vals) if diag_vals else 0

    finding = (
        f"Across the full uploaded dataset ({total_reviews} reviews, "
        f"{total_aspect_mentions} aspect mentions), the Overall Disagreement Rate is "
        f"**{odr*100:.1f}%** — meaning {n_disagree} reviews contain at least one aspect "
        f"sentiment that contradicts the whole-text label. "
        f"On average, **{avg_agreement:.1f}%** of aspect mentions within each sentiment "
        f"category agree with the document-level sentiment. "
    )

    if odr < 0.30:
        finding += (
            "This low ODR suggests that document-level sentiment analysis is broadly "
            "sufficient for this dataset, though aspect analysis still surfaces finer-grained insights."
        )
    elif odr < 0.60:
        finding += (
            "This moderate ODR confirms that aspect-level analysis adds meaningful value "
            "beyond whole-text sentiment, particularly for mixed reviews."
        )
    else:
        finding += (
            "This high ODR indicates that whole-text sentiment is an unreliable summary "
            "for this dataset — aspect-level analysis is critical for accurate interpretation."
        )

    st.info(finding)

    # Persist for download tab
    st.session_state['rq1_results'] = {
        'odr':            odr,
        'n_disagree':     n_disagree,
        'total_reviews':  total_reviews,
        'confusion_matrix': conf,
        'finding':        finding,
    }