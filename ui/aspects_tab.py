import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from analysis.visualization import Visualizer
from ui.helpers import build_aspect_heatmap


def render_aspects_tab(config, viz_options):
    st.markdown("### Aspect-Based Analysis")
    adf = st.session_state.aspect_df

    aspect_df = st.session_state.aspect_df
    sentiment_col = (
        'aspect_sentiment'
        if aspect_df is not None and 'aspect_sentiment' in aspect_df.columns
        else 'sentiment'
    )

    if config['type'] == "📚 Single Course Analysis":
        col1, col2 = st.columns(2)
        st.caption(
                "📊 **Raw counts — not normalised.** Years with more student enrolment will "
                "show higher bars. Cross-reference with the heatmap to see the sentiment "
                "makeup of each count."
            )
        with col1:
            counts = adf['aspect'].value_counts().reset_index()
            counts.columns = ['Aspect', 'Count']
            fig = px.bar(counts, x='Aspect', y='Count',
                         title=f'Most Discussed Aspects — {config["course"]}',
                         color='Count', color_continuous_scale='Viridis')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Aspect Sentiment Distribution")
            asp_sent = pd.crosstab(adf['aspect'], adf[sentiment_col]).reset_index()
            ycols = [c for c in adf[sentiment_col].unique() if c in asp_sent.columns]
            fig = px.bar(asp_sent, x='aspect', y=ycols,
                         title='Sentiment by Aspect', barmode='stack')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        if viz_options.get('radar'):
            st.markdown("#### 🕸️ Aspect Radar — Sections Compared")
            st.caption(
                "**How to read these charts:** The *Aspect Counts* radar shows how many times "
                "each aspect was mentioned — larger area = more discussion. Use the *Positive %* "
                "and *Negative %* radars to see the sentiment breakdown of those mentions. "
                "For example, if 'difficulty' has a large count but high negative %, students "
                "are talking about difficulty and finding it problematic."
            )
            groups = sorted(adf['section'].unique().tolist())
            if len(groups) >= 1:
                rc1, rc2, rc3 = st.columns(3)
                fig_counts, _ = Visualizer.radar_from_aspect_df(
                    adf, 'section', sentiment_col, groups,
                    mode='counts',
                    title=f"{config['course']} ({config['year']}) — Aspect Counts"
                )
                fig_pos, _ = Visualizer.radar_from_aspect_df(
                    adf, 'section', sentiment_col, groups,
                    mode='pos_pct',
                    title=f"{config['course']} ({config['year']}) — Positive %"
                )
                fig_neg, _ = Visualizer.radar_from_aspect_df(
                    adf, 'section', sentiment_col, groups,
                    mode='neg_pct',
                    title=f"{config['course']} ({config['year']}) — Negative %"
                )
                if fig_counts:
                    rc1.pyplot(fig_counts)
                    plt.close(fig_counts)
                if fig_pos:
                    rc2.pyplot(fig_pos)
                    plt.close(fig_pos)
                if fig_neg:
                    rc3.pyplot(fig_neg)
                    plt.close(fig_neg)

        if viz_options.get('heatmaps'):
            st.markdown("#### Aspect–Sentiment Heatmap (by Section)")
            st.caption(
                    "🗺️ **How to read:** Each row is a section-aspect pair. Columns show "
                    "how many of that aspect's mentions were Negative, Neutral, or Positive. "
                    "Use this alongside the bar chart — if an aspect has a high count on the "
                    "left but most cells land in the Negative column here, that is a priority concern."
                )
            hm_fig = build_aspect_heatmap(
                adf, 'section', sentiment_col,
                title=f'Aspect Sentiment by Section — {config["course"]} ({config["year"]})'
            )
            st.pyplot(hm_fig)
            plt.close(hm_fig)

    elif config['type'] == "🔄 Compare Sections (Same Course, Same Year)":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Aspect Frequency by Section")
            st.caption(
                "📊 **Raw counts — not normalised.** Years with more student enrolment will "
                "show higher bars. Cross-reference with the heatmap to see the sentiment "
                "makeup of each count."
            )
            asp_by_sec = pd.crosstab(adf['aspect'], adf['section'])
            if len(asp_by_sec) > 0:
                ycols = [c for c in config['sections'] if c in asp_by_sec.columns]
                fig = px.bar(asp_by_sec.reset_index(), x='aspect', y=ycols,
                             title='Aspect Mentions by Section', barmode='group')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if viz_options.get('heatmaps'):
                st.markdown("#### Aspect–Sentiment Heatmap by Section")
                st.caption(
                    "🗺️ **How to read:** Each row is a section-aspect pair. Columns show "
                    "how many of that aspect's mentions were Negative, Neutral, or Positive. "
                    "Use this alongside the bar chart — if an aspect has a high count on the "
                    "left but most cells land in the Negative column here, that is a priority concern."
                )
                hm_fig = build_aspect_heatmap(
                    adf, 'section', sentiment_col,
                    title=f'Aspect Sentiment by Section — {config["course"]} ({config["year"]})'
                )
                st.pyplot(hm_fig)
                plt.close(hm_fig)

        if viz_options.get('radar'):
            st.markdown("#### 🕸️ Aspect Radar — All Sections Combined")
            st.caption(
                "**How to read these charts:** The *Aspect Counts* radar shows how many times "
                "each aspect was mentioned — larger area = more discussion. Use the *Positive %* "
                "and *Negative %* radars to see the sentiment breakdown of those mentions. "
                "For example, if 'difficulty' has a large count but high negative %, students "
                "are talking about difficulty and finding it problematic."
            )
            rc1, rc2, rc3 = st.columns(3)
            fig_counts, _ = Visualizer.radar_from_aspect_df(
                adf, 'section', sentiment_col, config['sections'],
                mode='counts',
                title=f"{config['course']} ({config['year']}) — Aspect Counts"
            )
            fig_pos, _ = Visualizer.radar_from_aspect_df(
                adf, 'section', sentiment_col, config['sections'],
                mode='pos_pct',
                title=f"{config['course']} ({config['year']}) — Positive %"
            )
            fig_neg, _ = Visualizer.radar_from_aspect_df(
                adf, 'section', sentiment_col, config['sections'],
                mode='neg_pct',
                title=f"{config['course']} ({config['year']}) — Negative %"
            )
            if fig_counts:
                rc1.pyplot(fig_counts)
                plt.close(fig_counts)
            if fig_pos:
                rc2.pyplot(fig_pos)
                plt.close(fig_pos)
            if fig_neg:
                rc3.pyplot(fig_neg)
                plt.close(fig_neg)

    elif config['type'] == "📅 Compare Years (Same Course)":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Aspect Frequency by Year")
            st.caption(
                "📊 **Raw counts — not normalised.** Years with more student enrolment will "
                "show higher bars. Cross-reference with the heatmap to see the sentiment "
                "makeup of each count."
            )
            asp_by_yr = pd.crosstab(adf['aspect'], adf['academic_year'])
            if len(asp_by_yr) > 0:
                ycols = [c for c in config['years'] if c in asp_by_yr.columns]
                fig = px.bar(asp_by_yr.reset_index(), x='aspect', y=ycols,
                             title='Aspect Mentions by Year', barmode='group')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if viz_options.get('heatmaps'):
                st.markdown("#### Aspect–Sentiment Heatmap by Year")
                st.caption(
                    "🗺️ **How to read:** Each row is a section-aspect pair. Columns show "
                    "how many of that aspect's mentions were Negative, Neutral, or Positive. "
                    "Use this alongside the bar chart — if an aspect has a high count on the "
                    "left but most cells land in the Negative column here, that is a priority concern."
                )
                hm_fig = build_aspect_heatmap(
                    adf, 'academic_year', sentiment_col,
                    title=f'Aspect Sentiment by Year — {config["course"]}'
                )
                st.pyplot(hm_fig)
                plt.close(hm_fig)

        if viz_options.get('radar'):
            st.markdown("#### 🕸️ Aspect Radar — All Years Combined")
            st.caption(
                "**How to read these charts:** The *Aspect Counts* radar shows how many times "
                "each aspect was mentioned — larger area = more discussion. Use the *Positive %* "
                "and *Negative %* radars to see the sentiment breakdown of those mentions. "
                "For example, if 'difficulty' has a large count but high negative %, students "
                "are talking about difficulty and finding it problematic."
                "Note: A shrinking negative % over years on an aspect = improvement over time."
            )
            yr_groups = [str(y) for y in config['years']]
            rc1, rc2, rc3 = st.columns(3)
            fig_counts, _ = Visualizer.radar_from_aspect_df(
                adf, 'academic_year', sentiment_col, yr_groups,
                mode='counts',
                title=f"{config['course']} — Aspect Counts by Year"
            )
            fig_pos, _ = Visualizer.radar_from_aspect_df(
                adf, 'academic_year', sentiment_col, yr_groups,
                mode='pos_pct',
                title=f"{config['course']} — Positive % by Year"
            )
            fig_neg, _ = Visualizer.radar_from_aspect_df(
                adf, 'academic_year', sentiment_col, yr_groups,
                mode='neg_pct',
                title=f"{config['course']} — Negative % by Year"
            )
            if fig_counts:
                rc1.pyplot(fig_counts)
                plt.close(fig_counts)
            if fig_pos:
                rc2.pyplot(fig_pos)
                plt.close(fig_pos)
            if fig_neg:
                rc3.pyplot(fig_neg)
                plt.close(fig_neg)

    elif config['type'] == "🔬 Cross-Course Comparison":
        grp = 'course_year' if 'course_year' in adf.columns else 'course_code'
        if 'course_year' not in adf.columns and 'course_year' in st.session_state.df.columns:
            cy_map = (
                st.session_state.df[['review', 'course_year']]
                .drop_duplicates('review')
                .set_index('review')['course_year']
            )
            adf = adf.copy()
            adf['course_year'] = adf['review'].map(cy_map)
            adf = adf.dropna(subset=['course_year'])
            grp = 'course_year'

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Aspect Frequency by Course")
            st.caption(
                "📊 **Raw counts — not normalised.** Years with more student enrolment will "
                "show higher bars. Cross-reference with the heatmap to see the sentiment "
                "makeup of each count."
            )
            asp_by_cy = pd.crosstab(adf['aspect'], adf[grp])
            if len(asp_by_cy) > 0:
                fig = px.bar(asp_by_cy.reset_index(), x='aspect',
                             y=asp_by_cy.columns.tolist(),
                             title='Aspect Mentions by Course', barmode='group')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if viz_options.get('heatmaps'):
                st.markdown("#### Aspect–Sentiment Heatmap by Course")
                st.caption(
                    "🗺️ **How to read:** Each row is a section-aspect pair. Columns show "
                    "how many of that aspect's mentions were Negative, Neutral, or Positive. "
                    "Use this alongside the bar chart — if an aspect has a high count on the "
                    "left but most cells land in the Negative column here, that is a priority concern."
                )
                hm_fig = build_aspect_heatmap(
                    adf, grp, sentiment_col,
                    title='Aspect Sentiment by Course'
                )
                st.pyplot(hm_fig)
                plt.close(hm_fig)

        if viz_options.get('radar'):
            st.markdown("#### 🕸️ Aspect Radar — All Courses Combined")
            st.caption(
                "**How to read these charts:** The *Aspect Counts* radar shows how many times "
                "each aspect was mentioned — larger area = more discussion. Use the *Positive %* "
                "and *Negative %* radars to see the sentiment breakdown of those mentions. "
                "For example, if 'difficulty' has a large count but high negative %, students "
                "are talking about difficulty and finding it problematic."
            )
            cy_groups = sorted(adf[grp].dropna().unique().tolist())
            rc1, rc2, rc3 = st.columns(3)
            fig_counts, _ = Visualizer.radar_from_aspect_df(
                adf, grp, sentiment_col, cy_groups,
                mode='counts', title='Aspect Counts by Course'
            )
            fig_pos, _ = Visualizer.radar_from_aspect_df(
                adf, grp, sentiment_col, cy_groups,
                mode='pos_pct', title='Positive % by Course'
            )
            fig_neg, _ = Visualizer.radar_from_aspect_df(
                adf, grp, sentiment_col, cy_groups,
                mode='neg_pct', title='Negative % by Course'
            )
            if fig_counts:
                rc1.pyplot(fig_counts)
                plt.close(fig_counts)
            if fig_pos:
                rc2.pyplot(fig_pos)
                plt.close(fig_pos)
            if fig_neg:
                rc3.pyplot(fig_neg)
                plt.close(fig_neg)

    al = st.session_state.analysis_long
    if al is not None and 'question_text' in al.columns:
        st.divider()
        st.markdown("### 📋 Breakdown by Survey Question")
        st.caption(
            "Sentiment and top aspects grouped by the question students "
            "were answering — each group shown separately."
        )

        s_col_al = 'aspect_sentiment' if 'aspect_sentiment' in al.columns else 'sentiment'
        al = al.copy()
        al['q_short'] = al['question_text'].str.strip().str[:80]

        if config['type'] == "🔄 Compare Sections (Same Course, Same Year)":
            split_col   = 'section'
            split_vals  = config['sections']
        elif config['type'] == "📅 Compare Years (Same Course)":
            split_col   = 'academic_year'
            split_vals  = config['years']
        elif config['type'] == "🔬 Cross-Course Comparison":
            split_col   = 'course_year' if 'course_year' in st.session_state.df.columns else 'course_code'
            if split_col not in al.columns:
                cy_map = (
                    st.session_state.df[['review', split_col]]
                    .drop_duplicates('review')
                    .set_index('review')[split_col]
                )
                al[split_col] = al['review'].map(cy_map).fillna(al.get('course_code', ''))
            split_vals = al[split_col].dropna().unique().tolist()
        else:
            split_col  = None
            split_vals = [None]

        def build_q_table(subset_al):
            rows = []
            for q in sorted(subset_al['q_short'].dropna().unique()):
                q_sub = subset_al[subset_al['q_short'] == q].dropna(subset=[s_col_al])
                total = len(q_sub)
                if total == 0:
                    continue
                pos_pct = round((q_sub[s_col_al] == 'Positive').sum() / total * 100, 1)
                neg_pct = round((q_sub[s_col_al] == 'Negative').sum() / total * 100, 1)
                neu_pct = round(100 - pos_pct - neg_pct, 1)
                dom_sent = (
                    'Positive' if pos_pct >= neg_pct and pos_pct >= neu_pct
                    else 'Negative' if neg_pct >= pos_pct and neg_pct >= neu_pct
                    else 'Neutral'
                )
                top2 = q_sub['aspect'].value_counts().head(2).index.tolist()
                top2_neg = (
                    q_sub[q_sub[s_col_al] == 'Negative']['aspect']
                    .value_counts().head(2).index.tolist()
                )
                rows.append({
                    'Question':              q,
                    'Responses':             total,
                    'Dominant Sentiment':    dom_sent,
                    'Positive %':            pos_pct,
                    'Negative %':            neg_pct,
                    'Top Aspects':           ', '.join(top2)     if top2     else '—',
                    'Top Negative Aspects':  ', '.join(top2_neg) if top2_neg else '—',
                })
            return pd.DataFrame(rows)

        def colour_sentiment(val):
            colours = {
                'Positive': 'background-color: #d4edda; color: #155724',
                'Negative': 'background-color: #f8d7da; color: #721c24',
                'Neutral':  'background-color: #fff3cd; color: #856404',
            }
            return colours.get(val, '')

        for val in split_vals:
            if split_col is None:
                subset = al
                header = f"{config['course']} ({config['year']})"
            else:
                subset = al[al[split_col] == val]
                header = str(val)

            q_df = build_q_table(subset)
            if len(q_df) == 0:
                continue

            st.markdown(f"**{header}**")
            styled = q_df.style.map(
                colour_sentiment, subset=['Dominant Sentiment']
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.markdown("")
