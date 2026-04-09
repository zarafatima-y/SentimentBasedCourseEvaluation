import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ui.helpers import COLOR_MAP


def render_sentiment_tab(config):
    st.markdown("### Sentiment Analysis Results")
    df = st.session_state.df

    aspect_df = st.session_state.aspect_df
    sentiment_col = (
        'aspect_sentiment'
        if aspect_df is not None and 'aspect_sentiment' in aspect_df.columns
        else 'sentiment'
    )

    if config['type'] == "📚 Single Course Analysis":
        col1, col2 = st.columns(2)
        with col1:
            counts = df['sentiment'].value_counts().reset_index()
            counts.columns = ['Sentiment', 'Count']
            fig = px.pie(counts, values='Count', names='Sentiment',
                         title=f'Overall Sentiment — {config["course"]} ({config["year"]})',
                         color='Sentiment', color_discrete_map=COLOR_MAP)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Sentiment by Section")
            by_sec = pd.crosstab(df['section'], df['sentiment']).reset_index()
            ycols = [c for c in ['Positive', 'Neutral', 'Negative'] if c in by_sec.columns]
            fig = px.bar(by_sec, x='section', y=ycols,
                         title='Sentiment by Section', barmode='stack',
                         color_discrete_map=COLOR_MAP)
            st.plotly_chart(fig, use_container_width=True)

    elif config['type'] == "🔄 Compare Sections (Same Course, Same Year)":
        st.markdown("#### Sentiment Breakdown per Section")
        st.caption(
            "Each chart shows the sentiment split for that section independently. "
            "Compare the proportions across sections to identify where student "
            "experience differs within the same course and year."
        )
        pie_cols = st.columns(min(len(config['sections']), 3))
        for i, sec in enumerate(sorted(config['sections'])):
            sec_df = df[df['section'] == sec]
            if len(sec_df) == 0:
                continue
            sec_counts = sec_df['sentiment'].value_counts().reset_index()
            sec_counts.columns = ['Sentiment', 'Count']
            fig_sec = px.pie(
                sec_counts, values='Count', names='Sentiment',
                title=f'Section {sec}',
                color='Sentiment', color_discrete_map=COLOR_MAP,
            )
            fig_sec.update_layout(height=300, margin=dict(t=40, b=20))
            pie_cols[i % 3].plotly_chart(fig_sec, use_container_width=True)

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            by_sec = pd.crosstab(df['section'], df['sentiment']).reset_index()
            ycols = [c for c in ['Positive', 'Neutral', 'Negative'] if c in by_sec.columns]
            fig = px.bar(by_sec, x='section', y=ycols,
                         title='Sentiment Count by Section (All Sections)',
                         barmode='group',
                         color_discrete_map=COLOR_MAP)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            pct_rows = []
            for sec in config['sections']:
                sec_df = df[df['section'] == sec]
                total = len(sec_df)
                if total > 0:
                    for sent in ['Positive', 'Neutral', 'Negative']:
                        pct_rows.append({
                            'Section': sec, 'Sentiment': sent,
                            'Percentage': len(sec_df[sec_df['sentiment'] == sent]) / total * 100
                        })
            pct_df = pd.DataFrame(pct_rows)
            if len(pct_df) > 0:
                fig = px.bar(pct_df, x='Section', y='Percentage', color='Sentiment',
                             title='Sentiment % by Section', barmode='stack',
                             color_discrete_map=COLOR_MAP)
                st.plotly_chart(fig, use_container_width=True)

    elif config['type'] == "📅 Compare Years (Same Course)":
        st.markdown("#### Sentiment Breakdown per Year")
        st.caption(
            "Each chart shows the sentiment split for that year independently. "
            "Compare the proportions across years to see how student experience "
            "has shifted over time."
        )
        pie_cols = st.columns(min(len(config['years']), 3))
        for i, yr in enumerate(sorted(config['years'])):
            yr_df = df[df['academic_year'] == yr]
            if len(yr_df) == 0:
                continue
            yr_counts = yr_df['sentiment'].value_counts().reset_index()
            yr_counts.columns = ['Sentiment', 'Count']
            fig_yr = px.pie(
                yr_counts, values='Count', names='Sentiment',
                title=str(yr),
                color='Sentiment', color_discrete_map=COLOR_MAP,
            )
            fig_yr.update_layout(height=300, margin=dict(t=40, b=20))
            pie_cols[i % 3].plotly_chart(fig_yr, use_container_width=True)

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            by_yr = pd.crosstab(df['academic_year'], df['sentiment']).reset_index()
            by_yr['academic_year'] = by_yr['academic_year'].astype(str)
            ycols = [c for c in ['Positive', 'Neutral', 'Negative'] if c in by_yr.columns]
            fig = px.bar(by_yr, x='academic_year', y=ycols,
                         title='Sentiment Count by Year (All Years)',
                         barmode='group',
                         color_discrete_map=COLOR_MAP)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            trend_rows = []
            for yr in config['years']:
                yr_df = df[df['academic_year'] == yr]
                total = len(yr_df)
                if total > 0:
                    for sent in ['Positive', 'Neutral', 'Negative']:
                        trend_rows.append({
                            'Year': str(yr), 'Sentiment': sent,
                            'Percentage': len(yr_df[yr_df['sentiment'] == sent]) / total * 100
                        })
            trend_df = pd.DataFrame(trend_rows)
            if len(trend_df) > 0:
                fig = px.line(trend_df, x='Year', y='Percentage', color='Sentiment',
                              title='Sentiment % Trend Over Years', markers=True,
                              color_discrete_map=COLOR_MAP)
                st.plotly_chart(fig, use_container_width=True)

    elif config['type'] == "🔬 Cross-Course Comparison":
        grp = 'course_year' if 'course_year' in df.columns else 'course_code'
        cy_vals = sorted(df[grp].dropna().unique().tolist())

        st.markdown("#### Sentiment Breakdown per Course")
        st.caption(
            "Each chart shows the sentiment split for that course independently. "
            "Percentages are the fair way to compare across courses since "
            "review volumes may differ."
        )
        pie_cols = st.columns(min(len(cy_vals), 3))
        for i, cy in enumerate(cy_vals):
            cy_df = df[df[grp] == cy]
            if len(cy_df) == 0:
                continue
            cy_counts = cy_df['sentiment'].value_counts().reset_index()
            cy_counts.columns = ['Sentiment', 'Count']
            fig_cy = px.pie(
                cy_counts, values='Count', names='Sentiment',
                title=str(cy),
                color='Sentiment', color_discrete_map=COLOR_MAP,
            )
            fig_cy.update_layout(height=300, margin=dict(t=40, b=20))
            pie_cols[i % 3].plotly_chart(fig_cy, use_container_width=True)

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            by_cy = pd.crosstab(df[grp], df['sentiment']).reset_index()
            ycols = [c for c in ['Positive', 'Neutral', 'Negative'] if c in by_cy.columns]
            fig = px.bar(by_cy, x=grp, y=ycols,
                         title='Sentiment Distribution by Course',
                         barmode='group', color_discrete_map=COLOR_MAP)
            fig.update_layout(xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            pct_rows = []
            for cy in cy_vals:
                cy_df = df[df[grp] == cy]
                total = len(cy_df)
                if total > 0:
                    for sent in ['Positive', 'Neutral', 'Negative']:
                        pct_rows.append({
                            'Course': cy, 'Sentiment': sent,
                            'Percentage': len(cy_df[cy_df['sentiment'] == sent]) / total * 100
                        })
            pct_df = pd.DataFrame(pct_rows)
            if len(pct_df) > 0:
                fig = px.bar(pct_df, x='Course', y='Percentage', color='Sentiment',
                             title='Sentiment % by Course', barmode='stack',
                             color_discrete_map=COLOR_MAP)
                fig.update_layout(xaxis_tickangle=-20)
                st.plotly_chart(fig, use_container_width=True)

    # Aspect Sentiment Balance
    adf_sent = st.session_state.aspect_df
    s_col_sent = sentiment_col

    if adf_sent is not None and len(adf_sent) > 0:
        st.markdown("#### 📊 Aspect Sentiment Balance")
        st.caption(
            "How to read: Each group of bars represents one course aspect. "
            "Green shows the percentage of mentions for that aspect that were positive; "
            "red shows the percentage that were negative. "
            "A long red bar means students raised that topic mainly as a concern. "
            "A long green bar means it is an area students appreciate. "
            "Aspects are sorted from most to least negative so priority areas appear first. "
            "Each group is shown separately so you can compare directly. "
            "Note: green and red will not always add up to 100% — "
            "the gap is neutral mentions, where students raised the topic "
            "without expressing a clear positive or negative view."
        )

        if config['type'] == "🔄 Compare Sections (Same Course, Same Year)":
            grp_col_sb, grp_vals_sb = 'section', config['sections']
        elif config['type'] == "📅 Compare Years (Same Course)":
            grp_col_sb = 'academic_year'
            grp_vals_sb = config['years']
        elif config['type'] == "🔬 Cross-Course Comparison":
            cy_col_sb = 'course_year' if 'course_year' in adf_sent.columns else 'course_code'
            if 'course_year' not in adf_sent.columns and 'course_year' in st.session_state.df.columns:
                cy_map_sb = (
                    st.session_state.df[['review', 'course_year']]
                    .drop_duplicates('review')
                    .set_index('review')['course_year']
                )
                adf_sent = adf_sent.copy()
                adf_sent['course_year'] = adf_sent['review'].map(cy_map_sb)
                adf_sent = adf_sent.dropna(subset=['course_year'])
            grp_col_sb = cy_col_sb
            grp_vals_sb = sorted(adf_sent[cy_col_sb].dropna().unique().tolist())
        else:
            grp_col_sb, grp_vals_sb = None, [None]

        for gv_sb in grp_vals_sb:
            sub_sb = adf_sent[adf_sent[grp_col_sb] == gv_sb] if grp_col_sb else adf_sent
            label_sb = str(gv_sb) if gv_sb is not None else f"{config['course']} ({config['year']})"
            if len(sub_sb) == 0:
                continue

            bal_rows = []
            for asp in sorted(sub_sb['aspect'].dropna().unique()):
                asp_sub = sub_sb[sub_sb['aspect'] == asp]
                total = len(asp_sub)
                if total == 0:
                    continue
                pos_pct = round((asp_sub[s_col_sent] == 'Positive').sum() / total * 100, 1)
                neg_pct = round((asp_sub[s_col_sent] == 'Negative').sum() / total * 100, 1)
                bal_rows.append({'Aspect': asp, 'Positive %': pos_pct, 'Negative %': neg_pct, 'Total': total})

            if not bal_rows:
                continue

            bal_df = pd.DataFrame(bal_rows).sort_values('Negative %', ascending=False)

            st.markdown(f"**{label_sb}**")
            fig_bal = go.Figure()
            fig_bal.add_trace(go.Bar(
                name='Positive %',
                x=bal_df['Aspect'],
                y=bal_df['Positive %'],
                marker_color='#4CAF50',
            ))
            fig_bal.add_trace(go.Bar(
                name='Negative %',
                x=bal_df['Aspect'],
                y=bal_df['Negative %'],
                marker_color='#F44336',
            ))
            fig_bal.update_layout(
                barmode='group',
                height=340,
                margin=dict(t=30, b=60),
                yaxis=dict(title='% of Mentions', range=[0, 100]),
                xaxis=dict(tickangle=-35),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            )
            st.plotly_chart(fig_bal, use_container_width=True)
