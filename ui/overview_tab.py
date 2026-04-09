import streamlit as st
import pandas as pd
import plotly.express as px

from ui.helpers import COLOR_MAP


def render_overview_tab(config):
    st.markdown("### Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", len(st.session_state.df))
    with col2:
        st.metric("Courses", st.session_state.df['course_code'].nunique())
    with col3:
        st.metric("Years", st.session_state.df['academic_year'].nunique())
    with col4:
        st.metric("Sections", st.session_state.df['section'].nunique())

    if config['type'] == "📚 Single Course Analysis":
        col1, col2 = st.columns(2)
        with col1:
            sec_data = st.session_state.df['section'].value_counts().reset_index()
            sec_data.columns = ['Section', 'Count']
            fig = px.pie(sec_data, values='Count', names='Section',
                         title=f'Review Distribution — {config["course"]}')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            stats_df = pd.DataFrame({
                'Metric': ['Avg Review Length', 'Total Words', 'Unique Reviews'],
                'Value': [
                    int(st.session_state.df['review'].str.len().mean()),
                    st.session_state.df['review'].str.split().str.len().sum(),
                    st.session_state.df['review'].nunique()
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

    elif config['type'] == "🔄 Compare Sections (Same Course, Same Year)":
        comp = st.session_state.df.groupby('section').size().reset_index(name='count')
        fig = px.bar(comp, x='section', y='count',
                     title=f'Review Count by Section — {config["course"]} ({config["year"]})',
                     color='section')
        st.plotly_chart(fig, use_container_width=True)

    elif config['type'] == "🔬 Cross-Course Comparison":
        if 'course_year' in st.session_state.df.columns:
            comp = st.session_state.df.groupby('course_year').size().reset_index(name='count')
            fig = px.bar(comp, x='course_year', y='count',
                         title='Review Count by Course–Year',
                         color='course_year')
            st.plotly_chart(fig, use_container_width=True)

    elif config['type'] == "📅 Compare Years (Same Course)":
        comp = st.session_state.df.groupby('academic_year').size().reset_index(name='count')
        comp['academic_year'] = comp['academic_year'].astype(str)
        fig = px.bar(comp, x='academic_year', y='count',
                     title=f'Review Count by Year — {config["course"]}',
                     color='academic_year')
        st.plotly_chart(fig, use_container_width=True)

        al = st.session_state.analysis_long
        if al is not None and len(al) > 0:
            st.divider()
            st.markdown("### 📈 Sentiment & Aspect Trends Over Years")

            df_yr = st.session_state.df

            trend_rows = []
            for yr in sorted(config['years']):
                yr_data = df_yr[df_yr['academic_year'] == yr]
                total   = len(yr_data)
                if total > 0 and 'sentiment' in yr_data.columns:
                    for s in ['Positive', 'Neutral', 'Negative']:
                        trend_rows.append({
                            'Year':       str(yr),
                            'Sentiment':  s,
                            'Percentage': round(
                                len(yr_data[yr_data['sentiment'] == s]) / total * 100, 1
                            )
                        })
            if trend_rows:
                trend_df = pd.DataFrame(trend_rows)
                fig_trend = px.line(
                    trend_df, x='Year', y='Percentage', color='Sentiment',
                    markers=True,
                    title=f'Overall Sentiment Trend — {config["course"]}',
                    color_discrete_map=COLOR_MAP,
                    labels={'Percentage': '% of Reviews'}
                )
                fig_trend.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(fig_trend, use_container_width=True)

            s_col_al = 'aspect_sentiment' if 'aspect_sentiment' in al.columns else 'sentiment'
            aspects_present = al['aspect'].dropna().unique().tolist()

            asp_trend_rows = []
            for yr in sorted(config['years']):
                yr_al = al[al['academic_year'] == yr]
                for asp in aspects_present:
                    asp_sub = yr_al[yr_al['aspect'] == asp]
                    total   = len(asp_sub)
                    if total > 0:
                        neg_pct = round(
                            (asp_sub[s_col_al] == 'Negative').sum() / total * 100, 1
                        )
                        asp_trend_rows.append({
                            'Year':    str(yr),
                            'Aspect':  asp,
                            'Negative %': neg_pct
                        })

            if asp_trend_rows:
                asp_trend_df = pd.DataFrame(asp_trend_rows)
                yr_strs      = [str(y) for y in sorted(config['years'])]
                asp_counts   = asp_trend_df.groupby('Aspect')['Year'].nunique()
                full_aspects = asp_counts[asp_counts == len(yr_strs)].index.tolist()
                asp_trend_df = asp_trend_df[asp_trend_df['Aspect'].isin(full_aspects)]

                if len(asp_trend_df) > 0:
                    fig_asp = px.line(
                        asp_trend_df, x='Year', y='Negative %',
                        color='Aspect', markers=True,
                        title=f'Aspect Negative Sentiment Trend — {config["course"]}',
                        labels={'Negative %': 'Negative Sentiment (%)'}
                    )
                    fig_asp.update_layout(yaxis_range=[0, 100])
                    st.caption(
                        "Rising lines = worsening student experience for that aspect. "
                        "Only aspects present across all selected years are shown."
                    )
                    st.plotly_chart(fig_asp, use_container_width=True)

    with st.expander("🔍 View Sample Data"):
        st.dataframe(
            st.session_state.df[['course_code', 'academic_year', 'section', 'review']].head(10)
        )
