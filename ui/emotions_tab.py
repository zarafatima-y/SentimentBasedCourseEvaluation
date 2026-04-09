import streamlit as st
import pandas as pd
import plotly.express as px


def render_emotions_tab(config):
    st.markdown("### Emotion Analysis Results")
    st.caption(
        "**How to read:** The emotion distribution chart shows how often each emotion "
        "was detected across reviews. The emotion–sentiment heatmap shows, for each "
        "emotion, how many reviews were Positive, Neutral, or Negative — darker cells "
        "mean higher counts. This lets you see, for example, whether 'joy' co-occurs "
        "mostly with positive sentiment (expected) or whether 'fear' shows up even in "
        "neutral reviews (a subtler signal worth noting)."
    )
    df = st.session_state.df

    if config['type'] == "🔄 Compare Sections (Same Course, Same Year)":
        group_col = 'section'
    elif config['type'] == "📅 Compare Years (Same Course)":
        group_col = 'academic_year'
    elif config['type'] == "🔬 Cross-Course Comparison":
        group_col = 'course_year' if 'course_year' in df.columns else 'course_code'
    else:
        group_col = None

    def _emotion_heatmap(df_slice, title):
        if 'sentiment' not in df_slice.columns:
            return None
        pivot = pd.crosstab(df_slice['dominant_emotion'], df_slice['sentiment'])
        for col in ['Positive', 'Neutral', 'Negative']:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot[['Positive', 'Neutral', 'Negative']]
        fig = px.imshow(
            pivot,
            text_auto=True,
            color_continuous_scale='Teal',
            title=title,
            labels=dict(x='Sentiment', y='Emotion', color='Count'),
            aspect='auto',
        )
        fig.update_layout(height=350, coloraxis_showscale=False)
        return fig

    if group_col is not None:
        col1, col2 = st.columns(2)
        with col1:
            emo_grp = pd.crosstab(df[group_col], df['dominant_emotion'])
            if len(emo_grp) > 0:
                fig = px.bar(
                    emo_grp.reset_index(), x=group_col,
                    y=list(emo_grp.columns),
                    title=f'Emotion Distribution by {group_col}',
                    barmode='group'
                )
                fig.update_layout(xaxis_tickangle=-20)
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            hm = _emotion_heatmap(df, 'Emotion × Sentiment (all groups combined)')
            if hm:
                st.plotly_chart(hm, use_container_width=True)

        st.markdown("#### Emotion–Sentiment Breakdown per Group")
        grp_vals_emo = sorted(df[group_col].dropna().unique())
        emo_cols = st.columns(min(len(grp_vals_emo), 3))
        for i, gv in enumerate(grp_vals_emo):
            gv_df = df[df[group_col] == gv]
            hm = _emotion_heatmap(gv_df, str(gv))
            if hm:
                emo_cols[i % 3].plotly_chart(hm, use_container_width=True)

    else:
        col1, col2 = st.columns(2)
        with col1:
            ec = df['dominant_emotion'].value_counts().reset_index()
            ec.columns = ['Emotion', 'Count']
            fig = px.pie(ec, values='Count', names='Emotion',
                         title='Dominant Emotion Distribution',
                         color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            hm = _emotion_heatmap(df, 'Emotion × Sentiment Heatmap')
            if hm:
                st.plotly_chart(hm, use_container_width=True)
