import streamlit as st
import pandas as pd


def render_download_tab(config):
    st.markdown("### Download Results")

    aspect_df = st.session_state.aspect_df

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Full Datasets")
        if st.session_state.df is not None:
            st.download_button("📥 Download Full Analysis (CSV)",
                               data=st.session_state.df.to_csv(index=False),
                               file_name="full_analysis.csv", mime="text/csv",
                               use_container_width=True)
        if aspect_df is not None and len(aspect_df) > 0:
            st.download_button("📥 Download Aspect Analysis (CSV)",
                               data=aspect_df.to_csv(index=False),
                               file_name="aspect_analysis.csv", mime="text/csv",
                               use_container_width=True)
        if st.session_state.analysis_long is not None:
            st.download_button("📥 Download Long Format (CSV)",
                               data=st.session_state.analysis_long.to_csv(index=False),
                               file_name="analysis_long.csv", mime="text/csv",
                               use_container_width=True)
        if st.session_state.numeric_df is not None and len(st.session_state.numeric_df) > 0:
            st.download_button("📥 Download Numeric Ratings (CSV)",
                               data=st.session_state.numeric_df.to_csv(index=False),
                               file_name="numeric_ratings.csv", mime="text/csv",
                               use_container_width=True)

    with col2:
        st.markdown("#### 📈 Summary Report")
        df = st.session_state.df
        summary = {
            'Analysis Type': config['type'],
            'Total Reviews': len(df),
            'Courses': df['course_code'].nunique(),
            'Years': df['academic_year'].nunique(),
            'Sections': df['section'].nunique(),
        }
        if config['type'] == "📚 Single Course Analysis":
            summary['Course'] = config['course']
            summary['Year']   = config['year']
        if 'sentiment' in df.columns:
            sc = df['sentiment'].value_counts().to_dict()
            summary['Positive Reviews'] = sc.get('Positive', 0)
            summary['Neutral Reviews']  = sc.get('Neutral', 0)
            summary['Negative Reviews'] = sc.get('Negative', 0)
        if aspect_df is not None:
            summary['Total Aspect Mentions'] = len(aspect_df)
            summary['Unique Aspects']        = aspect_df['aspect'].nunique()
        if 'dominant_emotion' in df.columns:
            summary['Most Common Emotion'] = df['dominant_emotion'].mode()[0]

        summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.download_button("📥 Download Summary Report (CSV)",
                           data=summary_df.to_csv(index=False),
                           file_name="summary_report.csv", mime="text/csv",
                           use_container_width=True)
