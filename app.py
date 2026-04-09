import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile
import os
import sys
import time
from pathlib import Path
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all your modules
from data.loader import PDFLoader
from data.preprocessor import DataPreprocessor
from models.sentiment import SentimentAnalyzer
from models.aspect import AspectAnalyzer
from models.emotion import EmotionAnalyzer
from utils.helpers import clean_review_for_merge, save_dataframes
from utils.constants import ASPECT_KEYWORDS, EMOTION_TO_SENTIMENT

# UI tab modules
from ui.overview_tab import render_overview_tab
from ui.sentiment_tab import render_sentiment_tab
from ui.aspects_tab import render_aspects_tab
from ui.emotions_tab import render_emotions_tab
from ui.llm_tab import render_llm_tab
from ui.numeric_tab import render_numeric_tab
from ui.rq2_tab import render_rq2_tab
from ui.download_tab import render_download_tab


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Course Evaluation Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #2563EB;
    }
    .comparison-badge {
        background-color: #EFF6FF;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563EB;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_session_state():
    defaults = {
        'stage': 'upload',
        'df': None,
        'numeric_df': None,
        'aspect_df': None,
        'analysis_long': None,
        'processed_files': [],
        'processing_time': {},
        'viz_options': {'heatmaps': True, 'radar': True},
        'analysis_config': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://www.eapasa.co.za/wp-content/uploads/2023/03/customer-reviews-line-icon-vector.jpeg", width=80)
    st.markdown("## Navigation")

    stages = ["🍄 Upload", "🍄 Clean", "🍄 Analyze", "🍄 Results"]
    current_stage_idx = ["upload", "preprocess", "analyze", "results"].index(st.session_state.stage)

    for i, stage in enumerate(stages):
        if i < current_stage_idx:
            st.success(f"✅ {stage}")
        elif i == current_stage_idx:
            st.info(f"🔄 {stage}")
        else:
            st.write(f"⏳ {stage}")

    st.divider()

    if st.session_state.df is not None:
        st.markdown("### 📁 Current Data")
        st.write(f"Rows: {len(st.session_state.df)}")
        st.write(f"Reviews: {st.session_state.df['review'].nunique()}")
        if 'sentiment' in st.session_state.df.columns:
            st.write(f"Sentiment: {st.session_state.df['sentiment'].value_counts().to_dict()}")

    st.divider()

    if st.button("🔄 Start Over", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.markdown("<h1 class='main-header'>💻 Course Evaluation Analysis System</h1>", unsafe_allow_html=True)
st.markdown("Upload course evaluation PDFs for comprehensive sentiment, aspect, and emotion analysis")

# ===========================================================================
# STAGE 1: UPLOAD
# ===========================================================================

if st.session_state.stage == 'upload':
    st.markdown("<h2 class='sub-header'>📤 Upload PDF Files</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Choose PDF files to analyze",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload course evaluation PDFs containing 'ESSAY RESULTS' sections"
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) selected")
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size/1024:.1f} KB)")

    with col2:
        st.markdown("### 📋 Instructions")
        st.info(
            """
            1. Upload PDF files with course evaluations
            2. Files should contain 'ESSAY RESULTS' sections
            3. Each evaluation should have:
               - Course code (EECS XXXX)
               - Academic year
               - Section
               - Student comments
            """
        )
        with st.expander("📎 Expected Format"):
            st.code("""
Essay Results for: Course EECS 2021
Academic Year: 2023
Section(s): A, B

1) What did you like?
- The instructor was great
- Good course content

2) Suggestions for improvement?
- More examples in lectures
            """)

    if uploaded_files and st.button("Process PDFs", type="primary", use_container_width=True):
        with st.spinner("Extracting text from PDFs..."):
            start_time = time.time()
            loader = PDFLoader()
            all_dfs = []
            progress_bar = st.progress(0)

            for i, uploaded_file in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                df = loader.load_pdf(tmp_path)
                all_dfs.append(df)
                st.session_state.processed_files.append(uploaded_file.name)
                os.unlink(tmp_path)
                progress_bar.progress((i + 1) / len(uploaded_files))

            if all_dfs:
                st.session_state.df = pd.concat(all_dfs, ignore_index=True)
                st.session_state.processing_time['extraction'] = time.time() - start_time

                # Also extract numeric results from the same PDFs
                numeric_dfs = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    try:
                        num_df = loader.load_numeric_pdf(tmp_path)
                        if len(num_df) > 0:
                            numeric_dfs.append(num_df)
                    except Exception as e:
                        st.warning(f"⚠️ Could not extract numeric data: {e}")
                    finally:
                        os.unlink(tmp_path)

                if numeric_dfs:
                    st.session_state.numeric_df = pd.concat(numeric_dfs, ignore_index=True)

                st.success(f"Extracted {len(st.session_state.df)} reviews from {len(uploaded_files)} files")
                st.session_state.stage = 'preprocess'
                st.rerun()

# ===========================================================================
# STAGE 2: PREPROCESS
# ===========================================================================

elif st.session_state.stage == 'preprocess':
    st.markdown("<h2 class='sub-header'> Data Cleaning & Preprocessing</h2>", unsafe_allow_html=True)

    if st.session_state.df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(st.session_state.df))
        with col2:
            st.metric("Unique Reviews", st.session_state.df['review'].nunique())
        with col3:
            st.metric("Courses", st.session_state.df['course_code'].nunique())

        with st.expander("🔍 Preview Raw Data", expanded=True):
            st.dataframe(st.session_state.df.head(10), use_container_width=True)

        st.markdown("### Data Cleaning Options")

        col1, col2 = st.columns(2)
        with col1:
            remove_null = st.checkbox("Remove null/empty reviews", value=True)
            clean_questions = st.checkbox("Clean question text", value=True)
        with col2:
            remove_short = st.checkbox("Remove very short reviews (< 3 chars)", value=True)
            normalize_text = st.checkbox("Normalize text (lowercase, strip)", value=True)

        if st.button("Run Preprocessing", type="primary", use_container_width=True):
            with st.spinner("Cleaning data..."):
                start_time = time.time()
                preprocessor = DataPreprocessor()

                if clean_questions:
                    st.session_state.df = preprocessor.clean_question_text(st.session_state.df)

                st.session_state.df = preprocessor.clean_reviews(st.session_state.df)

                if remove_null:
                    st.session_state.df = preprocessor.remove_null_reviews(st.session_state.df)

                if remove_short:
                    st.session_state.df = st.session_state.df[
                        st.session_state.df['review'].str.len() >= 3
                    ].reset_index(drop=True)

                st.session_state.processing_time['cleaning'] = time.time() - start_time
                st.success(f"‼️ Cleaning complete — {len(st.session_state.df)} reviews remaining")
                st.session_state.stage = 'analyze'
                st.rerun()
    else:
        st.warning("No data loaded. Please upload PDFs first.")

# ===========================================================================
# STAGE 3: ANALYZE
# ===========================================================================

elif st.session_state.stage == 'analyze':
    st.markdown("<h2 class='sub-header'>🤖 Analysis Configuration</h2>", unsafe_allow_html=True)

    if st.session_state.df is not None:

        st.markdown("### 📊 What would you like to analyze?")

        analysis_type = st.radio(
            "Select Analysis Type:",
            [
                "📚 Single Course Analysis",
                "🔄 Compare Sections (Same Course, Same Year)",
                "📅 Compare Years (Same Course)",
                "🔬 Cross-Course Comparison",
            ],
            help="Choose the kind of analysis you want to perform",
            key="analysis_type_selector"
        )

        st.divider()

        config = {
            'type':     analysis_type,
            'course':   None,
            'year':     None,
            'sections': [],
            'years':    [],
            'courses':  [],
        }

        if analysis_type == "📚 Single Course Analysis":
            st.markdown("### 🎯 Select Course and Year")

            col1, col2 = st.columns(2)
            with col1:
                available_courses = sorted(st.session_state.df['course_code'].unique())
                config['course'] = st.selectbox("Choose Course", available_courses, key="single_course")

            with col2:
                course_years = sorted(
                    st.session_state.df[
                        st.session_state.df['course_code'] == config['course']
                    ]['academic_year'].unique().tolist()
                )
                config['year'] = st.selectbox("Select Year", course_years, key="single_year")

            available_sections = sorted(st.session_state.df[
                (st.session_state.df['course_code'] == config['course']) &
                (st.session_state.df['academic_year'] == config['year'])
            ]['section'].unique().tolist())

            st.info(
                f"📌 **{config['course']} ({config['year']})** has "
                f"**{len(available_sections)}** section(s): {', '.join(available_sections)}. "
                "All sections will be pooled together for this analysis."
            )

        elif analysis_type == "🔄 Compare Sections (Same Course, Same Year)":
            st.markdown("### 🔍 Select Course, Year, and Sections to Compare")

            col1, col2 = st.columns(2)
            with col1:
                available_courses = sorted(st.session_state.df['course_code'].unique())
                config['course'] = st.selectbox("Select Course", available_courses, key="compare_course")

            with col2:
                course_years = sorted(
                    st.session_state.df[
                        st.session_state.df['course_code'] == config['course']
                    ]['academic_year'].unique().tolist()
                )
                config['year'] = st.selectbox("Select Year", course_years, key="compare_year")

            available_sections = sorted(
                st.session_state.df[
                    (st.session_state.df['course_code'] == config['course']) &
                    (st.session_state.df['academic_year'] == config['year'])
                ]['section'].unique().tolist()
            )

            if len(available_sections) < 2:
                st.warning(
                    f"⚠️ **{config['course']} ({config['year']})** only has "
                    f"{len(available_sections)} section — need at least 2 for section comparison."
                )
            else:
                config['sections'] = st.multiselect(
                    f"Choose sections to compare — min 2, max 5 "
                    f"({len(available_sections)} available)",
                    available_sections,
                    default=available_sections[:2],
                    key="compare_sections"
                )

                n = len(config['sections'])
                if n < 2:
                    st.warning("⚠️ Please select at least 2 sections.")
                elif n > 5:
                    st.warning("⚠️ Maximum 5 sections allowed. Please deselect some.")
                    config['sections'] = config['sections'][:5]
                else:
                    st.success(f"✅ Will compare {n} sections: {', '.join(config['sections'])}")

        elif analysis_type == "📅 Compare Years (Same Course)":
            st.markdown("### 📅 Select Course and Years to Compare")

            available_courses = sorted(st.session_state.df['course_code'].unique())
            config['course'] = st.selectbox("Select Course", available_courses, key="years_course")

            course_years = sorted(
                st.session_state.df[
                    st.session_state.df['course_code'] == config['course']
                ]['academic_year'].unique().tolist()
            )

            if len(course_years) < 2:
                st.warning(
                    f"⚠️ **{config['course']}** only has data for {len(course_years)} year — "
                    "need at least 2 for year comparison."
                )
            else:
                config['years'] = st.multiselect(
                    f"Choose years to compare — min 2, max 5 "
                    f"({len(course_years)} available)",
                    course_years,
                    default=course_years[:2],
                    key="compare_years"
                )

                n = len(config['years'])
                if n < 2:
                    st.warning("⚠️ Please select at least 2 years.")
                elif n > 5:
                    st.warning("⚠️ Maximum 5 years allowed. Please deselect some.")
                    config['years'] = config['years'][:5]
                else:
                    st.success(
                        f"✅ Will compare {n} years: {', '.join(map(str, config['years']))} "
                        "(all sections within each year are pooled)"
                    )

        elif analysis_type == "🔬 Cross-Course Comparison":
            st.markdown("### 🔬 Select Courses to Compare")
            st.caption("Compare any 2–5 courses regardless of whether they are related. Each course is identified by its code + year.")

            combos = (
                st.session_state.df[['course_code', 'academic_year']]
                .drop_duplicates()
                .sort_values(['course_code', 'academic_year'])
            )
            combo_labels = [
                f"{r['course_code']} ({r['academic_year']})"
                for _, r in combos.iterrows()
            ]
            combo_tuples = [
                (r['course_code'], r['academic_year'])
                for _, r in combos.iterrows()
            ]
            label_to_tuple = dict(zip(combo_labels, combo_tuples))

            selected_labels = st.multiselect(
                "Choose course–year combinations to compare (min 2, max 5)",
                combo_labels,
                default=combo_labels[:min(2, len(combo_labels))],
                key="cross_course_select"
            )

            n = len(selected_labels)
            if n < 2:
                st.warning("⚠️ Please select at least 2 course–year combinations.")
            elif n > 5:
                st.warning("⚠️ Maximum 5 combinations allowed.")
                selected_labels = selected_labels[:5]
            else:
                config['courses'] = [label_to_tuple[l] for l in selected_labels]
                st.success(f"✅ Comparing: {', '.join(selected_labels)}")

        st.divider()

        st.markdown("### ⚙️ Analysis Modules")
        st.markdown("Select which analyses to run:")

        col1, col2, col3 = st.columns(3)
        with col1:
            run_sentiment = st.checkbox("📊 Sentiment Analysis", value=True)
        with col2:
            run_aspect = st.checkbox("🔍 Aspect Analysis", value=True)
        with col3:
            run_emotion = st.checkbox("😊 Emotion Analysis", value=True)

        if run_aspect:
            with st.expander("🔍 Aspects that will be detected"):
                aspect_cols = st.columns(3)
                for i, (aspect, keywords) in enumerate(ASPECT_KEYWORDS.items()):
                    aspect_cols[i % 3].markdown(f"**{aspect}**")
                    aspect_cols[i % 3].caption(f"e.g., {', '.join(keywords[:3])}")

        st.markdown("### 📈 Visualization Options")
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            show_heatmaps = st.checkbox("Show Heatmaps", value=True)
        with viz_col2:
            show_radar = st.checkbox("Show Radar Charts", value=True)

        st.divider()

        st.markdown("### 📋 Analysis Summary")

        if analysis_type == "📚 Single Course Analysis" and config['course'] and config['year']:
            st.info(f"📊 Will analyze: **{config['course']}** ({config['year']}) — all sections combined")

        elif analysis_type == "🔄 Compare Sections (Same Course, Same Year)" and len(config.get('sections', [])) >= 2:
            st.info(
                f"🔄 Comparing sections **{', '.join(config['sections'])}** of "
                f"**{config['course']}** ({config['year']})"
            )

        elif analysis_type == "📅 Compare Years (Same Course)" and len(config.get('years', [])) >= 2:
            st.info(
                f"📅 Comparing years **{', '.join(map(str, config['years']))}** of "
                f"**{config['course']}** (all sections pooled per year)"
            )

        elif analysis_type == "🔬 Cross-Course Comparison" and len(config.get('courses', [])) >= 2:
            labels = [f"{c} ({y})" for c, y in config['courses']]
            st.info(f"🔬 Cross-course: **{' vs '.join(labels)}**")

        analyses_to_run = (
            (["Sentiment"] if run_sentiment else []) +
            (["Aspect"] if run_aspect else []) +
            (["Emotion"] if run_emotion else [])
        )
        st.markdown(f"**Analyses to run:** {', '.join(analyses_to_run)}")

        can_run = False
        if analysis_type == "📚 Single Course Analysis" and config['course'] and config['year']:
            can_run = True
        elif analysis_type == "🔄 Compare Sections (Same Course, Same Year)" and 2 <= len(config.get('sections', [])) <= 5:
            can_run = True
        elif analysis_type == "📅 Compare Years (Same Course)" and 2 <= len(config.get('years', [])) <= 5:
            can_run = True
        elif analysis_type == "🔬 Cross-Course Comparison" and 2 <= len(config.get('courses', [])) <= 5:
            can_run = True

        if can_run and st.button("🚀 Run Selected Analysis", type="primary", use_container_width=True):
            results = {}
            start_total = time.time()

            progress_bar = st.progress(0)
            status_text = st.empty()

            st.session_state.analysis_config = config
            st.session_state.run_options = {
                'sentiment': run_sentiment,
                'aspect': run_aspect,
                'emotion': run_emotion,
            }
            st.session_state.viz_options = {
                'heatmaps': show_heatmaps,
                'radar': show_radar
            }

            filtered_df = st.session_state.df.copy()

            if analysis_type == "📚 Single Course Analysis":
                filtered_df = filtered_df[
                    (filtered_df['course_code'] == config['course']) &
                    (filtered_df['academic_year'] == config['year'])
                ]
            elif analysis_type == "🔄 Compare Sections (Same Course, Same Year)":
                filtered_df = filtered_df[
                    (filtered_df['course_code'] == config['course']) &
                    (filtered_df['academic_year'] == config['year']) &
                    (filtered_df['section'].isin(config['sections']))
                ]
            elif analysis_type == "📅 Compare Years (Same Course)":
                filtered_df = filtered_df[
                    (filtered_df['course_code'] == config['course']) &
                    (filtered_df['academic_year'].isin(config['years']))
                ]
            elif analysis_type == "🔬 Cross-Course Comparison":
                mask = pd.Series(False, index=filtered_df.index)
                for course_code, year in config['courses']:
                    mask |= (
                        (filtered_df['course_code'] == course_code) &
                        (filtered_df['academic_year'] == year)
                    )
                filtered_df = filtered_df[mask]
                filtered_df = filtered_df.copy()
                filtered_df['course_year'] = (
                    filtered_df['course_code'] + ' (' +
                    filtered_df['academic_year'].astype(str) + ')'
                )

            st.session_state.filtered_df = filtered_df

            if run_sentiment:
                status_text.text("📊 Running sentiment analysis...")
                start_time = time.time()
                sentiment = SentimentAnalyzer()
                results['sentiment'] = sentiment.analyze(filtered_df['review'].tolist())
                filtered_df = pd.concat(
                    [filtered_df.reset_index(drop=True), results['sentiment'].reset_index(drop=True)],
                    axis=1
                )
                st.session_state.processing_time['sentiment'] = time.time() - start_time
                progress_bar.progress(0.25)

            if run_aspect:
                status_text.text("🔍 Extracting aspects and analyzing sentiment...")
                start_time = time.time()
                filtered_df['review'] = filtered_df['review'].fillna('').astype(str)

                aspect = AspectAnalyzer()
                st.session_state.aspect_df = aspect.analyze_all(filtered_df)

                if len(st.session_state.aspect_df) > 0:
                    review_to_course  = {}
                    review_to_year    = {}
                    review_to_section = {}

                    for _, row in filtered_df.iterrows():
                        key = str(row['review']).strip().lower()
                        review_to_course[key]  = row['course_code']
                        review_to_year[key]    = row['academic_year']
                        review_to_section[key] = row['section']

                    adf = st.session_state.aspect_df
                    keys = adf['review'].str.strip().str.lower()
                    adf['course_code']   = keys.map(review_to_course).fillna('Unknown')
                    adf['academic_year'] = keys.map(review_to_year).fillna('Unknown')
                    adf['section']       = keys.map(review_to_section).fillna('Unknown')

                    if 'review_clean' not in adf.columns:
                        adf['review_clean'] = keys

                    st.session_state.aspect_df = adf
                else:
                    st.warning("⚠️ No aspects were detected in the reviews.")
                    st.session_state.aspect_df = pd.DataFrame(
                        columns=['review', 'aspect', 'sentiment', 'confidence',
                                 'method', 'course_code', 'academic_year', 'section', 'review_clean']
                    )

                st.session_state.processing_time['aspect'] = time.time() - start_time
                progress_bar.progress(0.5)

            if run_emotion:
                status_text.text("😊 Analyzing emotions...")
                start_time = time.time()
                emotion = EmotionAnalyzer()
                results['emotion'] = emotion.analyze(filtered_df['review'].tolist())
                filtered_df = pd.concat(
                    [filtered_df.reset_index(drop=True), results['emotion'].reset_index(drop=True)],
                    axis=1
                )
                st.session_state.processing_time['emotion'] = time.time() - start_time
                progress_bar.progress(0.75)

            st.session_state.df = filtered_df

            if run_aspect and st.session_state.aspect_df is not None and len(st.session_state.aspect_df) > 0:
                status_text.text("🔄 Creating analysis datasets...")
                df_clean        = clean_review_for_merge(st.session_state.df)
                aspect_df_clean = clean_review_for_merge(st.session_state.aspect_df)

                df_clean = df_clean.rename(columns={'sentiment': 'whole_sentiment'})
                if 'aspect_sentiment' not in aspect_df_clean.columns:
                    aspect_df_clean = aspect_df_clean.rename(columns={'sentiment': 'aspect_sentiment'})

                st.session_state.analysis_long = df_clean.merge(
                    aspect_df_clean[['review_clean', 'aspect', 'aspect_sentiment', 'confidence']],
                    on='review_clean',
                    how='left'
                )
                if 'question_text' in st.session_state.df.columns and \
                   'question_text' not in st.session_state.analysis_long.columns:
                    qt_map = (
                        st.session_state.df
                        .assign(review_clean=st.session_state.df['review'].str.strip().str.lower())
                        .drop_duplicates('review_clean')[['review_clean', 'question_text']]
                    )
                    st.session_state.analysis_long = st.session_state.analysis_long.merge(
                        qt_map, on='review_clean', how='left'
                    )

            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")
            st.session_state.processing_time['total'] = time.time() - start_total

            adf_len = len(st.session_state.aspect_df) if st.session_state.aspect_df is not None else 0
            st.success(f"""
            ✅ Analysis complete!
            - Time: {st.session_state.processing_time['total']:.2f} seconds
            - Reviews analyzed: {len(st.session_state.df)}
            - Aspects found: {adf_len}
            """)

            st.session_state.stage = 'results'
            st.rerun()

        elif not can_run:
            st.info("ℹ️ Complete your selection above to enable the Run button.")

    else:
        st.warning("No data available. Please complete preprocessing first.")

# ===========================================================================
# STAGE 4: RESULTS
# ===========================================================================

elif st.session_state.stage == 'results':
    st.markdown("<h2 class='sub-header'>📈 Analysis Results</h2>", unsafe_allow_html=True)

    if st.session_state.df is not None and st.session_state.analysis_config:
        config      = st.session_state.analysis_config
        run_options = st.session_state.run_options
        viz_options = st.session_state.viz_options

        aspect_df    = st.session_state.aspect_df
        sentiment_col = (
            'aspect_sentiment'
            if aspect_df is not None and 'aspect_sentiment' in aspect_df.columns
            else 'sentiment'
        )

        st.markdown('<div class="comparison-badge">', unsafe_allow_html=True)
        if config['type'] == "📚 Single Course Analysis":
            st.markdown(f"### 📊 **{config['course']}** ({config['year']}) — Complete Course Analysis")
            st.caption(f"All sections combined · {len(st.session_state.df)} reviews")
        elif config['type'] == "🔄 Compare Sections (Same Course, Same Year)":
            st.markdown(f"### 🔄 **Section Comparison:** {config['course']} ({config['year']})")
            st.caption(f"Sections: {', '.join(config['sections'])} · {len(st.session_state.df)} reviews")
        elif config['type'] == "📅 Compare Years (Same Course)":
            st.markdown(f"### 📅 **Year Comparison:** {config['course']}")
            st.caption(
                f"Years: {', '.join(map(str, config['years']))} · "
                f"all sections pooled · {len(st.session_state.df)} reviews"
            )
        elif config['type'] == "🔬 Cross-Course Comparison":
            labels = [f"{c} ({y})" for c, y in config['courses']]
            st.markdown(f"### 🔬 **Cross-Course Comparison:** {' vs '.join(labels)}")
            st.caption(f"{len(st.session_state.df)} total reviews across {len(config['courses'])} course–year combinations")
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        available_tabs = ["📊 Overview"]
        if run_options.get('sentiment') and 'sentiment' in st.session_state.df.columns:
            available_tabs.append("😊 Sentiment")
        if run_options.get('aspect') and aspect_df is not None and len(aspect_df) > 0:
            available_tabs.append("🔍 Aspects")
        if run_options.get('emotion') and 'dominant_emotion' in st.session_state.df.columns:
            available_tabs.append("🎭 Emotions")
        if run_options.get('aspect') and aspect_df is not None and len(aspect_df) > 0:
            available_tabs.append("🤖 LLM Summary")
        if run_options.get('aspect') and aspect_df is not None and len(aspect_df) > 0:
            available_tabs.append("📉 RQ2: Aspect Predictors")
        if st.session_state.numeric_df is not None and len(st.session_state.numeric_df) > 0:
            available_tabs.append("🔢 Numeric Insights")
        available_tabs.append("📥 Download")

        tabs     = st.tabs(available_tabs)
        tab_dict = dict(zip(available_tabs, tabs))

        # ===================================================================
        # TAB: OVERVIEW
        # ===================================================================
        with tab_dict["📊 Overview"]:
            render_overview_tab(config)

        # ===================================================================
        # TAB: SENTIMENT
        # ===================================================================
        if "😊 Sentiment" in tab_dict:
            with tab_dict["😊 Sentiment"]:
                render_sentiment_tab(config)
        # ===================================================================
        # TAB: ASPECTS
        # ===================================================================
        if "🔍 Aspects" in tab_dict:
            with tab_dict["🔍 Aspects"]:
                render_aspects_tab(config, viz_options)
        # ===================================================================
        # TAB: EMOTIONS
        # ===================================================================
        if "🎭 Emotions" in tab_dict:
            with tab_dict["🎭 Emotions"]:
                render_emotions_tab(config)
        # ===================================================================
        # TAB: LLM SUMMARY
        # ===================================================================
        if "🤖 LLM Summary" in tab_dict:
            with tab_dict["🤖 LLM Summary"]:
                render_llm_tab(config)
        # ===================================================================
        # TAB: NUMERIC INSIGHTS
        # ===================================================================
        if "🔢 Numeric Insights" in tab_dict:
            with tab_dict["🔢 Numeric Insights"]:
                render_numeric_tab(config)
        # ===================================================================
        # TAB: RQ2 — ASPECT PREDICTORS OF NEGATIVE EVALUATIONS
        # ===================================================================
        if "📉 RQ2: Aspect Predictors" in tab_dict:
            with tab_dict["📉 RQ2: Aspect Predictors"]:
                render_rq2_tab(config)
        # ===================================================================
        # TAB: DOWNLOAD
        # ===================================================================
        if "📥 Download" in tab_dict:
            with tab_dict["📥 Download"]:
                render_download_tab(config)
        with st.sidebar:
            st.markdown("### 📊 Quick Stats")
            st.metric("Total Reviews", len(st.session_state.df))
            if 'sentiment' in st.session_state.df.columns:
                pos_pct = (st.session_state.df['sentiment'] == 'Positive').mean() * 100
                st.metric("Positive %", f"{pos_pct:.1f}%")
            if aspect_df is not None and len(aspect_df) > 0:
                st.metric("Avg Aspects/Review",
                          f"{len(aspect_df)/len(st.session_state.df):.2f}")
    else:
        st.warning("No results available. Please run analysis first.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    Course Evaluation Analysis System | Built with Streamlit
</div>
""", unsafe_allow_html=True)