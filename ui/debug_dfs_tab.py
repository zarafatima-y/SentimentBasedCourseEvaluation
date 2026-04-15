import streamlit as st
import pandas as pd


def _show_df_block(name: str, df: pd.DataFrame | None, note: str) -> None:
    st.markdown(f"#### {name}")
    st.caption(note)

    if df is None:
        st.warning(f"{name} is not present in `st.session_state`.")
        return

    rows, cols = df.shape
    col1, col2 = st.columns(2)
    col1.metric("Rows", rows)
    col2.metric("Columns", cols)

    with st.expander(f"Columns in {name}", expanded=False):
        st.write(list(df.columns))

    with st.expander(f"Preview {name}", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)


def render_debug_dfs_tab() -> None:
    st.markdown("### Debug DataFrames")
    st.caption(
        "Use this tab to inspect the session-level DataFrames and verify which one "
        "RQ1 should use."
    )

    df_current = st.session_state.get("df")
    df_clean = st.session_state.get("df_clean")
    df_full = st.session_state.get("df_full")
    filtered_df = st.session_state.get("filtered_df")
    aspect_df = st.session_state.get("aspect_df")
    aspect_df_full = st.session_state.get("aspect_df_full")
    llm_ready_full = st.session_state.get("llm_ready_full")
    analysis_long = st.session_state.get("analysis_long")
    numeric_df = st.session_state.get("numeric_df")

    _show_df_block(
        "df",
        df_current,
        "Current working DataFrame. This gets overwritten as the app moves from raw data to filtered analysis results.",
    )
    _show_df_block(
        "df_clean",
        df_clean,
        "Cleaned full dataset saved right after preprocessing. No stage-3 filtering yet.",
    )
    _show_df_block(
        "df_full",
        df_full,
        "Global full dataset after sentiment analysis. RQ1 uses this for whole-text sentiment.",
    )
    _show_df_block(
        "filtered_df",
        filtered_df,
        "Stage-3 filtered subset before it is written back into `df`.",
    )
    _show_df_block(
        "aspect_df",
        aspect_df,
        "Aspect rows for the current stage-3 selection only.",
    )
    _show_df_block(
        "aspect_df_full",
        aspect_df_full,
        "Global aspect rows for all cleaned reviews. RQ1 uses this for aspect sentiment.",
    )
    _show_df_block(
        "llm_ready_full",
        llm_ready_full,
        "Notebook-style LLM-ready dataframe used directly by RQ1.",
    )
    _show_df_block(
        "analysis_long",
        analysis_long,
        "Merged long-format table used by several downstream visualizations.",
    )
    _show_df_block(
        "numeric_df",
        numeric_df,
        "Numeric survey extraction from the uploaded PDFs.",
    )

    st.divider()
    st.markdown("### RQ1 Input Check")

    if llm_ready_full is None or len(llm_ready_full) == 0:
        st.warning("`llm_ready_full` is missing or empty, so RQ1 cannot use it yet.")
        return
    if "Sentiment_Label" not in llm_ready_full.columns or "aspect_data" not in llm_ready_full.columns:
        st.warning("`llm_ready_full` needs `Sentiment_Label` and `aspect_data` for RQ1.")
        return
    llm_ready_preview = llm_ready_full.copy()
    llm_ready_preview["aspect_data"] = llm_ready_preview["aspect_data"].astype(str)

    col1, col2, col3 = st.columns(3)
    col1.metric("RQ1 source", "llm_ready_full")
    col2.metric("Rows", len(llm_ready_full))
    col3.metric("Rows with aspects", int((llm_ready_full["aspect_data"].astype(str) != "{}").sum()))

    st.info("RQ1 now uses `llm_ready_full` directly, without joining `df_full` and `aspect_df_full` inside the tab.")

    with st.expander("Preview `llm_ready_full` used by RQ1", expanded=True):
        st.dataframe(llm_ready_preview.head(50), use_container_width=True)
