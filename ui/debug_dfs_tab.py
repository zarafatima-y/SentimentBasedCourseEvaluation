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

    if df_full is None or len(df_full) == 0:
        st.warning("`df_full` is missing or empty, so RQ1 cannot use it yet.")
        return
    if aspect_df_full is None or len(aspect_df_full) == 0:
        st.warning("`aspect_df_full` is missing or empty, so RQ1 cannot use it yet.")
        return
    if "review" not in df_full.columns or "sentiment" not in df_full.columns:
        st.warning("`df_full` needs `review` and `sentiment` columns for RQ1.")
        return
    if "review" not in aspect_df_full.columns:
        st.warning("`aspect_df_full` needs a `review` column for RQ1.")
        return

    asp_sent_col = (
        "aspect_sentiment"
        if "aspect_sentiment" in aspect_df_full.columns
        else "sentiment"
        if "sentiment" in aspect_df_full.columns
        else None
    )
    if asp_sent_col is None:
        st.warning("`aspect_df_full` needs either `aspect_sentiment` or `sentiment` for RQ1.")
        return

    main_slim = (
        df_full[["review", "sentiment"]]
        .drop_duplicates("review")
        .copy()
        .rename(columns={"sentiment": "whole_sentiment"})
    )
    main_slim["whole_sentiment"] = main_slim["whole_sentiment"].astype(str).str.strip().str.title()

    if "review_id" in df_full.columns and "review_id" in aspect_df_full.columns:
        main_slim = (
            df_full[["review_id", "sentiment"]]
            .drop_duplicates("review_id")
            .copy()
            .rename(columns={"sentiment": "whole_sentiment"})
        )
        main_slim["whole_sentiment"] = main_slim["whole_sentiment"].astype(str).str.strip().str.title()

        aspect_cols = ["review_id", "review", asp_sent_col]
        if "aspect" in aspect_df_full.columns:
            aspect_cols.append("aspect")
        if "confidence" in aspect_df_full.columns:
            aspect_cols.append("confidence")

        aspect_slim = aspect_df_full[aspect_cols].copy()
        aspect_slim = aspect_slim.rename(columns={asp_sent_col: "aspect_sentiment"})
        aspect_slim["aspect_sentiment"] = aspect_slim["aspect_sentiment"].astype(str).str.strip().str.title()

        cor_aspect_df = aspect_slim.merge(main_slim, on="review_id", how="left")
        join_mode = "review_id"
    else:
        aspect_slim = aspect_df_full[["review", asp_sent_col]].copy()
        aspect_slim = aspect_slim.rename(columns={asp_sent_col: "aspect_sentiment"})
        aspect_slim["aspect_sentiment"] = aspect_slim["aspect_sentiment"].astype(str).str.strip().str.title()

        cor_aspect_df = aspect_slim.merge(main_slim, on="review", how="left")
        cor_aspect_df = cor_aspect_df.dropna(subset=["whole_sentiment", "aspect_sentiment"])

        review_ids = (
            cor_aspect_df[["review"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "review_id"})
        )
        cor_aspect_df = cor_aspect_df.merge(review_ids, on="review", how="left")
        join_mode = "review text fallback"

    cor_aspect_df = cor_aspect_df.dropna(subset=["whole_sentiment", "aspect_sentiment"])

    col1, col2, col3 = st.columns(3)
    col1.metric("RQ1 whole-text source", "df_full")
    col2.metric("RQ1 aspect source", "aspect_df_full")
    col3.metric("Merged rows", len(cor_aspect_df))

    st.caption(f"Current RQ1 join mode: `{join_mode}`")

    st.info(
        "RQ1 should use `df_full` plus `aspect_df_full`, not the filtered `df` or "
        "`aspect_df`, because RQ1 is defined over the full uploaded dataset."
    )

    with st.expander("Preview exact RQ1 merged DataFrame (`cor_aspect_df`)", expanded=True):
        st.dataframe(cor_aspect_df.head(50), use_container_width=True)
