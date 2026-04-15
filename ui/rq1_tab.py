import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


VALID_SENTIMENTS = ["Negative", "Neutral", "Positive"]


def render_rq1_tab():
    st.markdown("### 🔗 RQ1: Whole-Text vs Aspect Sentiment")
    st.caption("This view uses the full uploaded dataset and compares each review's overall sentiment with the sentiments of its detected aspects.")

    main_global = st.session_state.get("df_full")
    aspect_global = st.session_state.get("aspect_df_full")

    if main_global is None or len(main_global) == 0:
        st.warning("No global sentiment data found. Please run sentiment analysis first.")
        return
    if aspect_global is None or len(aspect_global) == 0:
        st.warning("No global aspect data found. Please run aspect analysis first.")
        return
    if "sentiment" not in main_global.columns:
        st.warning("`df_full` must contain a `sentiment` column.")
        return

    asp_sent_col = "aspect_sentiment" if "aspect_sentiment" in aspect_global.columns else "sentiment"
    if asp_sent_col not in aspect_global.columns:
        st.warning("`aspect_df_full` must contain either `aspect_sentiment` or `sentiment`.")
        return

    # Recreate llm_ready_df the same way the notebook did: map grouped aspect dicts by review_clean.
    df = main_global.copy()
    aspect_df = aspect_global.copy()

    df["review"] = df["review"].fillna("").astype(str)
    aspect_df["review"] = aspect_df["review"].fillna("").astype(str)
    df["review_clean"] = df["review"].str.strip().str.lower()
    aspect_df["review_clean"] = aspect_df["review"].str.strip().str.lower()

    aspect_dict = (
        aspect_df.groupby("review_clean", sort=False)
        .apply(
            lambda x: {
                row["aspect"]: {
                    "sentiment": str(row[asp_sent_col]).strip().title(),
                    "confidence": row.get("confidence"),
                }
                for _, row in x.iterrows()
                if pd.notna(row.get("aspect"))
            }
        )
        .to_dict()
    )

    llm_ready_df = df.copy().rename(columns={"sentiment": "Sentiment_Label"})
    llm_ready_df["Sentiment_Label"] = llm_ready_df["Sentiment_Label"].astype(str).str.strip().str.title()
    llm_ready_df["aspect_data"] = llm_ready_df["review_clean"].map(aspect_dict)
    llm_ready_df["aspect_data"] = llm_ready_df["aspect_data"].apply(
        lambda x: x if isinstance(x, dict) else {}
    )
    llm_ready_df["aspects_found"] = llm_ready_df["aspect_data"].apply(lambda x: list(x.keys()))
    llm_ready_df["num_aspects"] = llm_ready_df["aspects_found"].apply(len)

    aspect_rows = []
    for idx, row in llm_ready_df.iterrows():
        for aspect, sentiment_info in row["aspect_data"].items():
            aspect_rows.append(
                {
                    "review_id": idx,
                    "review_text": row["review"],
                    "course_code": row.get("course_code"),
                    "academic_year": row.get("academic_year"),
                    "whole_sentiment": row["Sentiment_Label"],
                    "aspect": aspect,
                    "aspect_sentiment": str(sentiment_info.get("sentiment", "")).strip().title(),
                    "aspect_confidence": sentiment_info.get("confidence"),
                }
            )

    cor_aspect_df = pd.DataFrame(aspect_rows)
    if len(cor_aspect_df) == 0:
        st.warning("No aspect-level rows were generated for RQ1.")
        return

    # Keep the same three-label setup used by the notebook and drop anything unexpected.
    cor_aspect_df["whole_sentiment"] = cor_aspect_df["whole_sentiment"].astype(str).str.strip().str.title()
    cor_aspect_df["aspect_sentiment"] = cor_aspect_df["aspect_sentiment"].astype(str).str.strip().str.title()
    cor_aspect_df = cor_aspect_df[
        cor_aspect_df["whole_sentiment"].isin(VALID_SENTIMENTS)
        & cor_aspect_df["aspect_sentiment"].isin(VALID_SENTIMENTS)
    ].copy()

    if len(cor_aspect_df) == 0:
        st.warning("No valid Negative / Neutral / Positive aspect rows were available for RQ1.")
        return

    total_reviews = len(llm_ready_df)
    total_aspect_rows = len(cor_aspect_df)
    avg_aspects = total_aspect_rows / total_reviews if total_reviews > 0 else 0

    def has_disagreement(group: pd.DataFrame) -> bool:
        return (group["aspect_sentiment"] != group["whole_sentiment"].iloc[0]).any()

    review_disagreement = (
        cor_aspect_df.groupby("review_id", sort=False)
        .apply(has_disagreement)
        .reset_index(name="has_disagreement")
    )

    reviews_with_disagreement = int(review_disagreement["has_disagreement"].sum())
    disagreement_rate = reviews_with_disagreement / len(review_disagreement) if len(review_disagreement) > 0 else 0

    confusion_pct = pd.crosstab(
        cor_aspect_df["whole_sentiment"],
        cor_aspect_df["aspect_sentiment"],
        normalize="index",
    ) * 100
    confusion_pct = confusion_pct.reindex(index=VALID_SENTIMENTS, columns=VALID_SENTIMENTS, fill_value=0).round(1)

    confusion_counts = pd.crosstab(
        cor_aspect_df["whole_sentiment"],
        cor_aspect_df["aspect_sentiment"],
    ).reindex(index=VALID_SENTIMENTS, columns=VALID_SENTIMENTS, fill_value=0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Original reviews", total_reviews)
    col2.metric("Aspect-level rows", total_aspect_rows)
    col3.metric("Avg aspects / review", f"{avg_aspects:.2f}")
    col4.metric("Disagreement rate", f"{disagreement_rate:.1%}")

    if disagreement_rate > 0.7:
        interp = "High disagreement"
    elif disagreement_rate > 0.4:
        interp = "Moderate disagreement"
    elif disagreement_rate > 0.2:
        interp = "Low disagreement"
    else:
        interp = "Minimal disagreement"
    st.info(f"{interp}. {reviews_with_disagreement} of {len(review_disagreement)} reviews contain at least one aspect whose sentiment differs from the whole-review label.")

    st.markdown("#### Sentiment Distribution (%)")
    st.dataframe(confusion_pct, use_container_width=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        confusion_pct,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        center=50,
        vmin=0,
        vmax=100,
        square=True,
        cbar_kws={"label": "Percentage (%)"},
        ax=ax,
    )
    ax.set_title("Whole-Text vs Aspect Sentiment", fontsize=13, pad=14)
    ax.set_xlabel("Aspect Sentiment", fontsize=11)
    ax.set_ylabel("Whole-Text Sentiment", fontsize=11)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    with st.expander("RQ1 debug preview", expanded=False):
        st.markdown("`llm_ready_df` preview")
        preview_cols = [
            col for col in ["review", "review_clean", "Sentiment_Label", "num_aspects", "aspects_found"]
            if col in llm_ready_df.columns
        ]
        st.dataframe(llm_ready_df[preview_cols].head(20), use_container_width=True)
        st.markdown("`cor_aspect_df` preview")
        st.dataframe(cor_aspect_df.head(30), use_container_width=True)
        st.markdown("Raw counts")
        st.dataframe(confusion_counts, use_container_width=False)

    finding = (
        f"Overall disagreement rate: {disagreement_rate:.1%}. "
        f"{reviews_with_disagreement} of {len(review_disagreement)} reviews showed at least one aspect-level disagreement."
    )

    st.session_state["rq1_results"] = {
        "odr": disagreement_rate,
        "n_disagree": reviews_with_disagreement,
        "total_reviews": len(review_disagreement),
        "confusion_pct": confusion_pct,
        "confusion_counts": confusion_counts,
        "finding": finding,
        "llm_ready_df": llm_ready_df,
        "cor_aspect_df": cor_aspect_df,
    }
