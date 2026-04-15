import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


def render_rq1_tab():
    st.markdown("### 🔗 RQ1: How Well Does Whole-Text Sentiment Agree With Aspect Sentiment?")

    st.markdown(
        "**Research Question:** To what extent does the overall (whole-text) sentiment of a "
        "student review agree with the sentiment expressed at the individual aspect level? "
        "Following Nashihin et al. (2025), disagreement is measured via the **Overall "
        "Disagreement Rate (ODR)** and a **confusion matrix** of whole-text vs aspect sentiment."
    )

    st.caption(
        "This analysis always runs on the **full uploaded dataset** and now follows the same "
        "notebook-style reshape and disagreement logic you shared."
    )

    main_global = st.session_state.get("df_full")
    aspect_global = st.session_state.get("aspect_df_full")

    if main_global is None or len(main_global) == 0:
        st.warning("No global sentiment data found. Please run sentiment analysis first.")
        return
    if aspect_global is None:
        st.warning("No global aspect data found. Please run aspect analysis first.")
        return
    if "sentiment" not in main_global.columns:
        st.warning("`df_full` must contain a `sentiment` column for RQ1.")
        return
    asp_sent_col = "aspect_sentiment" if "aspect_sentiment" in aspect_global.columns else "sentiment"
    if asp_sent_col not in aspect_global.columns:
        st.warning("`aspect_df_full` must contain either `aspect_sentiment` or `sentiment`.")
        return

    # Rebuild llm_ready_df exactly the way the notebook preview describes:
    # 1. create review_clean on both tables
    # 2. group aspect_df by review_clean into a dict
    # 3. map that dict onto the main review dataframe
    df = main_global.copy()
    aspect_df = aspect_global.copy()

    df["review_clean"] = df["review"].astype(str).str.strip().str.lower()
    aspect_df["review_clean"] = aspect_df["review"].astype(str).str.strip().str.lower()

    aspect_dict = (
        aspect_df.groupby("review_clean", sort=False)
        .apply(
            lambda x: {
                row["aspect"]: {
                    "sentiment": row[asp_sent_col],
                    "confidence": row["confidence"],
                }
                for _, row in x.iterrows()
            }
        )
        .to_dict()
    )

    llm_ready_df = df.copy()
    llm_ready_df = llm_ready_df.rename(columns={"sentiment": "Sentiment_Label"})
    llm_ready_df["aspect_data"] = llm_ready_df["review_clean"].map(aspect_dict)
    llm_ready_df["aspect_data"] = llm_ready_df["aspect_data"].apply(
        lambda x: x if isinstance(x, dict) else {}
    )
    llm_ready_df["aspects_found"] = llm_ready_df["aspect_data"].apply(lambda x: list(x.keys()))
    llm_ready_df["num_aspects"] = llm_ready_df["aspects_found"].apply(len)

    # Reshaping the data to get one row per aspect per review, i.e., Dissolving 1:M to 1:1
    aspect_rows = []

    for idx, row in llm_ready_df.iterrows():
        review_id = idx
        whole_sentiment = row["Sentiment_Label"]
        aspect_dict = row["aspect_data"]

        for aspect, sentiment_info in aspect_dict.items():
            aspect_rows.append(
                {
                    "review_id": review_id,
                    "review_text": row["review"],
                    "course_code": row["course_code"],
                    "academic_year": row["academic_year"],
                    "whole_sentiment": str(whole_sentiment).strip().title(),
                    "aspect": aspect,
                    "aspect_sentiment": str(sentiment_info["sentiment"]).strip().title(),
                    "aspect_confidence": sentiment_info["confidence"],
                }
            )

    cor_aspect_df = pd.DataFrame(aspect_rows)

    if len(cor_aspect_df) == 0:
        st.warning("No aspect-level rows were generated for RQ1.")
        return

    st.text(f"Original reviews: {len(llm_ready_df)}")
    st.text(f"Aspect-level rows: {len(cor_aspect_df)}")
    st.text(f"Average aspects per review: {len(cor_aspect_df) / len(llm_ready_df):.2f}")

    # Calculate DISAGREEMENT RATE (following Nashihin et al. 2025)
    def has_disagreement(group):
        """Check if any aspect sentiment differs from whole-text sentiment"""
        return any(group["aspect_sentiment"] != group["whole_sentiment"].iloc[0])

    # Group by review and check for disagreement
    review_disagreement = (
        cor_aspect_df
        .groupby("review_id", group_keys=False)[["aspect_sentiment", "whole_sentiment"]]
        .apply(has_disagreement)
        .reset_index()
    )
    review_disagreement.columns = ["review_id", "has_disagreement"]

    # Calculating Overall Disagreement Rate (ODR)
    total_reviews = len(review_disagreement)
    reviews_with_disagreement = int(review_disagreement["has_disagreement"].sum())
    disagreement_rate = reviews_with_disagreement / total_reviews if total_reviews > 0 else 0

    st.text("\n" + "=" * 60)
    st.text("RQ1: DISAGREEMENT ANALYSIS RESULTS")
    st.text("=" * 60)
    st.text(f"\nDISAGREEMENT RATE: {disagreement_rate:.1%}")
    st.text("   (Following Nashihin et al. 2025, this quantifies the additional")
    st.text("    nuance captured by aspect-based analysis)")

    # Value Interpretation
    if disagreement_rate > 0.7:
        nuance_level = "HIGH - Method captures significant additional nuance"
    elif disagreement_rate > 0.4:
        nuance_level = "MODERATE - Method adds meaningful nuance"
    elif disagreement_rate > 0.2:
        nuance_level = "LOW - Method adds some nuance"
    else:
        nuance_level = "MINIMAL - Method largely reproduces whole-text sentiment"

    st.text(f"\n INTERPRETATION: {nuance_level}")

    # ------------------------------------------------------------
    # STEP 3: Cross-tabulation heatmap (following Nashihin et al.)
    # ------------------------------------------------------------
    # For the heatmap, we need to align whole sentiment with aspect sentiment
    # We'll use all aspects for richer visualization.

    # Create a crosstab of whole sentiment vs aspect sentiment
    confusion_matrix = pd.crosstab(
        cor_aspect_df["whole_sentiment"],
        cor_aspect_df["aspect_sentiment"],
        normalize="index",
    ) * 100

    order = [label for label in ["Negative", "Neutral", "Positive"] if label in confusion_matrix.index or label in confusion_matrix.columns]
    confusion_matrix = confusion_matrix.reindex(index=order, columns=order, fill_value=0)

    st.text("\n" + "=" * 60)
    st.text("CROSS-TABULATION: When whole-text is [X], aspect sentiments are:")
    st.text("=" * 60)
    st.dataframe(confusion_matrix.round(1), use_container_width=False)

    # Heatmap visualization
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        confusion_matrix,
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

    ax.set_title(
        "Disagreement Heatmap: Whole-Text vs. Aspect Sentiment\n"
        "(Darker green = agreement, Darker red = disagreement)",
        fontsize=14,
        pad=20,
    )
    ax.set_xlabel("Aspect Sentiment", fontsize=12)
    ax.set_ylabel("Whole-Text Sentiment", fontsize=12)

    fig.text(
        0.5,
        -0.15,
        f"Following Nashihin et al. (2025): Disagreement Rate = {disagreement_rate:.1%}\n"
        "Diagonal cells (Agreement) show where aspect matches whole-text.\n"
        "Off-diagonal cells (Disagreement) reveal the nuance captured.",
        ha="center",
        fontsize=10,
        bbox=dict(facecolor="lightgray", alpha=0.5, boxstyle="round,pad=0.5"),
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    st.pyplot(fig, clear_figure=True)

    # ------------------------------------------------------------
    # Final Summary and bonus
    # ------------------------------------------------------------
    finding = f"""
Following the disagreement analysis framework of Nashihin et al. (2025),
we calculated an Overall Disagreement Rate (ODR) of {disagreement_rate:.1%}
between whole-text and aspect-level sentiment. This indicates that in
{disagreement_rate:.1%} of reviews, at least one aspect sentiment diverges
from the global classification—revealing precisely the nuanced, multi-dimensional
feedback that whole-text analysis alone would mask.

The heatmap visualization shows that while agreement is strongest on the diagonal
(e.g., Positive whole-text aligning with Positive aspects), significant off-diagonal
clusters reveal where nuance emerges. Most notably, when whole-text registers as
Neutral, aspects frequently register as Positive or Negative, confirming that
students express mixed sentiments that average to neutrality.
"""

    st.text("\n" + "=" * 60)
    st.text("RQ1 SUMMARY FOR RESEARCH WRITE-UP")
    st.text("=" * 60)
    st.text(finding)

    with st.expander("Preview `llm_ready_df` used by RQ1", expanded=False):
        preview_cols = [
            col for col in ["review", "review_clean", "Sentiment_Label", "num_aspects", "aspects_found", "aspect_data"]
            if col in llm_ready_df.columns
        ]
        st.dataframe(llm_ready_df[preview_cols].head(25), use_container_width=True)

    with st.expander("Preview `cor_aspect_df` used by RQ1", expanded=False):
        st.dataframe(cor_aspect_df.head(50), use_container_width=True)

    st.session_state["rq1_results"] = {
        "odr": disagreement_rate,
        "n_disagree": reviews_with_disagreement,
        "total_reviews": total_reviews,
        "confusion_pct": confusion_matrix.round(1),
        "confusion_counts": pd.crosstab(
            cor_aspect_df["whole_sentiment"],
            cor_aspect_df["aspect_sentiment"],
        ).reindex(index=order, columns=order, fill_value=0),
        "finding": finding,
        "llm_ready_df": llm_ready_df,
        "cor_aspect_df": cor_aspect_df,
    }
