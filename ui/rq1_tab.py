import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


DISAGREEMENT_RATE = 30.4
TOTAL_REVIEWS = 1537
ASPECT_ROWS = 1958
AVG_ASPECTS = 1.27
CONFUSION_PCT = pd.DataFrame(
    {
        "Negative": [78.2, 30.5, 12.1],
        "Neutral": [18.4, 60.1, 25.6],
        "Positive": [3.4, 9.4, 62.4],
    },
    index=["Negative", "Neutral", "Positive"],
)


def render_rq1_tab():
    st.markdown("### 🔗 RQ1: Whole-Text vs Aspect Sentiment")
    st.caption(
        "This section presents a research finding from the project dataset used during the study."
    )

    st.warning(
        "These results were obtained from the dataset provided during the project and are shown here "
        "as a supporting finding. They are not meant to represent or summarize whatever files may be "
        "uploaded into the app at this point."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Disagreement Rate", f"{DISAGREEMENT_RATE:.1f}%")
    col2.metric("Reviews", f"{TOTAL_REVIEWS}")
    col3.metric("Aspect Mentions", f"{ASPECT_ROWS}")
    col4.metric("Avg Aspects / Review", f"{AVG_ASPECTS:.2f}")

    st.info(
        "About one in three reviews contained at least one aspect whose sentiment did not match the overall review label. "
        "That gap is the practical reason aspect-based analysis adds value here: whole-text sentiment can miss important nuance."
    )

    st.caption(
        "Following Nashihin et al. (2025), this quantifies the additional nuance captured by aspect-based analysis."
    )

    lead_col, note_col = st.columns([1.7, 1.1])

    with lead_col:
        st.markdown("#### Disagreement Heatmap")
        fig, ax = plt.subplots(figsize=(7.2, 5.1))
        sns.heatmap(
            CONFUSION_PCT,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=100,
            square=True,
            cbar_kws={"label": "Percentage (%)"},
            annot_kws={"fontsize": 12, "fontweight": "bold"},
            ax=ax,
        )
        ax.set_title(
            "Disagreement Heatmap: Whole-Text vs. Aspect Sentiment\n"
            "(Darker green = agreement, Darker red = disagreement)",
            fontsize=15,
            pad=18,
        )
        ax.set_xlabel("Aspect Sentiment", fontsize=13)
        ax.set_ylabel("Whole-Text Sentiment", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        st.caption(
            "Diagonal cells represent agreement between whole-review and aspect-level sentiment. "
            "Off-diagonal cells capture the nuance that is lost when only one overall label is used."
        )

    with note_col:
        st.markdown("#### Why This Matters")
        st.markdown(
            """
Whole-text sentiment is useful for a quick summary, but it can flatten mixed feedback into a single label.

This result shows that:

- negative reviews do not always stay fully negative at the aspect level
- neutral reviews often contain clearly positive or negative aspect signals
- positive reviews can still include meaningful complaints about specific parts of the course

That is why the project emphasizes aspect-based analysis for interpretation and recommendations.
            """
        )

        st.markdown("#### Key Takeaway")
        st.success(
            "A 30.4% disagreement rate indicates that aspect-based sentiment captures additional information that whole-text sentiment alone would miss."
        )

    st.markdown("#### Interpretation")
    st.markdown(
        """
The strongest agreement appears on the diagonal, especially for clearly negative and clearly positive reviews.  
At the same time, the off-diagonal mass is large enough to matter. The clearest example is the neutral row: many reviews labelled neutral overall still contain positive or negative aspect-level judgments. That pattern suggests students often give balanced or mixed comments that average out at the document level but become visible once the feedback is broken down by aspect.

For this reason, aspect-based analysis is not just an optional extra layer. In this project, it is the more informative lens for understanding what students actually liked, disliked, or felt uncertain about.
        """
    )

    with st.expander("View matrix values", expanded=False):
        st.dataframe(CONFUSION_PCT, use_container_width=False)

    st.session_state["rq1_results"] = {
        "odr": DISAGREEMENT_RATE / 100.0,
        "n_disagree": round(TOTAL_REVIEWS * DISAGREEMENT_RATE / 100.0),
        "total_reviews": TOTAL_REVIEWS,
        "confusion_pct": CONFUSION_PCT,
        "confusion_counts": None,
        "finding": (
            "Whole-text and aspect-level sentiment disagree in 30.4% of reviews, "
            "showing that aspect-based analysis captures meaningful nuance beyond a single overall label."
        ),
    }
