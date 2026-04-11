import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def render_rq2_tab(config):
    st.markdown("### 📉 RQ2: Which Aspects Most Predict Negative Course Evaluations?")

    st.markdown(
        "**Research Question:** To what extent does sentiment toward specific course aspects "
        "(e.g., instructor quality, assignments, workload) predict whether a student's overall "
        "course evaluation is negative?"
    )
    st.markdown("#### What is this analysis doing?")

    st.caption(
        "Since reviews are anonymous, we cannot link individual student comments to an overall rating. "
        "Instead, this analysis works at the group level — each group is one section, year, or course. "
        "The model asks: across the groups you selected, which aspects explain why some groups are more "
        "negative overall than others? It is not asking what students in a single group complain about — "
        "it is asking what differentiates a high-negativity group from a low-negativity group. "
        "For each group we calculate the overall negative review rate (the outcome) and the negative "
        "mention rate per aspect (the predictors). OLS linear regression then estimates which aspect "
        "negative rates best explain the variation in overall negativity across groups. "
        "A positive coefficient means groups that rate that aspect negatively tend to have higher "
        "overall negative rates — that aspect is driving the difference between groups. "
        "A negative coefficient means groups that struggle with that aspect still tend to be positive "
        "overall — it does not drive overall negativity across groups. "
        "Following Pang, Lee & Vaithyanathan (2002) and Schouten & Frasincar (2016), aspect-level "
        "sentiment features carry predictive signal about overall evaluation outcomes beyond what "
        "whole-text sentiment alone captures."
    )

    st.caption(
        "References: Pang, B., Lee, L., & Vaithyanathan, S. (2002). Thumbs up? Sentiment "
        "classification using machine learning techniques. EMNLP. "
        "Schouten, K., & Frasincar, F. (2016). Survey on aspect-level sentiment analysis. "
        "IEEE Transactions on Knowledge and Data Engineering, 28(3), 813-830."
    )

    st.info(
        "Note: this model requires at least 2 groups to fit. With only 2 groups the model "
        "will fit perfectly (R² = 1.0) but coefficients are unreliable — treat results as "
        "descriptive only. For more robust estimates, use 3 or more groups by selecting "
        "Compare Years or Cross-Course with additional years or courses. "
        "The more groups, the more meaningful the coefficients."
    )

    st.divider()

    adf_rq2  = st.session_state.aspect_df
    main_rq2 = st.session_state.df

    if adf_rq2 is None or len(adf_rq2) == 0:
        st.warning("No aspect data available. Please run aspect analysis first.")
        return
    if 'sentiment' not in main_rq2.columns:
        st.warning("No sentiment data available. Please run sentiment analysis first.")
        return

    s_col_rq2 = (
        'aspect_sentiment'
        if 'aspect_sentiment' in adf_rq2.columns
        else 'sentiment'
    )

    if config['type'] == "🔄 Compare Sections (Same Course, Same Year)":
        grp_rq2 = 'section'
    elif config['type'] == "📅 Compare Years (Same Course)":
        grp_rq2 = 'academic_year'
    elif config['type'] == "🔬 Cross-Course Comparison":
        grp_rq2 = 'course_year' if 'course_year' in adf_rq2.columns else 'course_code'
        if 'course_year' not in adf_rq2.columns and 'course_year' in main_rq2.columns:
            cy_map_rq2 = (
                main_rq2[['review', 'course_year']]
                .drop_duplicates('review')
                .set_index('review')['course_year']
            )
            adf_rq2 = adf_rq2.copy()
            adf_rq2['course_year'] = adf_rq2['review'].map(cy_map_rq2)
            adf_rq2 = adf_rq2.dropna(subset=['course_year'])
    else:
        grp_rq2 = 'section'

    all_aspects = sorted(adf_rq2['aspect'].dropna().unique().tolist())
    groups_rq2  = sorted(main_rq2[grp_rq2].dropna().unique().tolist()) if grp_rq2 in main_rq2.columns else []

    if len(groups_rq2) < 2:
        st.warning(
            f"Only {len(groups_rq2)} group(s) found. At least 2 groups are needed to fit the model. "
            "Try Compare Sections, Compare Years, or Cross-Course analysis."
        )
        st.markdown("#### Aspect Negative Rate by Group")
        raw_rows = []
        for gv in groups_rq2:
            sub = adf_rq2[adf_rq2[grp_rq2] == gv] if grp_rq2 in adf_rq2.columns else adf_rq2
            for asp in all_aspects:
                asp_sub = sub[sub['aspect'] == asp]
                total = len(asp_sub)
                if total == 0:
                    continue
                neg_pct = round((asp_sub[s_col_rq2] == 'Negative').sum() / total * 100, 1)
                raw_rows.append({grp_rq2: gv, 'Aspect': asp, 'Negative %': neg_pct, 'Mentions': total})
        if raw_rows:
            raw_df = pd.DataFrame(raw_rows)
            pivot_raw = raw_df.pivot_table(index='Aspect', columns=grp_rq2, values='Negative %').round(1)
            st.dataframe(pivot_raw, use_container_width=True)
        return

    X_rows, y_vals, group_labels = [], [], []
    for gv in groups_rq2:
        sub_asp  = adf_rq2[adf_rq2[grp_rq2] == gv] if grp_rq2 in adf_rq2.columns else adf_rq2
        sub_main = main_rq2[main_rq2[grp_rq2] == gv] if grp_rq2 in main_rq2.columns else main_rq2

        total_reviews = len(sub_main)
        if total_reviews == 0:
            continue

        neg_rate = (sub_main['sentiment'] == 'Negative').sum() / total_reviews

        feat_row = {}
        for asp in all_aspects:
            asp_sub = sub_asp[sub_asp['aspect'] == asp]
            total_asp = len(asp_sub)
            feat_row[asp] = (
                (asp_sub[s_col_rq2] == 'Negative').sum() / total_asp
                if total_asp > 0 else 0.0
            )
        X_rows.append(feat_row)
        y_vals.append(neg_rate)
        group_labels.append(str(gv))

    X_df = pd.DataFrame(X_rows, index=group_labels).fillna(0)
    y    = np.array(y_vals)

    st.markdown("#### Model")
    st.markdown(
        "We apply OLS linear regression where the outcome is the "
        "proportion of negative reviews per group, and the predictors are the "
        "proportion of negative mentions per aspect:"
    )
    st.latex(
        r"\log\left(\frac{p_{neg}}{1 - p_{neg}}\right) = "
        r"\beta_0 + \sum_{k=1}^{K} \beta_k \cdot \text{NegRate}_{k}"
    )
    st.caption(
        "p_neg = proportion of overall negative reviews in the group. "
        "NegRate_k = proportion of negative mentions for aspect k. "
        "beta_k = coefficient for aspect k — a larger positive beta means that aspect "
        "is a stronger predictor of negative overall evaluations."
    )

    st.divider()

    var_mask   = X_df.std() > 0
    X_df_var   = X_df.loc[:, var_mask]

    max_predictors = min(5, X_df_var.shape[1])
    if X_df_var.shape[1] > max_predictors:
        top_var_aspects = (
            X_df_var.std()
            .nlargest(max_predictors)
            .index.tolist()
        )
        X_df_var = X_df_var[top_var_aspects]

    scaler   = StandardScaler()
    X_sc_var = scaler.fit_transform(X_df_var)

    lm = LinearRegression()
    lm.fit(X_sc_var, y)

    coef_df = pd.DataFrame({
        'Aspect':      X_df_var.columns.tolist(),
        'Coefficient': lm.coef_.round(4),
    }).sort_values('Coefficient', ascending=False).reset_index(drop=True)

    r2 = lm.score(X_sc_var, y)

    coef_range = coef_df['Coefficient'].abs().max()
    near_zero  = coef_range < 0.01

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### Standardised Coefficients — Aspect Influence on Negative Evaluations")
        st.caption(
            "Coefficients are standardised (mean = 0, SD = 1) so aspects are directly "
            "comparable regardless of their base rate. A positive coefficient means "
            "higher negative sentiment for that aspect predicts more negative overall "
            "evaluations. A negative coefficient means it has a dampening or protective effect."
        )
        st.caption(
            f"Showing top {max_predictors} aspects by cross-group variance. "
            f"With {len(groups_rq2)} group(s), R² should be interpreted with caution — "
            "a small number of groups relative to predictors can inflate R². "
            "Focus on the direction and relative size of coefficients rather than R² alone."
        )

        COLOR_POS = '#F44336'
        COLOR_NEG = '#4CAF50'
        bar_colors = [COLOR_POS if c > 0 else COLOR_NEG for c in coef_df['Coefficient']]

        fig_coef = go.Figure(go.Bar(
            x=coef_df['Coefficient'],
            y=coef_df['Aspect'],
            orientation='h',
            marker_color=bar_colors,
            text=coef_df['Coefficient'].apply(lambda x: f"{x:+.3f}"),
            textposition='outside',
        ))
        fig_coef.update_layout(
            height=max(350, len(coef_df) * 38),
            margin=dict(l=20, r=60, t=30, b=30),
            xaxis=dict(title='Standardised Coefficient', zeroline=True, zerolinewidth=2),
            yaxis=dict(autorange='reversed'),
            plot_bgcolor='white',
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    with col2:
        st.markdown("#### Model Fit")
        if len(groups_rq2) == 2:
            st.warning(
                "R² is not meaningful here — with only 2 groups a linear "
                "model always fits perfectly regardless of the data. "
                "Add more groups for a reliable R²."
            )
        else:
            st.metric("R² (variance explained)", f"{r2:.3f}")
            st.caption(
                f"R² = {r2:.3f} means the aspect negative rates explain "
                f"{r2*100:.1f}% of the variation in overall negative evaluation "
                "rates across groups. Higher is better — above 0.5 indicates "
                "aspects are strong predictors of overall sentiment."
            )

        if near_zero:
            st.warning(
                "All coefficients are near zero (< 0.01). This means the selected "
                "groups have very similar aspect negative rates — the model cannot "
                "identify a meaningful predictor. The direction (positive vs negative "
                "coefficient) may still indicate tendency, but should not be "
                "interpreted as a strong finding. Try groups with more contrasting "
                "sentiment profiles."
            )

        st.divider()
        st.markdown("#### Top Predictors")
        top_pos = coef_df[coef_df['Coefficient'] > 0].head(3)
        top_neg_coef = coef_df[coef_df['Coefficient'] < 0].tail(3)

        if len(top_pos) > 0:
            st.markdown("**Most predictive of negative evaluations:**")
            for _, row in top_pos.iterrows():
                st.write(f"- {row['Aspect']} (beta = {row['Coefficient']:+.3f})")

        if len(top_neg_coef) > 0:
            st.markdown("**Protective / dampening aspects:**")
            for _, row in top_neg_coef.iterrows():
                st.write(f"- {row['Aspect']} (beta = {row['Coefficient']:+.3f})")

    st.divider()

    st.markdown("#### Aspect Negative Rate by Group (Model Input)")
    st.caption(
        "Each cell shows the percentage of mentions for that aspect that were negative "
        "in that group. The rightmost column shows the overall negative review rate for "
        "that group — this is the outcome the model is trying to explain. "
        "Compare aspects row by row: aspects that vary across groups tend to have "
        "stronger coefficients. Aspects that are high everywhere may be a widespread "
        "concern but won't show up as a strong predictor because they don't differentiate the groups."
    )
    display_X = (X_df * 100).round(1)
    display_X['Overall Negative %'] = (y * 100).round(1)
    display_X.index.name = grp_rq2
    styled_X = display_X.style.background_gradient(
        cmap='RdYlGn_r', vmin=0, vmax=100, axis=None
    ).format('{:.1f}%')
    st.dataframe(styled_X, use_container_width=True)

    # ── Predicted vs Actual scatter ──────────────────────────────────────────
    st.divider()
    st.markdown("#### Predicted vs Actual Overall Negative Rate")
    st.caption(
        "Each point is one group. The x-axis shows the overall negative review rate predicted "
        "by the model based on aspect negative rates. The y-axis shows the actual overall negative "
        "rate. Points close to the diagonal line mean the model explains that group well. "
        "Points far from the line are groups where aspect rates alone do not fully explain "
        "the overall negativity — other factors may be at play."
    )
    if len(groups_rq2) > 2:
        y_pred = lm.predict(X_sc_var)
        fig_scatter = go.Figure()
        min_val = min(min(y), min(y_pred)) * 100
        max_val = max(max(y), max(y_pred)) * 100
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', line=dict(dash='dash', color='grey', width=1),
            name='Perfect fit', showlegend=True
        ))
        fig_scatter.add_trace(go.Scatter(
            x=(y_pred * 100).round(1),
            y=(y * 100).round(1),
            mode='markers+text',
            text=group_labels,
            textposition='top center',
            marker=dict(size=12, color='#2563EB'),
            name='Groups',
        ))
        fig_scatter.update_layout(
            xaxis=dict(title='Predicted Overall Negative % (from model)', range=[0, 100]),
            yaxis=dict(title='Actual Overall Negative %', range=[0, 100]),
            height=420,
            plot_bgcolor='white',
            margin=dict(t=30, b=40, l=40, r=20),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Predicted vs Actual chart requires 3 or more groups.")

    st.divider()


    st.divider()

    st.markdown("#### Interpretation")

    top1 = coef_df.iloc[0]
    top2 = coef_df.iloc[1] if len(coef_df) > 1 else None

    if near_zero:
        interp = (
            f"The selected groups have very similar aspect negative rates, "
            f"so no aspect emerges as a clear predictor. "
            f"Directionally, {top1['Aspect']} shows the largest positive tendency "
            f"(beta = {top1['Coefficient']:+.3f}), suggesting it may lean toward "
            f"predicting negative evaluations, but the difference is too small to "
            f"be meaningful. Consider running this analysis with groups that have "
            f"more contrasting sentiment profiles."
        )
    else:
        r2_str = "" if len(groups_rq2) == 2 else (
            f"OLS linear regression explains {r2*100:.1f}% of the variation "
            f"in negative evaluation rates across groups (R² = {r2:.3f}). "
        )
        interp = (
            f"{r2_str}"
            f"The aspect most predictive of negative overall evaluations is "
            f"{top1['Aspect']} (beta = {top1['Coefficient']:+.3f}), meaning groups "
            f"where students rate {top1['Aspect']} negatively tend to have higher "
            f"overall negative evaluation rates. "
        )
        if top2 is not None:
            interp += (
                f"{top2['Aspect']} is the second strongest predictor "
                f"(beta = {top2['Coefficient']:+.3f}). "
            )
        prot = coef_df[coef_df['Coefficient'] < 0]
        if len(prot) > 0:
            prot1 = prot.iloc[-1]
            interp += (
                f"Conversely, {prot1['Aspect']} shows a negative coefficient "
                f"(beta = {prot1['Coefficient']:+.3f}), suggesting it may have a "
                f"protective effect — groups where students feel positively about "
                f"{prot1['Aspect']} tend to report fewer negative evaluations overall."
            )

    st.write(interp)
    st.caption(
        "Note: With group-level data the sample size is small (one row per group). "
        "Coefficients indicate direction and relative importance but should be "
        "interpreted with caution. This analysis is exploratory and descriptive."
    )

    st.divider()
    st.markdown("#### Group-Specific Findings")
    st.caption(
        "For each group, the aspect with the highest negative mention rate is identified "
        "alongside the group's overall negative review rate. This gives a concrete, "
        "actionable picture of where concern is most concentrated for each specific group."
    )

    group_summaries = []
    for gv, y_val in zip(group_labels, y_vals):
        sub_asp = adf_rq2[adf_rq2[grp_rq2] == gv] if grp_rq2 in adf_rq2.columns else adf_rq2

        # Get neg and pos rates per aspect for this group
        asp_neg_rates = {}
        asp_pos_rates = {}
        for asp in all_aspects:
            asp_sub   = sub_asp[sub_asp['aspect'] == asp]
            total_asp = len(asp_sub)
            if total_asp > 0:
                asp_neg_rates[asp] = round(
                    (asp_sub[s_col_rq2] == 'Negative').sum() / total_asp * 100, 1
                )
                asp_pos_rates[asp] = round(
                    (asp_sub[s_col_rq2] == 'Positive').sum() / total_asp * 100, 1
                )

        if not asp_neg_rates:
            continue

        top_neg_asp     = max(asp_neg_rates, key=asp_neg_rates.get)
        top_neg_asp_pct = asp_neg_rates[top_neg_asp]
        top_pos_asp     = max(asp_pos_rates, key=asp_pos_rates.get) if asp_pos_rates else None
        top_pos_asp_pct = asp_pos_rates.get(top_pos_asp, 0) if top_pos_asp else 0
        overall_neg_pct = round(y_val * 100, 1)

        # Check if top neg aspect aligns with model coefficients
        if not coef_df.empty and top_neg_asp in coef_df['Aspect'].values:
            coef_val   = coef_df[coef_df['Aspect'] == top_neg_asp]['Coefficient'].values[0]
            model_note = (
                f" The regression model confirms {top_neg_asp} as a predictor "
                f"(beta = {coef_val:+.3f}), consistent with this group's profile."
                if coef_val > 0 else
                f" Interestingly, the regression model shows {top_neg_asp} has a "
                f"dampening coefficient across groups (beta = {coef_val:+.3f}), "
                f"suggesting it may not be the primary driver of overall negativity "
                f"when compared across all groups."
            )
        else:
            model_note = ""

        # Build group label string
        if config['type'] == "🔄 Compare Sections (Same Course, Same Year)":
            grp_str = f"Section {gv} of {config['course']} ({config['year']})"
        elif config['type'] == "📅 Compare Years (Same Course)":
            grp_str = f"{config['course']} in {gv}"
        elif config['type'] == "🔬 Cross-Course Comparison":
            grp_str = str(gv)
        else:
            grp_str = f"Section {gv}"

        para = (
            f"For {grp_str}, the aspect with the highest negative mention rate is "
            f"{top_neg_asp} ({top_neg_asp_pct:.1f}% of its mentions were negative). "
            f"The overall negative review rate for this group is {overall_neg_pct:.1f}%."
        )
        if top_pos_asp:
            para += (
                f" The most positively received aspect is {top_pos_asp} "
                f"({top_pos_asp_pct:.1f}% positive mentions)."
            )
        para += model_note

        st.write(para)
        group_summaries.append({'group': grp_str, 'summary': para})

    st.session_state['rq2_results'] = {
        'coef_df':         coef_df,
        'r2':              r2,
        'n_groups':        len(groups_rq2),
        'interp':          interp,
        'group_summaries': group_summaries,
    }