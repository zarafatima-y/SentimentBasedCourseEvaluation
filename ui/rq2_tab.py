import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def render_rq2_tab(config):
    st.markdown("### 📉 RQ2: Which Aspects Most Predict Negative Course Evaluations?")

    st.markdown(
        "**Research Question:** Which aspects are associated with negative course evaluations? "
        "This tab answers the question at **three levels** — global (all uploaded data), "
        "comparative (regression across the groups you selected), and per-group (one paragraph "
        "per selected group). These three views can legitimately disagree, and the disagreements "
        "are informative."
    )

    st.caption(
        "Since reviews are anonymous, we cannot link individual comments to individual overall "
        "ratings. The comparative view works at the group level — each group is one section, "
        "year, or course — and uses OLS linear regression to estimate which aspect negative rates "
        "track most closely with overall negativity across groups. "
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

    st.markdown("## 🌐 Global Answer — Across All Uploaded Data")
    st.caption(
        "This section uses **every review uploaded**, ignoring the groups you selected "
        "in Stage 3. It answers the department-wide question: *across all courses in your "
        "data, which aspects appear most often in negative reviews?* This is independent "
        "of the regression below and gives an absolute baseline that is not affected by "
        "which specific courses you are currently comparing."
    )

    neg_reviews_global = main_rq2[main_rq2['sentiment'] == 'Negative']
    total_neg_reviews  = len(neg_reviews_global)
    top_global_aspect  = None
    global_df          = None

    if total_neg_reviews == 0:
        st.info("No negative reviews found in the uploaded data — skipping global view.")
    else:
        if 'review' in adf_rq2.columns and 'review' in main_rq2.columns:
            neg_review_texts = set(neg_reviews_global['review'].tolist())
            adf_in_neg       = adf_rq2[adf_rq2['review'].isin(neg_review_texts)]
        else:
            adf_in_neg = adf_rq2[adf_rq2[s_col_rq2] == 'Negative']

        global_rows = []
        all_aspects_global = sorted(adf_rq2['aspect'].dropna().unique().tolist())

        for asp in all_aspects_global:
            reviews_with_asp = (
                adf_in_neg[adf_in_neg['aspect'] == asp]['review'].nunique()
                if 'review' in adf_in_neg.columns else 0
            )
            share_of_neg_reviews = (
                round(reviews_with_asp / total_neg_reviews * 100, 1)
                if total_neg_reviews > 0 else 0
            )

            asp_all = adf_rq2[adf_rq2['aspect'] == asp]
            total_mentions = len(asp_all)
            neg_rate = (
                round((asp_all[s_col_rq2] == 'Negative').sum() / total_mentions * 100, 1)
                if total_mentions > 0 else 0
            )

            global_rows.append({
                'Aspect': asp,
                'Share of Negative Reviews (%)': share_of_neg_reviews,
                'Negative Mention Rate (%)':     neg_rate,
                'Total Mentions':                total_mentions,
            })

        global_df = (
            pd.DataFrame(global_rows)
            .sort_values('Share of Negative Reviews (%)', ascending=False)
            .reset_index(drop=True)
        )

        col_g1, col_g2 = st.columns([3, 2])

        with col_g1:
            st.markdown("**Aspects most associated with negative reviews (global)**")
            fig_global = go.Figure(go.Bar(
                x=global_df['Share of Negative Reviews (%)'],
                y=global_df['Aspect'],
                orientation='h',
                marker_color='#F44336',
                text=global_df['Share of Negative Reviews (%)'].apply(lambda x: f"{x:.1f}%"),
                textposition='outside',
            ))
            fig_global.update_layout(
                height=max(320, len(global_df) * 34),
                margin=dict(l=20, r=60, t=20, b=30),
                xaxis=dict(title='% of negative reviews mentioning this aspect'),
                yaxis=dict(autorange='reversed'),
                plot_bgcolor='white',
            )
            st.plotly_chart(fig_global, use_container_width=True)

        with col_g2:
            st.markdown("**Global summary**")
            st.metric("Total reviews uploaded", f"{len(main_rq2):,}")
            st.metric("Negative reviews", f"{total_neg_reviews:,}")

            top_global = global_df.iloc[0]
            top_global_aspect = top_global['Aspect']
            st.markdown(
                f"**Top aspect department-wide:** {top_global_aspect} — "
                f"appears in {top_global['Share of Negative Reviews (%)']:.1f}% of all "
                f"negative reviews uploaded, and {top_global['Negative Mention Rate (%)']:.1f}% "
                f"of its own mentions were negative."
            )

        with st.expander("Full global table (all aspects)"):
            st.dataframe(global_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("## 📊 Comparative Answer — Regression Across Selected Groups")
    st.caption(
        "This section answers: *among the groups you picked in Stage 3, what separates "
        "the more-negative ones from the less-negative ones?* It is a comparison question, "
        "not an absolute one. If the top aspect here differs from the global top above, "
        "that is informative — it means the selected groups differ on something that is not "
        "the department's biggest overall concern."
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
            f"Only {len(groups_rq2)} group(s) found. At least 2 groups are needed to fit the "
            "regression. The global view above still applies. Try Compare Sections, Compare "
            "Years, or Cross-Course analysis for the comparative layer."
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

    st.info(
        "Note: this model requires at least 2 groups to fit. With only 2 groups the model "
        "will fit perfectly (R² = 1.0) but coefficients are unreliable — treat results as "
        "descriptive only. For more robust estimates, use 3 or more groups."
    )

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

    st.markdown("#### Standardised Coefficients — Aspect Influence on Negative Evaluations")
    st.caption(
        f"Showing top {max_predictors} aspects by cross-group variance. Coefficients are "
        "standardised so aspects are directly comparable regardless of their base rate. "
        "A positive (red) coefficient means higher negative sentiment for that aspect predicts "
        "more negative overall evaluations across groups. A negative (green) coefficient means "
        "it does not drive overall negativity in this comparison. Focus on direction and "
        "relative bar length rather than exact values."
    )

    if len(groups_rq2) == 2:
        st.warning(
            "With only 2 groups the model fits perfectly by construction (R² = 1.0). "
            "Coefficients below are descriptive only — add more groups for reliable estimates."
        )
    else:
        st.caption(f"**Model fit:** R² = {r2:.3f} — aspect negative rates explain "
                   f"{r2*100:.1f}% of the variation in overall negativity across groups.")

    if near_zero:
        st.warning(
            "All coefficients are near zero (< 0.01). The selected groups have very similar "
            "aspect negative rates, so no aspect emerges as a clear differentiator. The "
            "Group-Specific Findings below are the more useful view for this selection."
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

    # Global-vs-comparative agreement callout
    if top_global_aspect is not None and not coef_df.empty:
        top_comparative = coef_df.iloc[0]['Aspect']
        if top_comparative == top_global_aspect:
            st.success(
                f"✓ The comparative view agrees with the global view — **{top_comparative}** "
                f"is the top aspect at both levels. Strong signal."
            )
        else:
            st.info(
                f"Note: the comparative view picks **{top_comparative}** as the top "
                f"differentiator across your selected groups, while the global view "
                f"picks **{top_global_aspect}** as the department-wide top concern. "
                f"This means {top_global_aspect} is a shared concern across most courses "
                f"(so it doesn't differentiate your selected groups), while {top_comparative} "
                f"is what makes some of your selected groups worse than others."
            )
    st.divider()
    st.markdown("## 👥 Per-Group Answer — Group-Specific Findings")
    st.caption(
        "For each group, the aspect with the highest negative mention rate is identified "
        "alongside the group's overall negative review rate. This answers the practical "
        "question — *for this specific group, what is the biggest concern?* — independently "
        "of the global and comparative views above."
    )

    group_summaries = []
    for gv, y_val in zip(group_labels, y_vals):
        sub_asp = adf_rq2[adf_rq2[grp_rq2] == gv] if grp_rq2 in adf_rq2.columns else adf_rq2

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

        if not coef_df.empty and top_neg_asp in coef_df['Aspect'].values:
            coef_val   = coef_df[coef_df['Aspect'] == top_neg_asp]['Coefficient'].values[0]
            model_note = (
                f" The regression model confirms {top_neg_asp} as a cross-group predictor "
                f"(beta = {coef_val:+.3f})."
                if coef_val > 0 else
                f" Across groups, however, {top_neg_asp} has a negative coefficient "
                f"(beta = {coef_val:+.3f}), meaning it is a concern for this specific group "
                f"but not what differentiates high-negativity groups from low-negativity ones overall."
            )
        else:
            model_note = ""

        if config['type'] == "🔄 Compare Sections (Same Course, Same Year)":
            grp_str = f"Section {gv} of {config['course']} ({config['year']})"
        elif config['type'] == "📅 Compare Years (Same Course)":
            grp_str = f"{config['course']} in {gv}"
        elif config['type'] == "🔬 Cross-Course Comparison":
            grp_str = str(gv)
        else:
            grp_str = f"Section {gv}"

        para = (
            f"**{grp_str}** — overall negative review rate is {overall_neg_pct:.1f}%. "
            f"The aspect with the highest negative mention rate is {top_neg_asp} "
            f"({top_neg_asp_pct:.1f}% of its mentions were negative)."
        )
        if top_pos_asp:
            para += (
                f" The most positively received aspect is {top_pos_asp} "
                f"({top_pos_asp_pct:.1f}% positive mentions)."
            )
        para += model_note

        st.write(para)
        group_summaries.append({'group': grp_str, 'summary': para})

    st.divider()
    st.markdown("#### Aspect Negative Rate by Group (Regression Input)")
    st.caption(
        "Each cell shows the percentage of mentions for that aspect that were negative "
        "in that group. The rightmost column shows the overall negative review rate for "
        "that group — the outcome the regression is trying to explain. Aspects that vary "
        "across groups tend to have stronger coefficients; aspects that are high everywhere "
        "are widespread concerns but won't show up as strong predictors because they don't "
        "differentiate the groups."
    )
    display_X = (X_df * 100).round(1)
    display_X['Overall Negative %'] = (y * 100).round(1)
    display_X.index.name = grp_rq2
    styled_X = display_X.style.background_gradient(
        cmap='RdYlGn_r', vmin=0, vmax=100, axis=None
    ).format('{:.1f}%')
    st.dataframe(styled_X, use_container_width=True)

    st.caption(
        "Note: With group-level data the sample size is small (one row per group). "
        "Coefficients indicate direction and relative importance but should be "
        "interpreted with caution. This analysis is exploratory and descriptive."
    )

    top1 = coef_df.iloc[0]
    if near_zero:
        interp = (
            f"The selected groups have very similar aspect negative rates, so no aspect "
            f"emerges as a clear cross-group predictor. Directionally, {top1['Aspect']} "
            f"shows the largest positive tendency (beta = {top1['Coefficient']:+.3f})."
        )
    else:
        r2_str = "" if len(groups_rq2) == 2 else (
            f"OLS regression explains {r2*100:.1f}% of the variation in overall negativity "
            f"across groups (R² = {r2:.3f}). "
        )
        interp = (
            f"{r2_str}The aspect most predictive of negative overall evaluations across "
            f"groups is {top1['Aspect']} (beta = {top1['Coefficient']:+.3f})."
        )

    st.session_state['rq2_results'] = {
        'coef_df':           coef_df,
        'r2':                r2,
        'n_groups':          len(groups_rq2),
        'interp':            interp,
        'group_summaries':   group_summaries,
        'global_top_aspect': top_global_aspect,
        'global_df':         global_df,
    }