import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def render_rq2_tab(config):
    st.markdown("### 📉 RQ2: Which Aspects Are Most Associated With Negative Evaluations?")

    st.markdown(
        "**Research Question:** Which course aspects are most strongly associated with "
        "negative overall evaluations, and does that association hold across three lenses — "
        "global, comparative, and per-group?"
    )

    st.caption(
        "Reviews are anonymous, so aspect mentions cannot be linked to individual overall "
        "ratings. The comparative view therefore works at the group level (section, year, "
        "or course) using OLS regression. Following Pang, Lee & Vaithyanathan (2002) and "
        "Schouten & Frasincar (2016), aspect-level sentiment carries predictive signal "
        "beyond whole-text sentiment alone."
    )

    st.divider()

    adf_rq2  = st.session_state.aspect_df
    main_rq2 = st.session_state.df

    # Global layer reads from the full cleaned snapshot, not the filtered analysis subset.
    # Comparative and Per-group layers continue to use adf_rq2/main_rq2 (Stage 3 filtered).
    adf_global  = st.session_state.get('aspect_df_full', adf_rq2)
    main_global = st.session_state.get('df_full',        main_rq2)

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

    st.markdown("## 🌐 Global — Across All Uploaded Data")

    neg_reviews_global = main_global[main_global['sentiment'] == 'Negative']
    total_neg_reviews  = len(neg_reviews_global)
    top_global_aspect  = None
    global_df          = None

    if total_neg_reviews == 0:
        st.info("No negative reviews found in the uploaded data — skipping global view.")
    else:
        if 'review' in adf_global.columns and 'review' in main_global.columns:
            neg_review_texts = set(neg_reviews_global['review'].tolist())
            adf_in_neg       = adf_global[adf_global['review'].isin(neg_review_texts)]
        else:
            adf_in_neg = adf_global[adf_global[s_col_rq2] == 'Negative']

        global_rows = []
        all_aspects_global = sorted(adf_global['aspect'].dropna().unique().tolist())

        for asp in all_aspects_global:
            reviews_with_asp = (
                adf_in_neg[adf_in_neg['aspect'] == asp]['review'].nunique()
                if 'review' in adf_in_neg.columns else 0
            )
            share_of_neg_reviews = (
                round(reviews_with_asp / total_neg_reviews * 100, 1)
                if total_neg_reviews > 0 else 0
            )
            global_rows.append({
                'Aspect': asp,
                'Share of Negative Reviews (%)': share_of_neg_reviews,
            })

        global_df = (
            pd.DataFrame(global_rows)
            .sort_values('Share of Negative Reviews (%)', ascending=False)
            .reset_index(drop=True)
        )

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

        top_global = global_df.iloc[0]
        top_global_aspect = top_global['Aspect']
        st.caption(
            f"**Top aspect across all uploaded data:** {top_global_aspect} — appears in "
            f"{top_global['Share of Negative Reviews (%)']:.1f}% of negative reviews."
        )

    st.divider()
    st.markdown("## 📊 Comparative — Regression Across Selected Groups")

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
            f"Only {len(groups_rq2)} group(s) found. At least 2 groups are needed for the "
            "comparative view. The global view above still applies."
        )
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

    var_mask = X_df.std() > 0
    X_df_var = X_df.loc[:, var_mask]

    max_predictors = min(5, X_df_var.shape[1])
    if X_df_var.shape[1] > max_predictors:
        top_var_aspects = X_df_var.std().nlargest(max_predictors).index.tolist()
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
    near_zero = coef_df['Coefficient'].abs().max() < 0.01

    st.caption(
        "Positive (red) coefficients differentiate more-negative groups from less-negative "
        "ones; negative (green) coefficients do not."
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
        height=max(320, len(coef_df) * 38),
        margin=dict(l=20, r=60, t=20, b=30),
        xaxis=dict(title='Standardised Coefficient', zeroline=True, zerolinewidth=2),
        yaxis=dict(autorange='reversed'),
        plot_bgcolor='white',
    )
    st.plotly_chart(fig_coef, use_container_width=True)

    if len(groups_rq2) == 2:
        st.caption(
            "⚠️ With only 2 groups the model fits perfectly by construction (R² = 1.0). "
            "Coefficients are descriptive only. This analysis is exploratory — "
            "coefficients indicate direction, not statistical significance."
        )
    else:
        st.caption(
            f"**Model fit:** R² = {r2:.3f} ({r2*100:.1f}% of variation explained). "
            "With one row per group this analysis is exploratory — coefficients indicate "
            "direction, not statistical significance."
        )


    top_comparative_aspect = coef_df.iloc[0]['Aspect'] if not coef_df.empty else None

    if top_global_aspect is not None and top_comparative_aspect is not None:
        if top_comparative_aspect == top_global_aspect:
            st.success(
                f"✓ The comparative view agrees with the global view — **{top_comparative_aspect}** "
                f"is the top aspect at both levels. Strong signal."
            )
        else:
            st.info(
                f"The comparative view picks **{top_comparative_aspect}** as the top differentiator "
                f"across your selected groups, while the global view picks **{top_global_aspect}** "
                f"as the department-wide top concern. This means {top_global_aspect} is a shared "
                f"concern across most courses (so it doesn't differentiate your selected groups), "
                f"while {top_comparative_aspect} is what makes some of your selected groups worse "
                f"than others."
            )

    st.divider()
    st.markdown("## 👥 Per-Group — Group-Specific Findings")
    st.caption(
        "For each selected group, the aspect with the highest negative mention rate and the "
        "aspect with the highest positive mention rate."
    )

    group_summaries = []
    per_group_top_aspects = []

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

        per_group_top_aspects.append(top_neg_asp)

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
            f"The biggest concern is **{top_neg_asp}** "
            f"({top_neg_asp_pct:.1f}% of its mentions were negative)."
        )
        if top_pos_asp:
            para += (
                f" The most positively received aspect is {top_pos_asp} "
                f"({top_pos_asp_pct:.1f}% positive mentions)."
            )

        st.write(para)
        group_summaries.append({'group': grp_str, 'summary': para})

    st.divider()

   
    st.markdown("## 🎯 Key Finding")

   
    if per_group_top_aspects:
        from collections import Counter
        per_group_counter = Counter(per_group_top_aspects)
        most_common_per_group = per_group_counter.most_common(1)[0][0]
    else:
        most_common_per_group = None

    layers = [top_global_aspect, top_comparative_aspect, most_common_per_group]
    layers_present = [l for l in layers if l is not None]
    all_agree = len(set(layers_present)) == 1 and len(layers_present) == 3

    finding_parts = []
    if top_global_aspect:
        finding_parts.append(
            f"Across all uploaded data, **{top_global_aspect}** is the most common concern "
            f"in negative reviews."
        )
    if top_comparative_aspect:
        finding_parts.append(
            f"Within the groups you selected, **{top_comparative_aspect}** is what "
            f"differentiates the more-negative groups from the less-negative ones."
        )
    if most_common_per_group:
        finding_parts.append(
            f"At the group level, **{most_common_per_group}** appears most often as the "
            f"top concern."
        )

    if all_agree:
        finding_parts.append(
            f"All three lenses converge on **{top_global_aspect}** — this is a strong, "
            f"consistent signal and should be the priority for intervention."
        )
    elif near_zero:
        finding_parts.append(
            "The comparative regression did not identify a clear differentiator "
            "(coefficients near zero), so the global and per-group views are the more "
            "useful guides for this selection."
        )
    elif top_global_aspect and top_comparative_aspect and top_global_aspect != top_comparative_aspect:
        finding_parts.append(
            f"The divergence between layers is informative: **{top_global_aspect}** is a "
            f"shared baseline concern across the department, while **{top_comparative_aspect}** "
            f"is the specific driver of variation among the groups you are comparing. Both "
            f"deserve attention, but they call for different responses — baseline concerns "
            f"need department-wide action, while comparative drivers point to group-specific fixes."
        )

    st.info(" ".join(finding_parts))

    interp = " ".join(finding_parts)

    st.session_state['rq2_results'] = {
        'coef_df':              coef_df,
        'r2':                   r2,
        'n_groups':             len(groups_rq2),
        'interp':               interp,
        'group_summaries':      group_summaries,
        'global_top_aspect':    top_global_aspect,
        'global_df':            global_df,
        'top_comparative':      top_comparative_aspect,
        'most_common_per_group': most_common_per_group,
    }