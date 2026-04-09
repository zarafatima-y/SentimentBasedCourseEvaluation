import re as _re
import streamlit as st
import pandas as pd

from models.llmsum import LLMAnalyzer
from config.settings import HF_TOKEN, LLM_MODEL


def render_llm_tab(config):
    st.markdown("### 🤖 LLM-Generated Course Improvement Summary")
    st.markdown(
        "Generates a structured summary using **Meta-Llama-3-8B-Instruct**, "
        "grounded in the aspect-sentiment data from this analysis."
    )
    st.caption(
        "The model identifies top positive aspects, the 3 most negative aspects, "
        "and produces actionable improvement recommendations."
    )

    adf     = st.session_state.aspect_df
    main_df = st.session_state.df

    aspect_df = st.session_state.aspect_df
    s_col = (
        'aspect_sentiment'
        if aspect_df is not None and 'aspect_sentiment' in aspect_df.columns
        else 'sentiment'
    )

    def build_llm_prompt(adf, main_df, config, s_col):

        ASPECT_TO_NUMERIC = {
            'resources_materials':  'Course materials helped achieve objectives',
            'course_content':       'Course materials helped achieve objectives',
            'assessments':          'Tests/exams related to objectives',
            'learning_outcomes':    'Learning outcomes clearly stated & achieved',
            'instructor':           'Instructor: overall effectiveness',
            'engagement_interest':  'Instructor: clear & organized delivery',
            'support_help':         'Instructor: students feel welcome to seek help',
        }
        Q_LABELS = {
            ('core',   1): 'Syllabus provided',
            ('core',   4): 'Course materials helped achieve objectives',
            ('core',   6): 'Tests/exams related to objectives',
            ('course', 4): 'Learning outcomes clearly stated & achieved',
            ('lect',   1): 'Instructor: clear & organized delivery',
            ('lect',   3): 'Instructor: students feel welcome to seek help',
            ('lect',   7): 'Instructor: overall effectiveness',
        }
        improvement_hints = {
            'difficulty_challenge': 'Re-evaluate course pacing and difficulty progression; consider scaffolded assignments and clearer prerequisite communication.',
            'assignments_labs':     'Review lab structure and assignment clarity; add rubrics and more frequent low-stakes checkpoints.',
            'assessments':          'Reconsider exam format and grading fairness; provide practice exams and detailed feedback on evaluations.',
            'workload_pace':        'Audit weekly workload; redistribute heavy weeks and communicate schedule changes early.',
            'course_content':       'Update or trim course content to match student level; increase real-world relevance of topics covered.',
            'instructor':           'Encourage instructor training or peer review; gather mid-semester feedback to allow in-course corrections.',
            'resources_materials':  'Improve quality and accessibility of course materials; ensure slides, notes, and readings are aligned with assessments.',
            'support_help':         'Increase TA hours or office hours availability; add async support channels (e.g., discussion boards).',
            'engagement_interest':  'Incorporate more interactive elements such as case studies, demos, or group activities to boost engagement.',
            'learning_outcomes':    'Clarify stated learning outcomes and align them explicitly with assignments and exams.',
        }
        LOW_SCORE_THRESHOLD = 5.0

        if config['type'] == "🔄 Compare Sections (Same Course, Same Year)":
            grp_col    = 'section'
            grp_vals   = config['sections']
            report_for = f"{config['course']} ({config['year']})"
            grp_label  = "section"
        elif config['type'] == "📅 Compare Years (Same Course)":
            grp_col    = 'academic_year'
            grp_vals   = sorted(config['years'])
            report_for = config['course']
            grp_label  = "year"
        elif config['type'] == "🔬 Cross-Course Comparison":
            cy_col     = 'course_year' if 'course_year' in adf.columns else 'course_code'
            grp_col    = cy_col
            grp_vals   = sorted(adf[cy_col].dropna().unique().tolist())
            report_for = "Cross-Course Comparison"
            grp_label  = "course"
        else:
            grp_col    = None
            grp_vals   = [None]
            report_for = f"{config['course']} ({config['year']})"
            grp_label  = "course"

        def aspect_profile(df_slice):
            if len(df_slice) == 0:
                return {}
            grp_asp = (
                df_slice.groupby(['aspect', s_col])
                .size()
                .unstack(fill_value=0)
            )
            for c in ['Negative', 'Neutral', 'Positive']:
                if c not in grp_asp.columns:
                    grp_asp[c] = 0
            grp_asp['total']   = grp_asp[['Negative', 'Neutral', 'Positive']].sum(axis=1)
            grp_asp['pos_pct'] = (grp_asp['Positive'] / grp_asp['total'] * 100).round(1)
            grp_asp['neg_pct'] = (grp_asp['Negative'] / grp_asp['total'] * 100).round(1)
            return grp_asp[['pos_pct', 'neg_pct', 'total']].to_dict('index')

        def numeric_means_for(ndf_raw, filter_kwargs):
            if ndf_raw is None or len(ndf_raw) == 0:
                return {}
            ndf_s = ndf_raw.copy()
            for col, val in filter_kwargs.items():
                if isinstance(val, list):
                    ndf_s = ndf_s[ndf_s[col].isin(val)]
                else:
                    ndf_s = ndf_s[ndf_s[col] == val]
            if len(ndf_s) == 0:
                return {}
            ndf_s['q_label'] = ndf_s.apply(
                lambda r: Q_LABELS.get(
                    (r['subsection'], r['question_number']),
                    r['question_text'][:60]
                ), axis=1
            )
            return (
                ndf_s
                .drop_duplicates(subset=['q_label', 'academic_year', 'section'])
                .groupby('q_label')['mean']
                .mean()
                .round(2)
                .to_dict()
            )

        all_profiles = {}
        for gv in grp_vals:
            if grp_col is None:
                slice_df = adf
            else:
                slice_df = adf[adf[grp_col] == gv]
            all_profiles[gv] = aspect_profile(slice_df)

        def best_group_for(aspect):
            best_grp, best_neg = None, 100.0
            for gv, profile in all_profiles.items():
                if aspect in profile and profile[aspect]['total'] >= 5:
                    if profile[aspect]['neg_pct'] < best_neg:
                        best_neg  = profile[aspect]['neg_pct']
                        best_grp  = gv
            return best_grp, best_neg

        # Cross-group synthesis: aspects that are weak in some groups but strong in others
        all_asp_set = set()
        for profile in all_profiles.values():
            all_asp_set.update(profile.keys())

        contrast_lines = []
        for asp in sorted(all_asp_set):
            neg_groups = sorted(
                [(gv, all_profiles[gv][asp]['neg_pct'])
                 for gv in grp_vals
                 if gv in all_profiles and asp in all_profiles[gv]
                 and all_profiles[gv][asp]['total'] >= 3
                 and all_profiles[gv][asp]['neg_pct'] >= 35],
                key=lambda x: -x[1]
            )
            pos_groups = sorted(
                [(gv, all_profiles[gv][asp]['pos_pct'])
                 for gv in grp_vals
                 if gv in all_profiles and asp in all_profiles[gv]
                 and all_profiles[gv][asp]['total'] >= 3
                 and all_profiles[gv][asp]['pos_pct'] >= 55],
                key=lambda x: -x[1]
            )
            neg_gvs = {gv for gv, _ in neg_groups}
            pos_gvs = {gv for gv, _ in pos_groups}
            if neg_groups and pos_groups and not neg_gvs.issubset(pos_gvs):
                neg_str = ", ".join(f"{grp_label} '{gv}' ({pct}% neg)" for gv, pct in neg_groups)
                pos_str = ", ".join(f"{grp_label} '{gv}' ({pct}% pos)" for gv, pct in pos_groups)
                contrast_lines.append(
                    f"  {asp}: weak in {neg_str} but strong in {pos_str} — "
                    f"advise instructor to review what the stronger {grp_label} does differently"
                )

        contrast_block = ""
        if contrast_lines and len(grp_vals) > 1:
            contrast_block = (
                "\n\nCROSS-GROUP INSIGHTS (use these to advise cross-group learning):\n"
                + "\n".join(contrast_lines)
            )

        ndf_raw = st.session_state.numeric_df
        group_sections = []

        for gv in grp_vals:
            label = str(gv) if gv is not None else report_for
            profile = all_profiles[gv]
            if not profile:
                continue

            sorted_neg = sorted(profile.items(), key=lambda x: -x[1]['neg_pct'])
            sorted_pos = sorted(profile.items(), key=lambda x: -x[1]['pos_pct'])
            top3_neg   = sorted_neg[:3]
            top3_pos   = sorted_pos[:3]

            if grp_col == 'section':
                num_means = numeric_means_for(ndf_raw, {
                    'course_code': config['course'],
                    'academic_year': config['year'],
                    'section': gv
                })
            elif grp_col == 'academic_year':
                num_means = numeric_means_for(ndf_raw, {
                    'course_code': config['course'],
                    'academic_year': gv
                })
            elif grp_col is None:
                num_means = numeric_means_for(ndf_raw, {
                    'course_code': config['course'],
                    'academic_year': config['year']
                })
            else:
                matched = [(c, y) for c, y in config.get('courses', [])
                           if f"{c} ({y})" == str(gv)]
                if matched:
                    num_means = numeric_means_for(ndf_raw, {
                        'course_code': matched[0][0],
                        'academic_year': matched[0][1]
                    })
                else:
                    num_means = {}

            # Per-group sentiment breakdown
            if grp_col is None:
                grp_main = main_df
            else:
                grp_main = main_df[main_df[grp_col] == gv] if grp_col in main_df.columns else main_df
            grp_total = len(grp_main)
            if grp_total > 0 and 'sentiment' in grp_main.columns:
                sc_grp = grp_main['sentiment'].value_counts().to_dict()
                pos_n = sc_grp.get('Positive', 0)
                neu_n = sc_grp.get('Neutral', 0)
                neg_n = sc_grp.get('Negative', 0)
                sentiment_line = (
                    f"Reviews: {grp_total} | "
                    f"Positive: {pos_n} ({round(pos_n / grp_total * 100)}%), "
                    f"Neutral: {neu_n} ({round(neu_n / grp_total * 100)}%), "
                    f"Negative: {neg_n} ({round(neg_n / grp_total * 100)}%)"
                )
            else:
                sentiment_line = f"Reviews: {grp_total}"

            # Dominant emotion if available
            emotion_line = ""
            if 'dominant_emotion' in grp_main.columns and grp_total > 0:
                dom_emotion = grp_main['dominant_emotion'].dropna().mode()
                if len(dom_emotion) > 0:
                    emotion_line = f" | Dominant emotion: {dom_emotion.iloc[0]}"

            pos_line = ", ".join(
                f"{asp} ({d['pos_pct']}% positive, {d['total']} mentions)"
                for asp, d in top3_pos
            )

            neg_lines = []
            for asp, d in top3_neg:
                hint = improvement_hints.get(
                    asp, 'Gather more targeted student feedback to identify specific pain points.'
                )
                line = (
                    f"  {asp}: {d['neg_pct']}% negative, {d['pos_pct']}% positive "
                    f"({d['total']} total mentions)\n"
                    f"    Recommendation: {hint}"
                )

                if grp_col is not None and len(grp_vals) > 1:
                    best_grp, best_neg = best_group_for(asp)
                    if best_grp is not None and best_grp != gv:
                        line += (
                            f"\n    BENCHMARK: {grp_label} '{best_grp}' has lower negative rate "
                            f"({best_neg}% negative) — review how teaching or structure differs there."
                        )

                linked_q = ASPECT_TO_NUMERIC.get(asp)
                if linked_q and linked_q in num_means:
                    score = num_means[linked_q]
                    signal = " [DOUBLE SIGNAL — treat as priority]" if score < LOW_SCORE_THRESHOLD else ""
                    line += f"\n    Survey score: '{linked_q}' rated {score}/7{signal}"

                neg_lines.append(line)

            numeric_context = ""
            if num_means:
                low_scores = {q: sc for q, sc in num_means.items() if sc < LOW_SCORE_THRESHOLD}
                high_scores = {q: sc for q, sc in num_means.items() if sc >= 6.0}
                parts = []
                if low_scores:
                    low_str = "; ".join(f"{q} ({sc}/7)" for q, sc in low_scores.items())
                    parts.append(f"Low numeric scores (below 5/7): {low_str}.")
                if high_scores:
                    high_str = "; ".join(f"{q} ({sc}/7)" for q, sc in high_scores.items())
                    parts.append(f"High numeric scores (above 6/7): {high_str}.")
                if parts:
                    numeric_context = "\n  Numeric survey context: " + " ".join(parts)

            group_sections.append(
                f"\n=== {grp_label.upper()}: {label} ===\n"
                f"{sentiment_line}{emotion_line}\n"
                f"Appreciated aspects: {pos_line}\n"
                f"Aspects needing improvement:\n"
                + "\n".join(neg_lines)
                + numeric_context
            )

        yoy_block = ""
        if config['type'] == "📅 Compare Years (Same Course)":
            sorted_years = sorted(config['years'])
            yr_asp_pct = {gv: {
                asp: d['pos_pct']
                for asp, d in all_profiles[gv].items()
            } for gv in sorted_years}

            delta_lines = []
            for i in range(len(sorted_years) - 1):
                yr_a, yr_b = sorted_years[i], sorted_years[i + 1]
                for asp in sorted(set(yr_asp_pct.get(yr_a, {})) & set(yr_asp_pct.get(yr_b, {}))):
                    delta = round(yr_asp_pct[yr_b][asp] - yr_asp_pct[yr_a][asp], 1)
                    if abs(delta) >= 3:
                        arrow = "▲" if delta > 0 else "▼"
                        delta_lines.append(
                            f"  {asp}: {arrow}{abs(delta)}pp ({yr_a}→{yr_b})"
                        )
            if delta_lines:
                yoy_block = (
                    "\n\nYEAR-OVER-YEAR TRENDS (▲=improving ▼=worsening, ≥3pp shifts only):\n"
                    + "\n".join(delta_lines)
                )

        if 'sentiment' in main_df.columns:
            sc = main_df['sentiment'].value_counts().to_dict()
            overall_sent = (
                f"Positive: {sc.get('Positive', 0)}, "
                f"Neutral: {sc.get('Neutral', 0)}, "
                f"Negative: {sc.get('Negative', 0)}"
            )
        else:
            overall_sent = "N/A"

        if config['type'] == "🔄 Compare Sections (Same Course, Same Year)":
            context_str = (
                f"Course: {config['course']}, Year: {config['year']}. "
                f"Sections in this report: {', '.join(config['sections'])}. "
                f"Total reviews: {len(main_df)}. "
                f"Only refer to these sections — do not mention any other courses, sections, or years."
            )
        elif config['type'] == "📅 Compare Years (Same Course)":
            context_str = (
                f"Course: {config['course']}. "
                f"Years in this report: {', '.join(map(str, config['years']))}. "
                f"Total reviews: {len(main_df)}. "
                f"Only refer to these years — do not mention any other courses, sections, or years."
            )
        elif config['type'] == "🔬 Cross-Course Comparison":
            course_labels = [f"{c} ({y})" for c, y in config['courses']]
            context_str = (
                f"Courses in this report: {', '.join(course_labels)}. "
                f"Total reviews: {len(main_df)}. "
                f"Only refer to these courses — do not mention any other courses, sections, or years."
            )
        else:
            context_str = (
                f"Course: {config['course']}, Year: {config['year']}. "
                f"Total reviews: {len(main_df)}. All sections are pooled."
            )

        prompt = f"""Write a single cohesive faculty improvement report grounded strictly in the data below. \
Open with "Dear Instructor," exactly once — do not repeat it. Do not add a title, preamble, or closing note after the report ends.

=== CONTEXT ===
{context_str}
Overall sentiment: {overall_sent}.{yoy_block}{contrast_block}

=== OUTPUT FORMAT ===
ONE flowing report addressed to a single instructor (or faculty team). Structure:
1. "Dear Instructor," — once, at the very start. Name the course or {grp_label}(s) in the opening sentence.
2. 1-2 sentences summarising overall performance, noting which {grp_label}(s) performed strongest and which need the most attention.
3. For each major area of concern (work through the worst first, citing specific {grp_label} names and negative percentages):
   a. Describe the problem clearly.
   b. Give one concrete, actionable improvement strategy.
   c. If a CROSS-GROUP INSIGHT exists for this aspect, explicitly name the stronger {grp_label} and tell the instructor to review what that {grp_label} does differently.
   d. If [DOUBLE SIGNAL — treat as priority] appears, call it out as the top priority.
4. If year-over-year trends are present, include a sentence on what is improving and what is still declining.
5. Close with one encouraging sentence.

=== RULES ===
- Write the entire report as one block of plain prose — no repeated greetings, no bullet points, no bold, no headers, no hashtags, no sign-offs.
- Only name {grp_label}(s), course(s), and year(s) that appear in the DATA section below.
- Do not invent students counts, percentages, or course names not present in the data.
- Do not use placeholders like [NAME], [COURSE], or [YEAR] — use the actual values from the data.
- If data is limited, make your best analytical inference — never write "I cannot determine" or "insufficient data".

=== DATA ===
{chr(10).join(group_sections)}"""

        agg_summary = (
            adf.groupby(['aspect', s_col]).size().unstack(fill_value=0)
        )
        for c in ['Negative', 'Neutral', 'Positive']:
            if c not in agg_summary.columns:
                agg_summary[c] = 0
        agg_summary['total']   = agg_summary[['Negative', 'Neutral', 'Positive']].sum(axis=1)
        agg_summary['neg_pct'] = (agg_summary['Negative'] / agg_summary['total'] * 100).round(1)
        agg_summary['pos_pct'] = (agg_summary['Positive'] / agg_summary['total'] * 100).round(1)
        top_neg = agg_summary.nlargest(3, 'neg_pct')[['neg_pct', 'pos_pct', 'total']]

        return prompt, top_neg

    if st.button("🚀 Generate LLM Summary", type="primary", use_container_width=True):
        st.info(
            "⏳ LLM generation can take up to 5+ minutes depending on your hardware "
            "(primarily because the model runs on CPU when no GPU is available). "
            "Please do not refresh the page."
        )
        with st.spinner("Building prompt and loading LLM..."):
            try:
                prompt, top_neg = build_llm_prompt(adf, main_df, config, s_col)

                with st.expander("📋 Structured input sent to LLM"):
                    st.text(prompt)

                llm = LLMAnalyzer(model_name=LLM_MODEL, hf_token=HF_TOKEN)
                summary = llm.generate_summary(prompt, max_length=400)

                if config['type'] == "🔄 Compare Sections (Same Course, Same Year)":
                    allowed_groups = config['sections']
                elif config['type'] == "📅 Compare Years (Same Course)":
                    allowed_groups = [str(y) for y in config['years']]
                elif config['type'] == "🔬 Cross-Course Comparison":
                    allowed_groups = [f"{c} ({y})" for c, y in config['courses']]
                else:
                    allowed_groups = None

                summary = _re.sub(r'\[[^\]]{1,40}\]', '', summary)
                summary = _re.sub(r'#\w+', '', summary)
                summary = _re.sub(r'Best regards.*', '', summary, flags=_re.DOTALL | _re.IGNORECASE)
                cut_markers = [
                    'Please provide a concise',
                    'Use a structured format',
                    'Ensure the report is written',
                    'Remember, these are just',
                ]
                for marker in cut_markers:
                    idx = summary.find(marker)
                    if idx > 100:
                        summary = summary[:idx]
                        break
                last_stop = max(summary.rfind('.'), summary.rfind('!'), summary.rfind('?'))
                if last_stop > len(summary) // 2:
                    summary = summary[:last_stop + 1]
                summary = summary.strip()

                if allowed_groups is not None:
                    # Remove sentences that reference hallucinated group labels
                    for word in _re.findall(r'[A-Z]{2,5}\s*\d{3,4}|\(\d{4}\)', summary):
                        label = word.strip()
                        if not any(label in g for g in allowed_groups):
                            summary = _re.sub(
                                r'[^.!?]*' + _re.escape(label) + r'[^.!?]*[.!?]\s*',
                                '', summary
                            )
                    summary = summary.strip()

                st.markdown("---")
                st.markdown("#### 📝 Generated Summary")
                st.write(summary)

                st.markdown("#### 🔴 Top 3 Negative Aspects — Quick Reference")
                st.caption(
                    "Top 3 negative aspects per group, sorted by negative mention rate. "
                    "Use this alongside the report above."
                )

                if config['type'] == "🔄 Compare Sections (Same Course, Same Year)":
                    qr_grp_col, qr_grp_vals, qr_label = 'section', config['sections'], 'Section'
                elif config['type'] == "📅 Compare Years (Same Course)":
                    qr_grp_col, qr_grp_vals, qr_label = 'academic_year', sorted(config['years']), 'Year'
                elif config['type'] == "🔬 Cross-Course Comparison":
                    cy_qr = 'course_year' if 'course_year' in adf.columns else 'course_code'
                    qr_grp_col = cy_qr
                    qr_grp_vals = sorted(adf[cy_qr].dropna().unique().tolist())
                    qr_label = 'Course'
                else:
                    qr_grp_col, qr_grp_vals, qr_label = None, [None], 'Course'

                qr_rows = []
                for gv in qr_grp_vals:
                    sub_qr = adf[adf[qr_grp_col] == gv] if qr_grp_col else adf
                    grp_label_val = str(gv) if gv is not None else f"{config['course']} ({config['year']})"
                    asp_grp = sub_qr.groupby(['aspect', s_col]).size().unstack(fill_value=0)
                    for c in ['Negative', 'Neutral', 'Positive']:
                        if c not in asp_grp.columns:
                            asp_grp[c] = 0
                    asp_grp['total'] = asp_grp[['Negative', 'Neutral', 'Positive']].sum(axis=1)
                    asp_grp['neg_pct'] = (asp_grp['Negative'] / asp_grp['total'] * 100).round(1)
                    asp_grp['pos_pct'] = (asp_grp['Positive'] / asp_grp['total'] * 100).round(1)
                    top3 = asp_grp.nlargest(3, 'neg_pct')
                    for asp, row in top3.iterrows():
                        qr_rows.append({
                            qr_label: grp_label_val,
                            'Aspect': asp,
                            'Negative %': row['neg_pct'],
                            'Positive %': row['pos_pct'],
                            'Total Mentions': int(row['total']),
                        })

                if qr_rows:
                    qr_df = pd.DataFrame(qr_rows)
                    st.dataframe(qr_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ LLM generation failed: {e}")
                st.info("Make sure your HF_TOKEN is set and the Llama model is accessible.")
