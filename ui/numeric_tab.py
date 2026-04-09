import streamlit as st
import pandas as pd
import plotly.express as px


def render_numeric_tab(config):
    st.markdown("### 🔢 Numeric Rating Results")
    st.caption(
        "Based on Likert-scale responses from the evaluation form. "
        "Only the top response categories covering approximately 75% of responses are shown per question — "
        "less common responses are excluded to keep charts readable. "
        "This is why bars in the response distribution charts may not add up to 100%."
    )

    ndf = st.session_state.numeric_df.copy()

    if config['type'] == "📚 Single Course Analysis":
        ndf = ndf[
            (ndf['course_code']   == config['course']) &
            (ndf['academic_year'] == config['year'])
        ]
    elif config['type'] == "🔄 Compare Sections (Same Course, Same Year)":
        ndf = ndf[
            (ndf['course_code']   == config['course']) &
            (ndf['academic_year'] == config['year']) &
            (ndf['section'].isin(config['sections']))
        ]
    elif config['type'] == "📅 Compare Years (Same Course)":
        ndf = ndf[
            (ndf['course_code']   == config['course']) &
            (ndf['academic_year'].isin(config['years']))
        ]
    elif config['type'] == "🔬 Cross-Course Comparison":
        mask = pd.Series(False, index=ndf.index)
        for course_code, year in config['courses']:
            mask |= (
                (ndf['course_code'] == course_code) &
                (ndf['academic_year'] == year)
            )
        ndf = ndf[mask].copy()
        ndf['course_year'] = (
            ndf['course_code'] + ' (' +
            ndf['academic_year'].astype(str) + ')'
        )

    if len(ndf) == 0:
        st.warning("No numeric data found for the current selection.")
        return

    QUESTION_LABELS = {
        ('core',   1): 'Syllabus provided',
        ('core',   4): 'Course materials helped achieve objectives',
        ('core',   6): 'Tests/exams related to objectives',
        ('course', 4): 'Learning outcomes clearly stated & achieved',
        ('lect',   1): 'Instructor: clear & organized delivery',
        ('lect',   3): 'Instructor: students feel welcome to seek help',
        ('lect',   7): 'Instructor: overall effectiveness',
    }
    ndf['question_label'] = ndf.apply(
        lambda r: QUESTION_LABELS.get(
            (r['subsection'], r['question_number']),
            r['question_text'][:60]
        ), axis=1
    )

    ndf = ndf.sort_values('answer_value', ascending=False)

    st.markdown("#### 📋 Mean Scores by Question")

    if config['type'] == "📚 Single Course Analysis":
        grp_col = 'section'
    elif config['type'] == "🔄 Compare Sections (Same Course, Same Year)":
        grp_col = 'section'
    elif config['type'] == "🔬 Cross-Course Comparison":
        grp_col = 'course_year'
    else:
        grp_col = 'academic_year'

    means = (
        ndf.drop_duplicates(
            subset=['question_label', grp_col, 'mean']
        )
        .pivot_table(
            index='question_label',
            columns=grp_col,
            values='mean',
            aggfunc='first'
        )
        .round(2)
    )

    styled_means = means.style.background_gradient(
        cmap='RdYlGn', vmin=1, vmax=7, axis=None
    ).format('{:.2f}')
    st.dataframe(styled_means, use_container_width=True)

    st.divider()

    st.markdown("#### 📊 Response Distribution per Question")
    st.caption(
        "Bars show the percentage of students selecting each rating, as reported "
        "in the evaluation PDF. Only the top response categories are shown — these "
        "cover approximately 75% of all responses per question, so bars will not "
        "add up to 100%. The remaining percentage represents less common responses "
        "that were excluded to keep the chart readable."
    )

    ANSWER_COLOURS = {
        'Strongly Agree':            '#1a7a3d',
        'Strongly agree':            '#1a7a3d',
        'Agree':                     '#4CAF50',
        'Somewhat Agree':            '#a8d5b5',
        'Somewhat agree':            '#a8d5b5',
        'Neither Agree nor Disagree':'#FFC107',
        'Neither agree nor disagree':'#FFC107',
        'Somewhat Disagree':         '#f4a261',
        'Somewhat disagree':         '#f4a261',
        'Disagree':                  '#F44336',
        'Strongly Disagree':         '#b71c1c',
        'Strongly disagree':         '#b71c1c',
    }

    SUBSECTION_NAMES = {
        'core':   '🏛️ Core Institutional Questions',
        'course': '📚 Course Level Questions',
        'lect':   '👩‍🏫 Instructor (LECT 01)',
    }

    for sub_key, sub_label in SUBSECTION_NAMES.items():
        sub_ndf = ndf[ndf['subsection'] == sub_key]
        if len(sub_ndf) == 0:
            continue

        st.markdown(f"##### {sub_label}")

        if config['type'] == "🔬 Cross-Course Comparison":
            cy_vals = sorted(sub_ndf['course_year'].unique())
            for cy in cy_vals:
                cy_ndf = sub_ndf[sub_ndf['course_year'] == cy].drop_duplicates(
                    subset=['question_label', 'answer_label']
                )
                fig = px.bar(
                    cy_ndf,
                    x='question_label', y='percentage',
                    color='answer_label', barmode='stack',
                    title=f'{sub_label} — {cy}',
                    color_discrete_map=ANSWER_COLOURS,
                    labels={
                        'percentage':     '% of Responses',
                        'question_label': 'Question',
                        'answer_label':   'Rating',
                    }
                )
                fig.update_layout(
                    xaxis_tickangle=-30,
                    legend_title='Rating',
                    height=380,
                    yaxis_range=[0, 100],
                )
                st.plotly_chart(fig, use_container_width=True)

        elif config['type'] in [
            "🔄 Compare Sections (Same Course, Same Year)",
            "📅 Compare Years (Same Course)"
        ]:
            plot_ndf = sub_ndf.drop_duplicates(
                subset=['question_label', 'answer_label', grp_col]
            )
            fig = px.bar(
                plot_ndf,
                x='question_label', y='percentage',
                color='answer_label', barmode='stack',
                facet_col=grp_col,
                title=f'{sub_label} — Response % by {grp_col}',
                color_discrete_map=ANSWER_COLOURS,
                labels={
                    'percentage':    '% of Responses',
                    'question_label': 'Question',
                    'answer_label':  'Rating',
                }
            )
            fig.update_layout(legend_title='Rating', height=420, yaxis_range=[0, 100])
            fig.for_each_yaxis(lambda ya: ya.update(range=[0, 100]))
            fig.for_each_xaxis(lambda ax: ax.update(tickangle=-30))
            st.plotly_chart(fig, use_container_width=True)

        else:
            plot_ndf = sub_ndf.drop_duplicates(
                subset=['question_label', 'answer_label']
            )
            fig = px.bar(
                plot_ndf,
                x='question_label', y='percentage',
                color='answer_label', barmode='stack',
                title=sub_label,
                color_discrete_map=ANSWER_COLOURS,
                labels={
                    'percentage':    '% of Responses',
                    'question_label': 'Question',
                    'answer_label':  'Rating',
                }
            )
            fig.update_layout(
                xaxis_tickangle=-30,
                legend_title='Rating',
                height=400,
                yaxis_range=[0, 100],
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    if config['type'] == "📅 Compare Years (Same Course)":
        st.markdown("#### 📈 Mean Score Trend Over Years")
        st.caption("Rising = improving student ratings")

        trend_means = (
            ndf.drop_duplicates(
                subset=['question_label', 'academic_year', 'mean']
            )[['question_label', 'academic_year', 'mean']]
        )
        trend_means['academic_year'] = trend_means['academic_year'].astype(str)

        fig_tr = px.line(
            trend_means,
            x='academic_year',
            y='mean',
            color='question_label',
            markers=True,
            title=f'Mean Rating Trend — {config["course"]}',
            labels={
                'mean':          'Mean Score (1–7)',
                'academic_year': 'Year',
                'question_label': 'Question',
            }
        )
        fig_tr.update_layout(yaxis_range=[1, 7])
        st.plotly_chart(fig_tr, use_container_width=True)
