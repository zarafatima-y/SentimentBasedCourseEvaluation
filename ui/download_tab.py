import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import os
from datetime import datetime


from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable
)
from reportlab.platypus import KeepTogether


NAVY     = colors.HexColor('#1E3A8A')
BLUE     = colors.HexColor('#2563EB')
GREEN    = colors.HexColor('#4CAF50')
RED      = colors.HexColor('#F44336')
AMBER    = colors.HexColor('#FFC107')
LIGHT_BG = colors.HexColor('#EFF6FF')
GREY     = colors.HexColor('#6B7280')


def get_styles():
    base = getSampleStyleSheet()
    custom = {
        'Title': ParagraphStyle(
            'ReportTitle', parent=base['Title'],
            fontSize=22, textColor=NAVY, spaceAfter=6,
            alignment=TA_CENTER, fontName='Helvetica-Bold'
        ),
        'Subtitle': ParagraphStyle(
            'Subtitle', parent=base['Normal'],
            fontSize=11, textColor=GREY, spaceAfter=16,
            alignment=TA_CENTER
        ),
        'H1': ParagraphStyle(
            'H1', parent=base['Heading1'],
            fontSize=14, textColor=NAVY, spaceBefore=14,
            spaceAfter=6, fontName='Helvetica-Bold',
            borderPad=4
        ),
        'H2': ParagraphStyle(
            'H2', parent=base['Heading2'],
            fontSize=12, textColor=BLUE, spaceBefore=10,
            spaceAfter=4, fontName='Helvetica-Bold'
        ),
        'Body': ParagraphStyle(
            'Body', parent=base['Normal'],
            fontSize=9, spaceAfter=6, leading=14,
            alignment=TA_JUSTIFY
        ),
        'Caption': ParagraphStyle(
            'Caption', parent=base['Normal'],
            fontSize=8, textColor=GREY, spaceAfter=8,
            leading=12, alignment=TA_LEFT
        ),
        'Metric': ParagraphStyle(
            'Metric', parent=base['Normal'],
            fontSize=10, spaceAfter=4, fontName='Helvetica-Bold'
        ),
        'LLM': ParagraphStyle(
            'LLM', parent=base['Normal'],
            fontSize=9, spaceAfter=8, leading=14,
            leftIndent=12, rightIndent=12,
            borderPad=6, alignment=TA_JUSTIFY
        ),
    }
    return custom


def section_header(text, styles):
    """Returns a styled section header with a divider line."""
    return [
        Paragraph(text, styles['H1']),
        HRFlowable(width='100%', thickness=1, color=BLUE, spaceAfter=6),
    ]


def fig_to_image(fig, width_cm=15, height_cm=8):
    """Convert a plotly figure to a ReportLab Image object."""
    try:
        img_bytes = fig.to_image(format='png', width=900, height=480, scale=2)
        buf = io.BytesIO(img_bytes)
        return Image(buf, width=width_cm * cm, height=height_cm * cm)
    except Exception:
        return None


def mpl_to_image(fig, width_cm=15, height_cm=8):
    """Convert a matplotlib figure to a ReportLab Image object."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        return Image(buf, width=width_cm * cm, height=height_cm * cm)
    except Exception:
        return None


def df_to_table(df, styles, col_widths=None):
    """Convert a DataFrame to a styled ReportLab Table."""
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, 0),  NAVY),
        ('TEXTCOLOR',   (0, 0), (-1, 0),  colors.white),
        ('FONTNAME',    (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',    (0, 0), (-1, 0),  8),
        ('ALIGN',       (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE',    (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F4FF')]),
        ('GRID',        (0, 0), (-1, -1), 0.4, colors.HexColor('#D1D5DB')),
        ('TOPPADDING',  (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return t




def build_sentiment_chart(df, config):
    """Build overall sentiment pie chart."""
    COLOR_MAP = {'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'}
    counts = df['sentiment'].value_counts().reset_index()
    counts.columns = ['Sentiment', 'Count']
    fig = px.pie(counts, values='Count', names='Sentiment',
                 color='Sentiment', color_discrete_map=COLOR_MAP,
                 title='Overall Sentiment Distribution')
    fig.update_layout(margin=dict(t=40, b=20, l=20, r=20))
    return fig


def build_aspect_bar(adf, sentiment_col, config):
    """Build aspect frequency bar chart."""
    counts = adf['aspect'].value_counts().reset_index()
    counts.columns = ['Aspect', 'Count']
    fig = px.bar(counts, x='Aspect', y='Count',
                 title='Aspect Mention Frequency',
                 color='Count', color_continuous_scale='Viridis')
    fig.update_layout(xaxis_tickangle=-35, margin=dict(t=40, b=60, l=20, r=20))
    return fig


def build_emotion_chart(df):
    """Build emotion distribution chart."""
    ec = df['dominant_emotion'].value_counts().reset_index()
    ec.columns = ['Emotion', 'Count']
    fig = px.bar(ec, x='Emotion', y='Count',
                 title='Emotion Distribution',
                 color='Emotion')
    fig.update_layout(margin=dict(t=40, b=40, l=20, r=20))
    return fig


def build_aspect_sentiment_balance(adf, sentiment_col):
    """Build aspect sentiment balance chart."""
    bal_rows = []
    for asp in sorted(adf['aspect'].dropna().unique()):
        asp_sub = adf[adf['aspect'] == asp]
        total = len(asp_sub)
        if total == 0:
            continue
        pos_pct = round((asp_sub[sentiment_col] == 'Positive').sum() / total * 100, 1)
        neg_pct = round((asp_sub[sentiment_col] == 'Negative').sum() / total * 100, 1)
        bal_rows.append({'Aspect': asp, 'Positive %': pos_pct, 'Negative %': neg_pct})
    if not bal_rows:
        return None
    bal_df = pd.DataFrame(bal_rows).sort_values('Negative %', ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Positive %', x=bal_df['Aspect'],
                         y=bal_df['Positive %'], marker_color='#4CAF50'))
    fig.add_trace(go.Bar(name='Negative %', x=bal_df['Aspect'],
                         y=bal_df['Negative %'], marker_color='#F44336'))
    fig.update_layout(
        barmode='group', title='Aspect Sentiment Balance',
        xaxis_tickangle=-35, yaxis_range=[0, 100],
        margin=dict(t=40, b=80, l=20, r=20)
    )
    return fig



def build_pdf(config, run_options):
    """
    Build a comprehensive PDF report from session state data.
    Returns bytes of the PDF.
    """
    buf     = io.BytesIO()
    styles  = get_styles()
    story   = []
    df      = st.session_state.df
    adf     = st.session_state.aspect_df
    num_df  = st.session_state.numeric_df
    al      = st.session_state.analysis_long

    sentiment_col = (
        'aspect_sentiment'
        if adf is not None and 'aspect_sentiment' in adf.columns
        else 'sentiment'
    )

    if config['type'] == "📚 Single Course Analysis":
        report_title = f"{config['course']} ({config['year']}) — Course Evaluation Report"
        context_line = f"Single Course Analysis · All sections pooled · {len(df)} reviews"
    elif config['type'] == "🔄 Compare Sections (Same Course, Same Year)":
        report_title = f"{config['course']} ({config['year']}) — Section Comparison Report"
        context_line = f"Sections: {', '.join(config['sections'])} · {len(df)} reviews"
    elif config['type'] == "📅 Compare Years (Same Course)":
        report_title = f"{config['course']} — Year Comparison Report"
        context_line = f"Years: {', '.join(map(str, config['years']))} · {len(df)} reviews"
    else:
        labels = [f"{c} ({y})" for c, y in config['courses']]
        report_title = "Cross-Course Comparison Report"
        context_line = f"{' vs '.join(labels)} · {len(df)} reviews"

    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("Sentiment Based Course Evaluation Analysis System", styles['Title']))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(report_title, styles['H1']))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(context_line, styles['Subtitle']))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
        styles['Subtitle']
    ))
    story.append(Spacer(1, 1 * cm))

    
    if 'sentiment' in df.columns:
        sc = df['sentiment'].value_counts().to_dict()
        metrics_data = [
            ['Metric', 'Value'],
            ['Total Reviews', str(len(df))],
            ['Positive Reviews', str(sc.get('Positive', 0))],
            ['Neutral Reviews',  str(sc.get('Neutral', 0))],
            ['Negative Reviews', str(sc.get('Negative', 0))],
            ['Analysis Type', config['type'].split(' ', 1)[-1]],
        ]
        if adf is not None:
            metrics_data.append(['Aspect Mentions', str(len(adf))])
        if 'dominant_emotion' in df.columns:
            metrics_data.append(['Most Common Emotion', df['dominant_emotion'].mode()[0]])

        mt = Table(metrics_data, colWidths=[8 * cm, 8 * cm])
        mt.setStyle(TableStyle([
            ('BACKGROUND',  (0, 0), (-1, 0),  NAVY),
            ('TEXTCOLOR',   (0, 0), (-1, 0),  colors.white),
            ('FONTNAME',    (0, 0), (-1, 0),  'Helvetica-Bold'),
            ('FONTSIZE',    (0, 0), (-1, -1), 9),
            ('ALIGN',       (0, 0), (-1, -1), 'CENTER'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ('GRID',        (0, 0), (-1, -1), 0.4, GREY),
            ('TOPPADDING',  (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(mt)

    story.append(PageBreak())

    if run_options.get('sentiment') and 'sentiment' in df.columns:
        story += section_header('1. Sentiment Analysis', styles)
        story.append(Paragraph(
            'The sentiment analysis classifies each student review as Positive, Neutral, or Negative '
            'using a RoBERTa-based model. The pie chart below shows the overall distribution across '
            'all reviews in this analysis. A high positive proportion indicates generally favourable '
            'student experience; a high negative proportion signals areas requiring attention.',
            styles['Body']
        ))

        fig_sent = build_sentiment_chart(df, config)
        img_sent = fig_to_image(fig_sent, width_cm=14, height_cm=8)
        if img_sent:
            story.append(img_sent)
            story.append(Paragraph(
                'Figure 1: Overall sentiment distribution. Green = Positive, Yellow = Neutral, Red = Negative.',
                styles['Caption']
            ))

        if config['type'] in ["🔄 Compare Sections (Same Course, Same Year)",
                               "📅 Compare Years (Same Course)",
                               "🔬 Cross-Course Comparison"]:
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph('Sentiment Breakdown by Group', styles['H2']))

            if config['type'] == "🔄 Compare Sections (Same Course, Same Year)":
                grp_col = 'section'
            elif config['type'] == "📅 Compare Years (Same Course)":
                grp_col = 'academic_year'
            else:
                grp_col = 'course_year' if 'course_year' in df.columns else 'course_code'

            breakdown_rows = []
            for gv in sorted(df[grp_col].dropna().unique()):
                sub = df[df[grp_col] == gv]
                total = len(sub)
                if total == 0:
                    continue
                breakdown_rows.append({
                    'Group': str(gv),
                    'Total': total,
                    'Positive': f"{(sub['sentiment']=='Positive').sum()} ({(sub['sentiment']=='Positive').mean()*100:.1f}%)",
                    'Neutral':  f"{(sub['sentiment']=='Neutral').sum()} ({(sub['sentiment']=='Neutral').mean()*100:.1f}%)",
                    'Negative': f"{(sub['sentiment']=='Negative').sum()} ({(sub['sentiment']=='Negative').mean()*100:.1f}%)",
                })
            if breakdown_rows:
                bd_df = pd.DataFrame(breakdown_rows)
                story.append(df_to_table(bd_df, styles))
                story.append(Paragraph(
                    'Table 1: Sentiment counts and percentages per group.',
                    styles['Caption']
                ))

        story.append(PageBreak())

    if run_options.get('aspect') and adf is not None and len(adf) > 0:
        story += section_header('2. Aspect-Based Sentiment Analysis', styles)
        story.append(Paragraph(
            'Aspect-based sentiment analysis identifies which course dimensions students are discussing '
            'and whether their comments about each dimension are positive, neutral, or negative. '
            'Ten aspects are tracked: instructor, difficulty/challenge, assignments/labs, assessments, '
            'course content, workload/pace, resources/materials, support/help, engagement/interest, '
            'and learning outcomes.',
            styles['Body']
        ))

    
        fig_asp = build_aspect_bar(adf, sentiment_col, config)
        img_asp = fig_to_image(fig_asp, width_cm=15, height_cm=7)
        if img_asp:
            story.append(img_asp)
            story.append(Paragraph(
                'Figure 2: Aspect mention frequency. Taller bars indicate aspects discussed more often.',
                styles['Caption']
            ))

        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph('Aspect Sentiment Balance', styles['H2']))
        story.append(Paragraph(
            'This chart shows for each aspect what percentage of its mentions were positive (green) '
            'versus negative (red). The gap between the two bars represents neutral mentions. '
            'Aspects sorted by negative rate — leftmost aspects have the highest student concern.',
            styles['Body']
        ))
        fig_bal = build_aspect_sentiment_balance(adf, sentiment_col)
        if fig_bal:
            img_bal = fig_to_image(fig_bal, width_cm=15, height_cm=8)
            if img_bal:
                story.append(img_bal)
                story.append(Paragraph(
                    'Figure 3: Aspect sentiment balance. Green = % positive mentions, Red = % negative mentions. '
                    'Bars do not add to 100% because neutral mentions fill the gap.',
                    styles['Caption']
                ))

        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph('Top Negative Aspects', styles['H2']))
        asp_summary = adf.groupby(['aspect', sentiment_col]).size().unstack(fill_value=0)
        for c in ['Negative', 'Neutral', 'Positive']:
            if c not in asp_summary.columns:
                asp_summary[c] = 0
        asp_summary['Total']   = asp_summary[['Negative', 'Neutral', 'Positive']].sum(axis=1)
        asp_summary['Neg %']   = (asp_summary['Negative'] / asp_summary['Total'] * 100).round(1)
        asp_summary['Pos %']   = (asp_summary['Positive'] / asp_summary['Total'] * 100).round(1)
        top_neg = asp_summary.nlargest(5, 'Neg %')[['Neg %', 'Pos %', 'Total']].reset_index()
        top_neg.columns = ['Aspect', 'Negative %', 'Positive %', 'Total Mentions']
        story.append(df_to_table(top_neg, styles))
        story.append(Paragraph(
            'Table 2: Top 5 aspects by negative mention rate.',
            styles['Caption']
        ))

        story.append(PageBreak())

    if run_options.get('emotion') and 'dominant_emotion' in df.columns:
        story += section_header('3. Emotion Analysis', styles)
        story.append(Paragraph(
            'Emotion detection identifies the dominant emotional tone of each review across six '
            'categories: joy, anger, fear, sadness, surprise, and neutral. This adds interpretive '
            'depth beyond polarity — for example, negative sentiment driven by anger (perceived '
            'unfairness) requires a different response than negative sentiment driven by sadness '
            '(personal academic struggle).',
            styles['Body']
        ))

        fig_emo = build_emotion_chart(df)
        img_emo = fig_to_image(fig_emo, width_cm=14, height_cm=7)
        if img_emo:
            story.append(img_emo)
            story.append(Paragraph(
                'Figure 4: Dominant emotion distribution across all reviews.',
                styles['Caption']
            ))

        if 'sentiment' in df.columns:
            story.append(Spacer(1, 0.3 * cm))
            pivot = pd.crosstab(df['dominant_emotion'], df['sentiment'])
            for c in ['Positive', 'Neutral', 'Negative']:
                if c not in pivot.columns:
                    pivot[c] = 0
            pivot = pivot[['Positive', 'Neutral', 'Negative']].reset_index()
            pivot.columns.name = None
            story.append(Paragraph('Emotion × Sentiment Cross-Tabulation', styles['H2']))
            story.append(Paragraph(
                'This table shows how emotions co-occur with sentiment labels. '
                'Joy should cluster with Positive; anger and fear with Negative. '
                'Unexpected patterns (e.g. fear in Positive reviews) signal nuanced feedback.',
                styles['Body']
            ))
            story.append(df_to_table(pivot, styles))
            story.append(Paragraph(
                'Table 3: Count of reviews per emotion-sentiment combination.',
                styles['Caption']
            ))

        story.append(PageBreak())


    llm_summary = st.session_state.get('llm_summary_text')
    if llm_summary:
        story += section_header('4. LLM-Generated Course Improvement Report', styles)
        story.append(Paragraph(
            'The following report was generated by Meta Llama 3 8B Instruct based on the '
            'aspect-sentiment data from this analysis. It is addressed to course instructors '
            'and provides plain-language improvement recommendations grounded in student feedback.',
            styles['Body']
        ))
        story.append(Spacer(1, 0.3 * cm))

        for para in llm_summary.split('\n\n'):
            para = para.strip()
            if para:
                story.append(Paragraph(para, styles['LLM']))

        story.append(PageBreak())

    rq2_data = st.session_state.get('rq2_results')
    if rq2_data is not None:
        story += section_header('5. RQ2 — Aspect Predictors of Negative Evaluations', styles)
        story.append(Paragraph(
            'This section addresses the research question: which course aspects most strongly '
            'predict whether a group\'s overall evaluation is negative? Following Pang, Lee & '
            'Vaithyanathan (2002) and Schouten & Frasincar (2016), OLS linear regression is '
            'applied at the group level. Standardised coefficients indicate relative predictive '
            'strength — positive values predict more negative evaluations, negative values suggest '
            'a protective or dampening effect.',
            styles['Body']
        ))

        coef_df         = rq2_data.get('coef_df')
        r2              = rq2_data.get('r2')
        n_grps          = rq2_data.get('n_groups')
        interp          = rq2_data.get('interp', '')
        group_summaries = rq2_data.get('group_summaries', [])

        if coef_df is not None and len(coef_df) > 0:
            story.append(Spacer(1, 0.3 * cm))

            if r2 is not None and n_grps is not None and n_grps > 2:
                story.append(Paragraph(
                    f"Model fit: R2 = {r2:.3f} ({r2*100:.1f}% of variance explained). "
                    f"Number of groups: {n_grps}.",
                    styles['Body']
                ))

            display_coef = coef_df.copy()
            display_coef['Direction'] = display_coef['Coefficient'].apply(
                lambda x: 'Predicts negative' if x > 0 else 'Protective'
            )
            story.append(df_to_table(display_coef, styles))
            story.append(Paragraph(
                'Table 4: Standardised OLS coefficients. Sorted by coefficient value. '
                'Focus on direction and relative size rather than absolute values.',
                styles['Caption']
            ))

        if interp:
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph('Overall Interpretation', styles['H2']))
            story.append(Paragraph(interp, styles['Body']))

        if group_summaries:
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph('Group-Specific Findings', styles['H2']))
            story.append(Paragraph(
                "For each group, the aspect with the highest negative mention rate is identified "
                "alongside the group's overall negative review rate.",
                styles['Body']
            ))
            for gs in group_summaries:
                story.append(Paragraph(gs['summary'], styles['Body']))
                story.append(Spacer(1, 0.15 * cm))

        story.append(PageBreak())

    if num_df is not None and len(num_df) > 0:
        story += section_header('6. Numeric Survey Ratings', styles)
        story.append(Paragraph(
            'Likert-scale survey responses (1–7) provide a quantitative complement to the '
            'essay-based analysis. Only the top response categories covering approximately '
            '75% of responses are shown. Scores below 5.0/7 are flagged as potential concerns.',
            styles['Body']
        ))

        QUESTION_LABELS = {
            ('core',   1): 'Syllabus provided',
            ('core',   4): 'Course materials helped achieve objectives',
            ('core',   6): 'Tests/exams related to objectives',
            ('course', 4): 'Learning outcomes clearly stated & achieved',
            ('lect',   1): 'Instructor: clear & organized delivery',
            ('lect',   3): 'Instructor: students feel welcome to seek help',
            ('lect',   7): 'Instructor: overall effectiveness',
        }
        ndf = num_df.copy()
        if config['type'] == "📚 Single Course Analysis":
            ndf = ndf[(ndf['course_code'] == config['course']) &
                      (ndf['academic_year'] == config['year'])]
            grp_col_n = 'section'
        elif config['type'] == "🔄 Compare Sections (Same Course, Same Year)":
            ndf = ndf[(ndf['course_code'] == config['course']) &
                      (ndf['academic_year'] == config['year']) &
                      (ndf['section'].isin(config['sections']))]
            grp_col_n = 'section'
        elif config['type'] == "📅 Compare Years (Same Course)":
            ndf = ndf[(ndf['course_code'] == config['course']) &
                      (ndf['academic_year'].isin(config['years']))]
            grp_col_n = 'academic_year'
        else:
            mask = pd.Series(False, index=ndf.index)
            for cc, yy in config['courses']:
                mask |= (ndf['course_code'] == cc) & (ndf['academic_year'] == yy)
            ndf = ndf[mask].copy()
            ndf['course_year'] = ndf['course_code'] + ' (' + ndf['academic_year'].astype(str) + ')'
            grp_col_n = 'course_year'

        if len(ndf) > 0:
            ndf['question_label'] = ndf.apply(
                lambda r: QUESTION_LABELS.get(
                    (r['subsection'], r['question_number']),
                    r['question_text'][:50]
                ), axis=1
            )
            means = (
                ndf.drop_duplicates(subset=['question_label', grp_col_n, 'mean'])
                .pivot_table(index='question_label', columns=grp_col_n,
                             values='mean', aggfunc='first')
                .round(2)
            )
            means_reset = means.reset_index()
            means_reset.columns.name = None
            story.append(df_to_table(means_reset, styles))
            story.append(Paragraph(
                'Table 5: Mean Likert scores (1–7) per question and group. '
                'Scores below 5.0 indicate potential concern areas.',
                styles['Caption']
            ))

        story.append(PageBreak())

    story += section_header('References', styles)
    refs = [
        'AI@Meta (2024). Llama 3 Model Card. https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md',
    ]
    for ref in refs:
        story.append(Paragraph(ref, styles['Body']))
        story.append(Spacer(1, 0.2 * cm))

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm,   bottomMargin=2 * cm,
        title=report_title,
        author='Course Evaluation Analysis System',
    )

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(GREY)
        canvas.drawString(2 * cm, 1.2 * cm,
                          f'Course Evaluation Analysis System — {datetime.now().strftime("%B %Y")}')
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f'Page {doc.page}')
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    buf.seek(0)
    return buf.read()


def render_download_tab(config):
    st.markdown("### Download Results")

    aspect_df   = st.session_state.aspect_df
    run_options = st.session_state.get('run_options', {})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Raw Data Downloads")
        if st.session_state.df is not None:
            st.download_button(
                "📥 Full Analysis (CSV)",
                data=st.session_state.df.to_csv(index=False),
                file_name="full_analysis.csv", mime="text/csv",
                use_container_width=True
            )
        if aspect_df is not None and len(aspect_df) > 0:
            st.download_button(
                "📥 Aspect Analysis (CSV)",
                data=aspect_df.to_csv(index=False),
                file_name="aspect_analysis.csv", mime="text/csv",
                use_container_width=True
            )
        if st.session_state.analysis_long is not None:
            st.download_button(
                "📥 Long Format (CSV)",
                data=st.session_state.analysis_long.to_csv(index=False),
                file_name="analysis_long.csv", mime="text/csv",
                use_container_width=True
            )
        if st.session_state.numeric_df is not None and len(st.session_state.numeric_df) > 0:
            st.download_button(
                "📥 Numeric Ratings (CSV)",
                data=st.session_state.numeric_df.to_csv(index=False),
                file_name="numeric_ratings.csv", mime="text/csv",
                use_container_width=True
            )

    with col2:
        st.markdown("#### 📈 Session Summary")
        df = st.session_state.df
        summary = {
            'Analysis Type':  config['type'],
            'Total Reviews':  len(df),
            'Courses':        df['course_code'].nunique(),
            'Years':          df['academic_year'].nunique(),
            'Sections':       df['section'].nunique(),
        }
        if config['type'] == "📚 Single Course Analysis":
            summary['Course'] = config['course']
            summary['Year']   = config['year']
        if 'sentiment' in df.columns:
            sc = df['sentiment'].value_counts().to_dict()
            summary['Positive Reviews'] = sc.get('Positive', 0)
            summary['Neutral Reviews']  = sc.get('Neutral', 0)
            summary['Negative Reviews'] = sc.get('Negative', 0)
        if aspect_df is not None:
            summary['Aspect Mentions'] = len(aspect_df)
            summary['Unique Aspects']  = aspect_df['aspect'].nunique()
        if 'dominant_emotion' in df.columns:
            summary['Most Common Emotion'] = df['dominant_emotion'].mode()[0]

        summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.divider()
    
    st.markdown("#### 📄 Comprehensive PDF Report")
    st.caption(
        "Generates a full report including all visualizations, explanatory text, "
        "sentiment and aspect analysis, emotion analysis, LLM improvement report "
        "(if generated), RQ2 findings, and numeric survey results. "
        "Chart generation may take 15–30 seconds."
    )

    llm_generated = bool(st.session_state.get('llm_summary_text'))
    rq2_generated = st.session_state.get('rq2_results') is not None

    if not llm_generated:
        st.info(
            "💡 LLM Summary not included — generate it in the LLM Summary tab first "
            "if you want it in the report."
        )

    if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
        with st.spinner("Building PDF — generating charts and compiling report..."):
            try:
                pdf_bytes = build_pdf(config, run_options)
                filename = (
                    f"course_evaluation_report_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                )
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.success("✅ PDF ready — click the button above to download.")
            except Exception as e:
                st.error(f"❌ PDF generation failed: {e}")
                st.info(
                    "Make sure kaleido is installed: pip install kaleido. "
                    "If running on Streamlit Cloud, add kaleido to requirements.txt."
                )