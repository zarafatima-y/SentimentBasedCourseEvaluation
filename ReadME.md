## Sentiment Based Course Evaluation Analysis System

A Streamlit web application that analyses student course evaluation PDFs using NLP, sentiment analysis, emotion detection, aspect-level analysis, and LLM-generated faculty improvement reports.

**Live App:** [sentimentbasedcourseevaluation.streamlit.app](https://sentimentbasedcourseevaluation.streamlit.app/)

---

## What It Does

Upload one or more course evaluation PDFs and the application will:

- Extract and clean student written reviews automatically
- Run whole-text sentiment analysis (Positive / Neutral / Negative)
- Detect the dominant emotion per review (joy, sadness, anger, fear, surprise, neutral)
- Break down sentiment by course aspect using a custom-built ABSA pipeline (no pre-trained aspect model). Reviews are split into sentences, a keyword dictionary maps each sentence to one of ten course aspects, and the same RoBERTa sentiment model used for whole-text analysis evaluates each sentence independently
- Present **RQ1** as a project finding showing why aspect-based analysis reveals nuance that whole-text sentiment can miss
- Compare results across sections, years, or different courses
- Identify which aspects are most associated with negative evaluations across global, comparative, and per-group views (**RQ2**)
- Generate a plain-English faculty improvement report using an LLM
- Export all results as CSV files or a comprehensive PDF report

---

## Accessing the App

**Web (no setup required):** Visit [sentimentbasedcourseevaluation.streamlit.app](https://sentimentbasedcourseevaluation.streamlit.app/) in any browser.

**Local development:** See the [Local Setup](#setup--local-development) section below.

---

## How to Use the App

The app walks you through four stages in the sidebar: Upload, Clean, Analyze, and Results.

**Stage 1 — Upload.** Upload one or more course evaluation PDFs and click **Process PDFs**. Each PDF must contain both an `ESSAY RESULTS` section and a `NUMERIC RESULTS` section in the same report. The app was built against that combined format. If only essay or only numeric content is present, the app will stop with a user-facing error instead of continuing.

**Stage 2 — Clean.** Review the extracted data, select cleaning options (remove nulls, remove very short reviews, normalize text), and click **Run Preprocessing**.

**Stage 3 — Analyze.** Select your analysis type, the courses or sections you want to include, and which analysis modules to run (sentiment, aspect, emotion). Click **Run Selected Analysis**.

**Stage 4 — Results.** Results are displayed in a tabbed dashboard described below.

### Step-by-Step Navigation

1. **Upload**
   Choose one or more PDFs and confirm the filenames shown on screen.
2. **Process PDFs**
   Click **Process PDFs**. If extraction succeeds, the app moves automatically to the cleaning stage.
3. **Check the extracted rows**
   Use the preview table to confirm that course code, year, section, question text, and review text were captured correctly.
4. **Run cleaning**
   Leave the default cleaning options on unless you have a specific reason to change them, then click **Run Preprocessing**.
5. **Choose analysis type**
   Decide whether you want a single-course view, section comparison, year comparison, or cross-course comparison.
6. **Select analysis modules**
   Enable the modules you want to run. Sentiment and aspect are the most important for the research-question tabs.
7. **Generate results**
   Click **Run Selected Analysis** and wait for the app to finish building the dashboard.
8. **Read the dashboard from left to right**
   Start with Overview, then move through Sentiment / Aspects / Emotions / LLM Summary / RQ1 / RQ2 / Numeric / Download depending on what was generated.
9. **Generate written outputs if needed**
   In the LLM tab, click the summary-generation control before downloading if you want the generated report included in the PDF export.
10. **Download**
    Use the Download tab for CSV exports and the full PDF report.

### How to Read the Graphs

- **Pie charts / grouped bars** show overall proportions or counts for sentiment categories.
- **Heatmaps** show where values cluster. In sentiment-style heatmaps, stronger diagonal values mean stronger agreement; off-diagonal values indicate disagreement or mixed patterns.
- **Radar charts** compare the same aspects across groups on one shape. Larger distance from the centre means a larger value.
- **Trend lines** are most useful for year comparisons; focus on direction over time rather than tiny point-to-point changes.
- **Aspect balance charts** compare positive and negative rates for the same aspect. Large gaps indicate clearer student consensus.

### File Requirement Note

Do **not** upload essay-only PDFs or numeric-only PDFs. The current app expects the original combined course-evaluation report format containing both sections. If a file is missing one section, extraction will stop and a friendly error message will be shown.

---

## Analysis Types

- **Single Course Analysis** — one course in one year, all sections pooled.
- **Compare Sections** — 2 to 5 sections of the same course in the same year.
- **Compare Years** — same course across 2 to 5 years.
- **Cross-Course Comparison** — 2 to 5 different course–year combinations.

---

## Results Dashboard — Tab by Tab

### Overview
Total reviews, courses, years, and sections loaded, plus a review count breakdown by group. For Compare Years, also shows overall sentiment trend lines and per-aspect negative sentiment trends over time.

### Sentiment
- **Individual pie charts** — one per group showing that group's sentiment split.
- **Grouped bar chart** — all groups side by side with raw counts.
- **Trend line** (Compare Years only) — how each sentiment category changed over time.
- **Aspect Sentiment Balance** — per-aspect positive vs negative mention percentages, sorted from most negative to least so priority concerns appear first.

### Aspects
- **Aspect Frequency bar chart** — raw count of mentions per aspect (not normalised).
- **Aspect–Sentiment Heatmap** — group–aspect pairs with Negative / Neutral / Positive counts.
- **Radar charts** — Aspect Counts, Positive %, and Negative % side by side for fair cross-group comparison.
- **Survey Question Breakdown table** — sentiment and top aspects grouped by the question students were answering.

### Emotions
- **Emotion distribution chart** — frequency of joy, anger, sadness, fear, surprise, neutral.
- **Emotion × Sentiment heatmap** — how emotions co-occur with sentiment labels.
- **Per-group heatmaps** — one per group, side by side.

### RQ1: Sentiment Agreement

**How often does whole-text sentiment agree with aspect-level sentiment?** This tab presents the project's RQ1 finding: whole-review sentiment can miss nuance that becomes visible once student comments are broken into aspect-level judgments.

- **Purpose** — to justify why aspect-based analysis is worth doing rather than relying only on a single overall sentiment label per review.
- **Displayed result** — a disagreement heatmap and summary metrics based on the project dataset used during the study.
- **Interpretation** — diagonal cells indicate agreement between whole-text and aspect sentiment; off-diagonal cells indicate additional nuance captured by aspect analysis.
- **Important note** — this tab is presented as a project finding, not as a live recalculation from whatever files happen to be uploaded during a later app session.

This RQ1 framing follows the disagreement-analysis perspective discussed by Nashihin et al. (2025), where disagreement is treated as evidence that aspect-based analysis captures additional information beyond document-level sentiment.

### RQ2: Aspect Associations

**Which course aspects are most strongly associated with negative overall evaluations?** This tab answers the question at three levels — global, comparative, and per-group — that can legitimately disagree, and the disagreements are themselves informative.

#### How each layer works

The three layers are not the same formula applied to different slices. They use **different methods entirely**:

- **Global = simple counting. No regression.** Just: *"of all the negative reviews uploaded, what % mention each aspect?"* This uses every PDF you uploaded in Stage 1, regardless of what you selected in Stage 3.
- **Comparative = OLS regression.** One row per group, predicts overall negativity from per-aspect negativity. This uses only the groups selected in Stage 3.
- **Per-group = simple ranking within each group. No regression.** Just: *"for this group, which aspect has the highest negative mention rate?"*

Global and Per-group are descriptive (counts and rankings). Only Comparative uses the regression. That is why they can disagree on the same data — they are asking structurally different questions.

#### What OLS regression actually does

> **OLS draws a straight-line relationship between how negative each aspect is and how negative the overall reviews are, across the groups you selected. The coefficients tell you which aspect's pattern matches the overall pattern most closely. The aspect with the biggest positive coefficient is the one whose ups and downs explain the ups and downs in overall negativity best.**

For each group, the regression takes the overall negative review rate as the **outcome** and the per-aspect negative mention rates as the **predictors**. The coefficients (the β values) are the weights that best line up the predictors with the outcome across rows. A large positive coefficient means that aspect's variation tracks overall negativity closely. A coefficient near zero means the aspect is either flat across groups or its variation does not line up with the outcome.

A worked example: if you compare three courses where instructor negativity is 80% in all three but workload climbs 30% → 60% → 90% as overall negativity climbs 20% → 50% → 80%, OLS will give workload a large positive coefficient (it tracks overall negativity perfectly) and instructor a coefficient near zero (it does not move, so it cannot explain why one course is worse than another). That does not mean instructor is unimportant — it means instructor is a shared baseline concern, while workload is what differentiates the groups. The Global and Per-group views would still flag instructor as the top concern in absolute terms. This is exactly the kind of disagreement the three layers are designed to surface.

#### When the layers agree vs disagree

- **All three agree** → strong, convergent signal. Prioritise that aspect for intervention.
- **Global and Per-group agree but Comparative disagrees** → the agreed aspect is a shared baseline concern across all uploaded data; the comparative aspect is the specific driver of variation among the groups you selected.
- **Comparative is near-zero across the board** → the selected groups have very similar profiles. Trust the Global and Per-group views for this selection.

#### Tab contents

- **Global bar chart** — aspects ranked by share of negative reviews they appear in, across all uploaded data.
- **Coefficient chart** — standardised OLS coefficients across selected groups. Red bars differentiate more-negative groups; green bars do not.
- **Agreement callout** — flags whether the global and comparative views pick the same top aspect, and explains any disagreement in plain language.
- **Per-group paragraphs** — one sentence per group naming its biggest concern and most positively received aspect.
- **Key Finding** — synthesis block at the bottom combining all three layers into a single takeaway.

#### Methodological note

Because the course evaluation PDFs contain anonymous reviews, individual aspect mentions cannot be linked to individual overall ratings. The regression is therefore necessarily specified at the group level, with one observation per section, year, or course. This limits statistical power — the analysis is exploratory and descriptive rather than inferential. Coefficients indicate direction and relative importance, not statistical significance. This is a constraint imposed by the data structure, not a methodological choice.

#### Research grounding

This RQ2 view is motivated by prior work on document-level sentiment classification and aspect-level sentiment analysis:

- Pang, Lee, and Vaithyanathan (2002): document-level sentiment classification overview via EMNLP. Link: https://aclanthology.org/W02-1011/
- Schouten and Frasincar (2016): broad survey of aspect-level sentiment analysis methods. Link: https://pure.eur.nl/en/publications/survey-on-aspect-level-sentiment-analysis/

In practical terms, the app uses those ideas to move from overall review polarity toward more actionable aspect-level signals for teaching improvement.

### LLM Summary
Generates a plain-English faculty improvement report using Meta Llama 3 8B Instruct via the HuggingFace Inference API. The report acknowledges what students appreciate, identifies the top three areas of concern with specific recommendations, flags any aspects confirmed as problems by both essay and numeric data (Double Signal), and closes with an encouraging note. One paragraph is generated per group. Requires a valid HuggingFace token.

### Numeric Insights
Displays results from the multiple-choice Likert-scale section of the evaluation PDF (1–7 scale). Only the top response categories covering approximately 75% of responses are shown per question, so bars may not sum to 100%.

- **Mean Scores table** — colour-coded from red (low) to green (high). Scores below 5.0/7 indicate potential concern areas.
- **Response Distribution charts** — stacked bars per question, broken down by group.
- **Mean Score Trend** (Compare Years only) — line chart showing how each question's mean score has changed over the selected years.

### Download

**CSV exports:** Full Analysis, Aspect Analysis, Long Format, Numeric Ratings.

**Comprehensive PDF Report:** generates a structured PDF with all chart images, explanatory text, sentiment and aspect analysis, emotion analysis, numeric survey results, RQ2 findings including group-specific paragraphs, and the LLM improvement report if generated. The LLM section is only included if Generate LLM Summary was clicked before downloading.

---

## Setup — Local Development

```bash
git clone https://github.com/zarafatima-y/SentimentBasedCourseEvaluation.git
cd SentimentBasedCourseEvaluation
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your HuggingFace token:
```
HUGGINGFACE_TOKEN=hf_your_actual_token_here
```
You need a HuggingFace account and must accept the licence terms for Meta-Llama-3-8B-Instruct at [huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) for the LLM tab to work. The sentiment, emotion, and aspect tabs do not require a token.

Run the app:
```bash
streamlit run app.py
```

---

## Streamlit Cloud Deployment

1. Push the repository to GitHub — do not commit your `.env` file.
2. Connect your GitHub repository at [share.streamlit.io](https://share.streamlit.io).
3. Add your HuggingFace token under Settings → Secrets:
   ```toml
   HUGGINGFACE_TOKEN = "hf_your_actual_token_here"
   ```
4. Deploy.

---

## Technologies

| Category | Library | Purpose |
|---|---|---|
| Web Framework | Streamlit 1.28+ | UI, session state, widgets |
| Data | pandas, numpy, scipy | DataFrames, statistics |
| ML | scikit-learn 1.3+ | OLS regression (RQ2) |
| PDF | pdfplumber, PyPDF2 | Text extraction |
| NLP | Transformers, PyTorch, NLTK | Model loading, tokenisation |
| Sentiment Model | cardiffnlp/twitter-roberta-base-sentiment-latest | Whole-text and aspect-level sentiment |
| Aspect Pipeline | Custom-built (keyword detection + sentence splitting) | No pre-trained ABSA model |
| Emotion Model | j-hartmann/emotion-english-distilroberta-base | 7-class emotion detection |
| LLM | meta-llama/Meta-Llama-3-8B-Instruct (HF API) | Faculty report generation |
| Visualisation | Plotly, Matplotlib, Seaborn | Interactive and static charts |
| PDF Export | ReportLab, kaleido | Comprehensive report generation |

---

## Notes

- Sentiment and emotion models are downloaded from HuggingFace on first run and cached locally.
- The LLM summary runs via the HuggingFace cloud API and requires a valid token with access to the gated Llama 3 model.
- The app expects the original PDF structure to contain both essay and numeric sections in the same report.
- RQ2's comparative (regression) layer requires at least 2 groups. The global layer works with any amount of data. With only 2 groups R² = 1.0 by construction.
- If your PDFs use a different format than expected, parsing logic can be adjusted in `data/loader.py`.

---

## References

- AI@Meta (2024). Llama 3 Model Card. https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md
- Pang, B., Lee, L., & Vaithyanathan, S. (2002). *Thumbs up? Sentiment Classification using Machine Learning Techniques*. EMNLP 2002. https://aclanthology.org/W02-1011/
- Schouten, K., & Frasincar, F. (2016). *Survey on Aspect-Level Sentiment Analysis*. IEEE Transactions on Knowledge and Data Engineering, 28(3), 813–830. https://pure.eur.nl/en/publications/survey-on-aspect-level-sentiment-analysis/
- Nashihin, D., Lisnani, L., & Hanafi, I. (2025). *Disagreement Analysis of Sentiment Predictions on Student Satisfaction Surveys Using Two IndoBERT Models*. Brilliance: Research of Artificial Intelligence, 5(2), 965–972. https://jurnal.itscience.org/index.php/brilliance/article/view/7093

---

*EECS 4080 Computer Science Project — York University, Lassonde School of Engineering*
