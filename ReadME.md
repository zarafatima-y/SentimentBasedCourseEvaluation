# Sentiment Based Course Evaluation Analysis System

A Streamlit web application that analyses student course evaluation PDFs using NLP, sentiment analysis, emotion detection, aspect-level analysis, and LLM-generated faculty improvement reports.

**Live App:** [sentimentbasedcourseevaluation.streamlit.app](https://sentimentbasedcourseevaluation.streamlit.app/)

---

## What It Does

Upload one or more course evaluation PDFs and the application will:

- Extract and clean student written reviews automatically
- Run whole-text sentiment analysis (Positive / Neutral / Negative)
- Detect the dominant emotion per review (joy, sadness, anger, fear, surprise, neutral)
- Break down sentiment by course aspect (instructor quality, workload, assessments, course content, difficulty, resources, and more) using a custom-built ABSA pipeline — no pre-trained aspect model is used. Reviews are split into individual sentences, a keyword dictionary maps each sentence to one of ten course aspects, and the same RoBERTa sentiment model used for whole-text analysis evaluates each sentence independently
- Compare results across sections, years, or different courses
- Identify which aspects most predict negative overall evaluations (RQ2)
- Generate a plain-English faculty improvement report using an LLM
- Export all results as CSV files or a comprehensive PDF report

---

## Accessing the App

**Web (no setup required)**
Visit [sentimentbasedcourseevaluation.streamlit.app](https://sentimentbasedcourseevaluation.streamlit.app/) in any browser. No account or installation needed.

**Local development**
See the [Local Setup](#setup--local-development) section below.

---

## How to Use the App

The app walks you through four stages shown in the left sidebar: Upload, Clean, Analyze, and Results. Each stage must be completed before moving to the next.

**Stage 1 — Upload**
Upload one or more course evaluation PDFs using the file uploader. The app expects PDFs containing an `ESSAY RESULTS` section with free-text student responses. It automatically extracts the course code, academic year, and section from the PDF header. Multiple PDFs can be uploaded at once.

**Stage 2 — Clean**
Review the extracted data and select cleaning options (remove nulls, remove very short reviews, normalize text). Click Run Preprocessing to confirm. The number of reviews remaining after cleaning is shown.

**Stage 3 — Analyze**
Select your analysis type, the courses or sections you want to include, which analysis modules to run (sentiment, aspect, emotion), and whether to show heatmaps and radar charts. Click Run Selected Analysis to proceed.

**Stage 4 — Results**
Results are displayed in a tabbed dashboard. Each tab is described in detail below.

---

## Analysis Types

**Single Course Analysis**
Analyzes one course in one year. All sections are pooled together. Best used when you want an overall picture of how a course performed in a specific year.
- Select: 1 course, 1 year
- Sections available: all sections pooled automatically

**Compare Sections (Same Course, Same Year)**
Side-by-side comparison of multiple sections of the same course in the same year. Useful for identifying differences between instructors or section delivery.
- Select: 1 course, 1 year, 2 to 5 sections
- Minimum: 2 sections | Maximum: 5 sections

**Compare Years (Same Course)**
Tracks how student sentiment and aspect feedback have changed over time for the same course. Sections within each year are pooled.
- Select: 1 course, 2 to 5 years
- Minimum: 2 years | Maximum: 5 years

**Cross-Course Comparison**
Compares multiple different courses against each other. Each course is identified by its code and year. Useful for department-level analysis.
- Select: 2 to 5 course–year combinations
- Minimum: 2 combinations | Maximum: 5 combinations

---

## Results Dashboard — Tab by Tab

### Overview
Shows the total number of reviews, courses, years, and sections loaded. Displays a review count breakdown by group (section, year, or course depending on analysis type). For Compare Years, also shows overall sentiment trend lines and per-aspect negative sentiment trends over time.

### Sentiment
Shows how student reviews break down into Positive, Neutral, and Negative categories.

- **Individual pie charts** — one per group (section, year, or course) showing that group's sentiment split independently. Compare proportions across groups to spot differences.
- **Grouped bar chart** — all groups side by side showing raw counts. Use this alongside the pies to see both absolute volume and proportion.
- **Trend line** (Compare Years only) — shows how each sentiment category has changed as a percentage over the selected years. A rising Negative line indicates worsening student experience.
- **Aspect Sentiment Balance** — for each course aspect, shows the percentage of mentions that were positive (green) and negative (red). Green and red do not always add up to 100% because neutral mentions fill the gap. Aspects are sorted from most negative to least so priority concerns appear first. One chart per group.

### Aspects
Shows which course dimensions students discussed and how they felt about them.

- **Aspect Frequency bar chart** — raw count of how many times each aspect was mentioned. Note this is not normalised — groups with more students will naturally have higher bars. Use the radar charts for a proportional comparison.
- **Aspect–Sentiment Heatmap** — each row is a group–aspect pair. Columns show how many mentions were Negative, Neutral, or Positive. Cross-reference with the frequency bar chart: a high count that lands mostly in the Negative column is a strong concern signal.
- **Radar charts** — three side-by-side charts showing Aspect Counts, Positive %, and Negative % for all groups on the same axes. Use Positive % and Negative % for fair cross-group comparison since they normalise for review volume. A large area on the Negative % radar for an aspect means students across groups consistently find it problematic.
- **Survey Question Breakdown table** — sentiment and top aspects grouped by the question students were answering. Colour-coded by dominant sentiment (green = positive, red = negative, yellow = neutral).

### Emotions
Shows the emotional tone of reviews beyond positive and negative polarity.

- **Emotion distribution chart** — how often each emotion (joy, anger, sadness, fear, surprise, neutral) appeared across reviews. Joy should dominate in well-received courses; high anger or sadness signals deeper dissatisfaction.
- **Emotion × Sentiment heatmap** — shows how emotions co-occur with sentiment labels. Joy clustering with Positive is expected. Anger or fear appearing in Neutral reviews is a subtler signal worth noting — students may be suppressing negative sentiment in their overall rating but expressing it emotionally.
- **Per-group heatmaps** — one heatmap per group shown side by side so you can compare emotional profiles directly.

### RQ2: Aspect Predictors
Answers the question: which aspects most strongly predict whether a group's overall evaluation is negative?

- **Coefficient chart** — horizontal bar chart showing standardised OLS regression coefficients. Red bars (positive coefficients) indicate aspects that predict more negative overall evaluations when students rate them negatively. Green bars (negative coefficients) suggest a dampening or protective effect — groups that struggle with this aspect still tend to leave positive overall reviews. Focus on direction and relative bar length rather than exact values, especially with small group counts.
- **Model Fit (R²)** — shown in the right panel. Above 0.5 indicates aspects are strong predictors. With only 2 groups R² is suppressed because a line through 2 points always fits perfectly — add more groups for a reliable R².
- **Model Input table** — shows the raw negative mention rate per aspect per group, with the overall negative review rate in the rightmost column. This lets you see the cause-and-effect directly: which aspect rates are high in groups that also have high overall negativity.
- **What Do Students Complain About Most chart** — average negative mention rate per aspect across all groups, independent of the model. Aspects high here but low in the coefficient chart are widespread concerns that affect everyone equally and therefore don't differentiate groups.
- **Group-Specific Findings** — one plain-English paragraph per group naming the most negatively rated aspect, the overall negative rate, and the most positively received aspect for that specific group.

### LLM Summary
Generates a plain-English faculty improvement report using Meta Llama 3 8B Instruct via the HuggingFace Inference API. Click Generate LLM Summary to run. Generation typically takes under 30 seconds. The report is addressed directly to the course instructor, acknowledges what students appreciate, identifies the top three areas of concern with specific recommendations, flags any aspects confirmed as problems by both essay and numeric data (Double Signal), and closes with an encouraging note. One paragraph is generated per group.

Note: LLM generation requires a valid HuggingFace token with access to Meta-Llama-3-8B-Instruct.

### Numeric Insights
Displays results from the multiple-choice Likert-scale section of the evaluation PDF (1–7 scale). Only the top response categories covering approximately 75% of responses are shown per question — less common responses are excluded to keep charts readable. This is why bars may not sum to 100%.

- **Mean Scores table** — colour-coded from red (low) to green (high). Scores below 5.0/7 indicate potential concern areas.
- **Response Distribution charts** — stacked bars showing the percentage of students selecting each rating per question, broken down by group.
- **Mean Score Trend** (Compare Years only) — line chart showing how each question's mean score has changed over the selected years. Rising lines indicate improvement.

### Download
Export all analysis results.

**CSV exports**
- Full Analysis — all reviews with sentiment, emotion, and metadata
- Aspect Analysis — all aspect-level sentiment labels
- Long Format — merged review and aspect data
- Numeric Ratings — raw Likert-scale data

**Comprehensive PDF Report**
Generates a structured PDF containing all chart images, explanatory text, sentiment and aspect analysis, emotion analysis, numeric survey results, RQ2 findings including group-specific paragraphs, and the LLM improvement report if it was generated during the session. Click Generate PDF Report to build it. Chart generation may take 15–30 seconds. The LLM section is only included if you clicked Generate LLM Summary in the LLM tab before downloading.

---

## Setup — Local Development

**1. Clone the repository**
```bash
git clone https://github.com/zarafatima-y/SentimentBasedCourseEvaluation.git
cd SentimentBasedCourseEvaluation
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set your HuggingFace token**

Copy `.env.example` to `.env` and fill in your token:
```bash
cp .env.example .env
```
Open `.env` and replace the placeholder:
```
HUGGINGFACE_TOKEN=hf_your_actual_token_here
```
You need a HuggingFace account and must accept the licence terms for Meta-Llama-3-8B-Instruct at [huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) before the LLM tab will work. The sentiment, emotion, and aspect tabs do not require a token.

**5. Run the application**
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`

---

## Streamlit Cloud Deployment

1. Push the repository to GitHub — do not commit your `.env` file
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub repository
3. Add your HuggingFace token as a secret under Settings → Secrets:
```toml
HUGGINGFACE_TOKEN = "hf_your_actual_token_here"
```
4. Deploy — the app loads the token automatically from Streamlit Secrets

---

## Technologies

| Category | Library | Purpose |
|---|---|---|
| Web Framework | Streamlit 1.28+ | UI, session state, widgets |
| Data | pandas, numpy, scipy | DataFrames, statistics |
| ML | scikit-learn 1.3+ | OLS regression (RQ2) |
| PDF | pdfplumber, PyPDF2 | Text extraction |
| NLP | Transformers, PyTorch, NLTK | Model loading, tokenisation |
| Sentiment Model | cardiffnlp/twitter-roberta-base-sentiment-latest | Whole-text sentiment and aspect-level sentence sentiment |
| Aspect Pipeline | Custom-built (keyword detection + sentence splitting) | No pre-trained ABSA model — pipeline designed and implemented from scratch |
| Emotion Model | j-hartmann/emotion-english-distilroberta-base | 7-class emotion detection |
| LLM | meta-llama/Meta-Llama-3-8B-Instruct (HF API) | Faculty report generation |
| Visualisation | Plotly, Matplotlib, Seaborn | Interactive and static charts |
| PDF Export | ReportLab, kaleido | Comprehensive report generation |
| Utilities | python-dotenv, requests, tqdm | Token loading, API calls |

---

## Notes

- Sentiment and emotion models are downloaded from HuggingFace on first run and cached locally. Initial startup may take a few minutes.
- The LLM summary runs via the HuggingFace cloud API. It does not run locally and requires a valid token with access to the gated Llama 3 model.
- RQ2 regression requires at least 2 groups. With only 2 groups R² = 1.0 by construction — add more groups for meaningful coefficients.
- If your PDFs use a different format than expected, parsing logic can be adjusted in `data/loader.py`.

---

## References

- AI@Meta (2024). Llama 3 Model Card. https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md
- Pang, B., Lee, L., & Vaithyanathan, S. (2002). Thumbs up? Sentiment classification using machine learning techniques. *EMNLP*.
- Schouten, K., & Frasincar, F. (2016). Survey on aspect-level sentiment analysis. *IEEE Transactions on Knowledge and Data Engineering*, 28(3), 813–830.
- Nashihin et al. (2025). Disagreement analysis framework for sentiment model evaluation.

---

*EECS 4080 Computer Science Project — York University, Lassonde School of Engineering*