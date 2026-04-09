================================================================================
  Course Evaluation Analysis System
  EECS 4080 Final Project
================================================================================

A Streamlit web application that analyses student course evaluation PDFs using
NLP, sentiment analysis, emotion detection, aspect-level analysis, and LLM-
generated faculty improvement reports.

--------------------------------------------------------------------------------
  WHAT IT DOES
--------------------------------------------------------------------------------

Upload one or more course evaluation PDFs and the application will:

  - Extract and clean student written reviews automatically
  - Run whole-text sentiment analysis (Positive / Neutral / Negative)
  - Detect the dominant emotion per review (joy, sadness, anger, fear, etc.)
  - Break down sentiment by course aspect (instructor quality, workload,
    assessments, course content, difficulty, resources, and more)
  - Compare results across sections, years, or different courses
  - Identify which aspects most predict negative overall evaluations (RQ2)
  - Generate a plain-English faculty improvement report using an LLM

Results are displayed in an interactive multi-tab dashboard and can be
downloaded as CSV files.

--------------------------------------------------------------------------------
  ANALYSIS MODES
--------------------------------------------------------------------------------

  Single Course Analysis
    Analyses one course in one year. Shows overall sentiment distribution
    and a breakdown by section.

  Compare Sections (Same Course, Same Year)
    Side-by-side comparison of multiple sections of the same course.
    Useful for identifying instructor or section-level differences.

  Compare Years (Same Course)
    Tracks how student sentiment and aspect feedback have changed over
    multiple years for the same course.

  Cross-Course Comparison
    Compares multiple different courses against each other to identify
    patterns across a department or programme.

--------------------------------------------------------------------------------
  TABS IN THE RESULTS DASHBOARD
--------------------------------------------------------------------------------

  Overview        — Total reviews, courses, years, sections loaded
  Sentiment       — Pie charts, bar charts, trend lines, aspect sentiment balance
  Aspects         — Heatmaps of aspect frequency and sentiment by group
  Emotions        — Emotion distribution across courses, years, or sections
  RQ2 Analysis    — OLS regression identifying which aspects predict negative
                    evaluations; raw complaint rate chart
  LLM Summary     — AI-generated faculty improvement report grounded in the data
  Numeric Results — Multiple-choice question distributions from the PDF
  Download        — Export all analysis results as CSV

--------------------------------------------------------------------------------
  TECHNOLOGIES USED
--------------------------------------------------------------------------------

  Web Framework
    Streamlit 1.28+           — UI, session state, interactive widgets

  Data Processing
    pandas 2.0+               — DataFrames and all tabular operations
    numpy 1.24+               — Numerical operations
    scipy 1.11+               — Statistical tests
    scikit-learn 1.3+         — OLS regression (RQ2), preprocessing

  PDF Extraction
    pdfplumber 0.10+          — Primary PDF text extraction
    PyPDF2 3.0+               — Fallback PDF reading

  NLP & Machine Learning
    Hugging Face Transformers — Model loading and tokenisation
    PyTorch 2.0+              — Tensor operations (used by transformers)
    NLTK 3.8+                 — Sentence tokenisation for aspect extraction
    rapidfuzz 3.0+            — Fuzzy matching for null review detection

  Sentiment Model
    cardiffnlp/twitter-roberta-base-sentiment-latest
    Fine-tuned RoBERTa for social-text sentiment (Negative / Neutral / Positive)

  Emotion Model
    j-hartmann/emotion-english-distilroberta-base
    DistilRoBERTa fine-tuned for 7-class emotion detection

  LLM (Faculty Report Generation)
    meta-llama/Meta-Llama-3-8B-Instruct via HuggingFace Inference API
    Runs on HuggingFace servers — no local GPU required

  Visualisation
    Plotly 5.14+              — All interactive charts in the dashboard
    Matplotlib / Seaborn      — Static chart utilities

  Utilities
    python-dotenv 1.0+        — .env file loading for local development
    requests 2.31+            — HuggingFace API calls
    tqdm 4.66+                — Progress bars during batch inference

--------------------------------------------------------------------------------
  SETUP — LOCAL DEVELOPMENT
--------------------------------------------------------------------------------

  1. Clone the repository

       git clone <repo-url>
       cd EECS4080_Project

  2. Create and activate a virtual environment

       python3 -m venv venv
       source venv/bin/activate          # macOS / Linux
       venv\Scripts\activate             # Windows

  3. Install dependencies

       pip install -r requirements.txt

  4. Set your HuggingFace token

       Copy .env.example to .env and fill in your token:

         cp .env.example .env

       Then open .env and replace the placeholder:

         HUGGINGFACE_TOKEN=hf_your_actual_token_here

       You need a HuggingFace account and must accept the licence terms for
       Meta-Llama-3-8B-Instruct at huggingface.co before the LLM tab will work.
       The sentiment and emotion tabs do NOT require a token.

  5. Run the application

       streamlit run app.py

       The app will open in your browser at http://localhost:8501

--------------------------------------------------------------------------------
  SETUP — STREAMLIT CLOUD DEPLOYMENT
--------------------------------------------------------------------------------

  1. Push the repository to GitHub (do not commit your .env file)

  2. Go to share.streamlit.io and connect your GitHub repository

  3. Set your HuggingFace token as a secret:
       Settings → Secrets → add the following line:

         HUGGINGFACE_TOKEN = "hf_your_actual_token_here"

  4. Deploy — the app will load the token automatically from Streamlit Secrets

--------------------------------------------------------------------------------
  HUGGINGFACE TOKEN
--------------------------------------------------------------------------------

  A HuggingFace token is only required for the LLM Summary tab.
  All other analysis (sentiment, emotion, aspects, RQ2) runs locally using
  downloaded model weights and does not need the token.

  To get a token:
    1. Create a free account at huggingface.co
    2. Go to Settings → Access Tokens → New token (read permission is enough)
    3. Visit huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct and accept
       the licence agreement while logged in

  IMPORTANT: Never commit your .env file or paste your token into code.
  The .env file is listed in .gitignore and will not be uploaded to GitHub.

--------------------------------------------------------------------------------
  PROJECT STRUCTURE
--------------------------------------------------------------------------------

  app.py                    Main Streamlit application entry point
  requirements.txt          Python dependencies
  .env.example              Token template (safe to commit)
  .env                      Your actual token — DO NOT commit this

  config/
    settings.py             Model names, token loading, file paths

  data/
    loader.py               PDF text extraction and metadata parsing
    preprocessor.py         Review cleaning and null filtering
    raw/                    Place input PDF files here
    output/                 Generated CSV exports appear here

  models/
    sentiment.py            Whole-text sentiment classifier
    aspect.py               Aspect extraction and aspect-level sentiment
    emotion.py              Emotion detection
    llmsum.py               LLM summary generation via HF Inference API

  analysis/
    comparison.py           Cross-group data aggregation
    correlation.py          Disagreement rate and correlation statistics
    visualization.py        Radar chart and polar plot utilities

  ui/
    overview_tab.py         Dataset overview tab
    sentiment_tab.py        Sentiment results tab
    aspects_tab.py          Aspect analysis tab
    emotions_tab.py         Emotion analysis tab
    rq2_tab.py              RQ2 regression analysis tab
    llm_tab.py              LLM summary tab
    numeric_tab.py          Multiple-choice results tab
    download_tab.py         CSV export tab
    helpers.py              Shared colour maps and chart utilities

  utils/
    constants.py            Aspect keyword dictionaries, emotion mappings
    helpers.py              File I/O and DataFrame utilities

--------------------------------------------------------------------------------
  PDF FORMAT
--------------------------------------------------------------------------------

  The application is designed for course evaluation PDFs that contain an
  "ESSAY RESULTS" section with free-text student responses. It automatically
  extracts the course code, academic year, and section number from the PDF
  header.

  If your PDFs use a different format you may need to adjust the parsing
  logic in data/loader.py.

--------------------------------------------------------------------------------
  NOTES
--------------------------------------------------------------------------------

  - Sentiment and emotion models are downloaded from HuggingFace on first run
    and cached locally. Initial startup may take a few minutes.

  - The LLM summary runs via the HuggingFace cloud API and is typically fast
    (under 30 seconds). It does not run locally.

  - RQ2 regression requires at least 2 groups (sections, years, or courses).
    With only 2 groups the model fits perfectly (R² = 1.0) and coefficients
    should be treated as descriptive only.

  - All analysis results can be exported from the Download tab as CSV files.

================================================================================
