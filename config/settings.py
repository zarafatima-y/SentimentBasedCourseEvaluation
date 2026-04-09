# Configuration settings and constants
import os
from pathlib import Path
from dotenv import load_dotenv

# Always load from the project root .env regardless of working directory
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Hugging Face token
# Local dev: set HUGGINGFACE_TOKEN in a .env file (never commit .env)
# Streamlit Cloud: add HUGGINGFACE_TOKEN under Settings → Secrets as:
#   HUGGINGFACE_TOKEN = "hf_..."
# The st.secrets lookup below falls back gracefully if streamlit is not running.
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
if not HF_TOKEN:
    try:
        import streamlit as st
        HF_TOKEN = st.secrets.get('HUGGINGFACE_TOKEN')
    except Exception:
        pass

# Model configurations
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# Sentiment mapping
SENT_MAP = {"negative": -1, "neutral": 0, "positive": 1}

# File paths
DATA_DIR = "data/raw"
OUTPUT_DIR = "data/output"