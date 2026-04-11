import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

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