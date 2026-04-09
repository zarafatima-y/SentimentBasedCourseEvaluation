import requests
import pandas as pd
from typing import Dict, Any


class LLMAnalyzer:
    """
    Handle LLM-based analysis and summary generation via HuggingFace Inference API.
    Uses the hosted API instead of loading the model locally, so it works on
    Streamlit Cloud and any machine without a GPU.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        hf_token: str = None,
        **kwargs,   # absorb any leftover kwargs from old call sites
    ):
        self.model_name = model_name
        self.hf_token   = hf_token
        # HuggingFace router — OpenAI-compatible chat completions endpoint
        self.api_url    = "https://router.huggingface.co/v1/chat/completions"
        self.headers    = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type":  "application/json",
        }

    # ── Data preparation (unchanged) ─────────────────────────────────────────

    def prepare_llm_data(
        self,
        llm_ready_df: pd.DataFrame,
        analysis_long_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare data in LLM-ready format"""
        aspect_dict = (
            analysis_long_df
            .groupby('review_clean')
            .apply(
                lambda x: {
                    row['aspect']: {
                        'sentiment':  row['aspect_sentiment'],
                        'confidence': row.get('confidence', None),
                    }
                    for _, row in x.iterrows()
                }
            )
            .to_dict()
        )

        llm_ready = llm_ready_df.copy()
        llm_ready['review_clean']  = llm_ready['review'].str.strip().str.lower()
        llm_ready['aspect_data']   = llm_ready['review_clean'].map(aspect_dict)
        llm_ready['aspect_data']   = llm_ready['aspect_data'].apply(
            lambda x: x if isinstance(x, dict) else {}
        )
        llm_ready['aspects_found'] = llm_ready['aspect_data'].apply(list)
        llm_ready['num_aspects']   = llm_ready['aspects_found'].apply(len)
        return llm_ready

    # ── Generation ───────────────────────────────────────────────────────────

    def generate_summary(self, prompt: str, max_length: int = 500) -> str:
        """
        Generate summary via HuggingFace router (OpenAI-compatible chat completions).
        The model runs on HuggingFace servers — no local GPU or RAM needed.
        """

        payload = {
            "model": f"{self.model_name}:novita",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert educational data analyst writing faculty improvement reports. "
                        "Output ONLY the report paragraphs — no preamble, no title, no sign-off, no meta-commentary. "
                        "Write in plain flowing paragraphs. "
                        "Never use bullet points, bold text, headers, hashtags, numbered lists, or sign-offs. "
                        "Base your analysis strictly on the data provided in the user message. "
                        "If data is limited or ambiguous, make your best analytical inference from what is available "
                        "rather than refusing to answer or saying you cannot determine something. "
                        "Only reference the courses, sections, or years that appear in the provided data."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "max_tokens":         max_length,
            "temperature":        0.7,
            "repetition_penalty": 1.15,
        }

        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=300,
        )

        if response.status_code != 200:
            raise Exception(
                f"HuggingFace API returned status {response.status_code}: {response.text}"
            )

        result = response.json()

        try:
            return result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            raise Exception(f"Unexpected API response format: {result}")