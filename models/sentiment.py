import nltk
import pandas as pd
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from typing import List


class SentimentAnalyzer:
    """Handle whole-text sentiment analysis."""

    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        self.mode = "vader"
        self.vader = None
        self.pipeline = None

        try:
            nltk.download("vader_lexicon", quiet=True)
            self.vader = SentimentIntensityAnalyzer()
        except Exception:
            self.mode = "hf"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                truncation=True,
                padding=True,
                max_length=512,
                top_k=None,
            )

    @staticmethod
    def _label_from_compound(compound: float) -> str:
        if compound >= 0.05:
            return "Positive"
        if compound <= -0.05:
            return "Negative"
        return "Neutral"

    def _analyze_with_vader(self, reviews: List[str]) -> pd.DataFrame:
        rows = []
        for review in reviews:
            scores = self.vader.polarity_scores(review)
            neg = float(scores.get("neg", 0.0))
            neu = float(scores.get("neu", 0.0))
            pos = float(scores.get("pos", 0.0))
            compound = float(scores.get("compound", 0.0))
            label = self._label_from_compound(compound)

            rows.append(
                {
                    "Positive": pos,
                    "Negative": neg,
                    "Neutral": neu,
                    "Compound": compound,
                    "positive_score": pos,
                    "negative_score": neg,
                    "neutral_score": neu,
                    "Sentiment_Label": label,
                    "sentiment": label,
                }
            )

        return pd.DataFrame(rows)

    def _analyze_with_hf(self, reviews: List[str], batch_size: int) -> pd.DataFrame:
        results = self.pipeline(reviews, batch_size=batch_size)

        rows = []
        for score_list in results:
            score_map = {
                item["label"].lower(): float(item["score"])
                for item in score_list
            }
            neg = score_map.get("negative", 0.0)
            neu = score_map.get("neutral", 0.0)
            pos = score_map.get("positive", 0.0)
            compound = pos - neg
            label = max(
                [("Negative", neg), ("Neutral", neu), ("Positive", pos)],
                key=lambda x: x[1],
            )[0]

            rows.append(
                {
                    "Positive": pos,
                    "Negative": neg,
                    "Neutral": neu,
                    "Compound": compound,
                    "positive_score": pos,
                    "negative_score": neg,
                    "neutral_score": neu,
                    "Sentiment_Label": label,
                    "sentiment": label,
                }
            )

        return pd.DataFrame(rows)

    def analyze(self, reviews: List[str], batch_size: int = 8) -> pd.DataFrame:
        """Analyze sentiment for a list of reviews."""
        clean_reviews = ["" if review is None else str(review) for review in reviews]

        if self.mode == "vader" and self.vader is not None:
            return self._analyze_with_vader(clean_reviews)

        return self._analyze_with_hf(clean_reviews, batch_size=batch_size)
