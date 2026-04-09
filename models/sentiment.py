import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List, Dict, Any
from tqdm import tqdm

class SentimentAnalyzer:
    """Handle whole-text sentiment analysis"""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            return_all_scores=True,
            truncation=True,
            padding=True,
            max_length=512
        )
    
    def analyze(self, reviews: List[str], batch_size: int = 8) -> pd.DataFrame:
        """Analyze sentiment for a list of reviews"""
        results = self.pipeline(reviews, batch_size=batch_size)
        
        neg_scores, neu_scores, pos_scores, final_labels = [], [], [], []
        
        for res_dict in results:
            current_label = res_dict['label']
            current_score = res_dict['score']

            neg = 0.0
            neu = 0.0
            pos = 0.0

            if current_label == 'negative':
                neg = current_score
            elif current_label == 'neutral':
                neu = current_score
            elif current_label == 'positive':
                pos = current_score

            neg_scores.append(neg)
            neu_scores.append(neu)
            pos_scores.append(pos)
            final_labels.append(current_label.capitalize())
        
        return pd.DataFrame({
            "negative_score": neg_scores,
            "neutral_score": neu_scores,
            "positive_score": pos_scores,
            "sentiment": final_labels
        })