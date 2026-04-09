import torch
import pandas as pd
from transformers import pipeline
from typing import List, Dict, Any
from tqdm import tqdm

class EmotionAnalyzer:
    """Handle emotion analysis"""
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        self.device = 0 if torch.cuda.is_available() else -1
        
        self.pipeline = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=self.device,
            truncation=True,
            padding=True,
            max_length=512,
            return_all_scores=False
        )
    
    def analyze(self, reviews: List[str]) -> pd.DataFrame:
        """Analyze emotions for a list of reviews"""
        emotions = []
        confidences = []
        
        for review in tqdm(reviews, desc="Analyzing emotions"):
            try:
                result = self.pipeline(review)[0]
                emotions.append(result['label'])
                confidences.append(round(result['score'], 3))
            except Exception as e:
                print(f"Error processing: {review[:50]}... Error: {e}")
                emotions.append('unknown')
                confidences.append(0.0)
        
        return pd.DataFrame({
            'dominant_emotion': emotions,
            'emotion_confidence': confidences
        })