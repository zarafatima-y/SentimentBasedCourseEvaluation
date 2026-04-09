import re
import torch
import pandas as pd
from transformers import pipeline
from typing import List, Dict, Any
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from utils.constants import ASPECT_KEYWORDS

class AspectAnalyzer:
    """Handle aspect extraction and aspect-based sentiment analysis"""
    
    def __init__(self, sentiment_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.device = 0 if torch.cuda.is_available() else -1
        
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=sentiment_model_name,
            tokenizer=sentiment_model_name,
            device=self.device,
            truncation=True,
            padding=True,
            max_length=512,
            return_all_scores=False
        )
    
    def extract_aspects(self, text: Any) -> List[str]:
        """Extract aspects from text based on keyword matching"""
        # Convert to string and handle non-string inputs
        if pd.isna(text) or text is None:
            return []
        
        # Convert to string and lowercase
        text_str = str(text).strip()
        if not text_str or text_str == 'nan' or text_str == 'None':
            return []
        
        text_lower = text_str.lower()
        aspects_found = []

        for aspect, keywords in ASPECT_KEYWORDS.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                    aspects_found.append(aspect)
                    break

        return aspects_found
    
    def analyze_aspect(self, text: Any, aspect: str) -> Dict[str, Any]:
        """Analyze sentiment for a specific aspect"""
        # Handle non-string inputs
        if pd.isna(text) or text is None:
            text = ""
        text_str = str(text)
        
        # Split into sentences
        try:
            sentences = sent_tokenize(text_str)
        except:
            sentences = [text_str]
        
        # Find sentences relevant to this aspect
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in ASPECT_KEYWORDS[aspect]):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            aspect_text = " ".join(relevant_sentences)
            method = "sentence_level"
        else:
            aspect_text = f"When discussing {aspect}: {text_str}"
            method = "prompt_fallback"
        
        # Truncate if too long
        if len(aspect_text) > 2000:
            aspect_text = aspect_text[:2000]
        
        try:
            result = self.sentiment_pipeline(aspect_text)[0]
            sentiment = result['label'].capitalize()
            confidence = round(result['score'], 3)
        except Exception as e:
            print(f"Error analyzing aspect '{aspect}': {e}")
            sentiment = "Unknown"
            confidence = 0.0
        
        return {
            'aspect': aspect,
            'sentiment': sentiment,
            'confidence': confidence,
            'method': method
        }
    
    def analyze_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze all reviews for aspects"""
        all_aspect_rows = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing aspects"):
            text = row['review']
            
            # Handle non-string reviews
            if pd.isna(text) or text is None:
                continue
                
            # Convert to string and clean
            text_str = str(text).strip()
            if not text_str or text_str == 'nan' or text_str == 'None':
                continue
            
            aspects = self.extract_aspects(text_str)
            
            for aspect in aspects:
                result = self.analyze_aspect(text_str, aspect)
                # Add metadata from the original row
                result['review'] = text_str
                result['course_code'] = row.get('course_code', 'Unknown')
                result['academic_year'] = row.get('academic_year', 'Unknown')
                result['section'] = row.get('section', 'Unknown')
                result['review_clean'] = text_str.lower()  # Add clean version for merging
                all_aspect_rows.append(result)
        
        if all_aspect_rows:
            return pd.DataFrame(all_aspect_rows)
        else:
            # Return empty dataframe with expected columns
            return pd.DataFrame(columns=['review', 'aspect', 'sentiment', 'confidence', 
                                        'method', 'course_code', 'academic_year', 
                                        'section', 'review_clean'])