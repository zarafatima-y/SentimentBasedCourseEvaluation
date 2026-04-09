import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz
from typing import Optional

class DataPreprocessor:
    """Handle data cleaning and preprocessing"""
    
    def __init__(self):
        self.NULL_WORDS = {'na', 'n/a', 'none', 'nil', 'n.a'}
    
    def clean_question_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean question text by removing numbers and colons"""
        df['question_text'] = (
            df['question_text']
            .astype(str)
            .str.replace(r'^\s*\d+\s*[\)\.\-]\s*', '', regex=True)
            .str.replace(r':\s*$', '', regex=True)
            .str.strip()
        )
        return df
    
    def clean_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic review cleaning"""
        df['review'] = (
            df['review']
            .astype(str)
            .str.replace(r'\b[A-Z]\d+-\d+\b', '', regex=True)
            .str.replace(r'^\s*-\s*', '', regex=True)
            .str.strip()
        )
        return df
    
    def is_nullish(self, text: Optional[str]) -> Optional[str]:
        """Check if text is null-like (empty, na, none, etc.)"""
        if pd.isna(text):
            return np.nan

        # Remove spaces
        compact = re.sub(r'\s+', '', text.lower().strip())

        # Catch misspellings of no and none
        if len(compact) <= 3 and compact.startswith("no"):
            return np.nan

        # Normalize misspellings of "nothing"
        if fuzz.ratio(compact, "nothing") >= 80:
            return "nothing"

        # Other null-like responses
        if (fuzz.ratio(compact, "none") >= 90 or
            fuzz.ratio(compact, "na") >= 90 or
            fuzz.ratio(compact, "nil") >= 90):
            return np.nan

        return text
    
    def remove_null_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove null-like reviews"""
        # Normalize text
        df['review_norm'] = (
            df['review']
            .str.lower()
            .str.strip()
            .str.replace(r'[^\w\s]', '', regex=True)
        )

        # Remove symbols only
        df.loc[
            ~df['review_norm'].str.contains(r'[a-z0-9]', regex=True, na=False),
            'review_norm'
        ] = np.nan

        # Remove null-semantic words
        normalized = df['review_norm'].str.replace(r'[^a-z]', '', regex=True)
        df.loc[normalized.isin(self.NULL_WORDS), 'review_norm'] = np.nan

        # Remove numbers only
        df.loc[df['review_norm'].str.fullmatch(r'\d+', na=False), 'review_norm'] = np.nan

        # Remove single letters
        df.loc[df['review_norm'].str.fullmatch(r'[a-z]', na=False), 'review_norm'] = np.nan

        # Apply fuzzy null detection
        df['review_norm'] = df['review_norm'].apply(self.is_nullish)

        # Assign back to review and drop norm column
        df['review'] = df['review_norm']
        df = df.drop(columns=['review_norm'])
        
        # Drop NaN reviews
        df = df.dropna(subset=['review']).reset_index(drop=True)
        
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full preprocessing pipeline"""
        df = self.clean_question_text(df)
        df = self.clean_reviews(df)
        df = self.remove_null_reviews(df)
        return df