import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from config.settings import SENT_MAP
from utils.constants import EMOTION_TO_SENTIMENT, NEGATIVE_EMOTIONS

class CorrelationAnalyzer:
    """Handle RQ1: Correlation between whole-text and aspect sentiment"""
    
    @staticmethod
    def create_long_format(llm_ready_df: pd.DataFrame) -> pd.DataFrame:
        """Create long format for correlation analysis"""
        long_df = (
            llm_ready_df
            .explode("aspects_found")
            .rename(columns={"aspects_found": "aspect"})
            .reset_index(drop=True)
        )
        
        long_df["aspect_sentiment"] = long_df.apply(
            lambda r: (r["aspect_data"] or {}).get(r["aspect"], {}).get("sentiment"), axis=1
        )
        
        # Use whole_sentiment if available, otherwise try other names
        if 'whole_sentiment' in long_df.columns:
            long_df["whole_sent_num"] = long_df["whole_sentiment"].str.lower().map(SENT_MAP)
        elif 'Sentiment_Label' in long_df.columns:
            long_df["whole_sent_num"] = long_df["Sentiment_Label"].str.lower().map(SENT_MAP)
        elif 'sentiment' in long_df.columns:
            long_df["whole_sent_num"] = long_df["sentiment"].str.lower().map(SENT_MAP)
        
        long_df["aspect_sent_num"] = long_df["aspect_sentiment"].str.lower().map(SENT_MAP)
        
        return long_df
    
    @staticmethod
    def calculate_disagreement_rate(long_df: pd.DataFrame) -> float:
        """Calculate Overall Disagreement Rate (ODR) following Nashihin et al. 2025"""
        
        # Determine which sentiment column to use
        if 'whole_sentiment' in long_df.columns:
            sentiment_col = 'whole_sentiment'
        elif 'Sentiment_Label' in long_df.columns:
            sentiment_col = 'Sentiment_Label'
        elif 'sentiment' in long_df.columns:
            sentiment_col = 'sentiment'
        else:
            print(f"Available columns in calculate_disagreement_rate: {long_df.columns.tolist()}")
            raise KeyError("No sentiment column found in dataframe")
        
        def has_disagreement(group):
            """Check if any aspect sentiment differs from whole-text sentiment"""
            # Get the whole-text sentiment (should be the same for all rows in group)
            whole_sentiment = group[sentiment_col].iloc[0]
            
            # Check if any aspect sentiment is different
            return any(group['aspect_sentiment'] != whole_sentiment)
        
        # Create a review identifier if it doesn't exist
        if 'review_id' not in long_df.columns:
            long_df = long_df.reset_index().rename(columns={'index': 'review_id'})
        
        review_disagreement = long_df.groupby('review_id').apply(has_disagreement).reset_index()
        review_disagreement.columns = ['review_id', 'has_disagreement']
        
        total_reviews = len(review_disagreement)
        reviews_with_disagreement = review_disagreement['has_disagreement'].sum()
        disagreement_rate = reviews_with_disagreement / total_reviews if total_reviews > 0 else 0
        
        return disagreement_rate
    
    @staticmethod
    def create_confusion_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
        """Create confusion matrix of whole vs aspect sentiment"""
        
        # Determine which sentiment column to use
        if 'whole_sentiment' in long_df.columns:
            whole_col = 'whole_sentiment'
        elif 'Sentiment_Label' in long_df.columns:
            whole_col = 'Sentiment_Label'
        elif 'sentiment' in long_df.columns:
            whole_col = 'sentiment'
        else:
            print(f"Available columns in create_confusion_matrix: {long_df.columns.tolist()}")
            raise KeyError("No sentiment column found in dataframe")
        
        confusion_matrix = pd.crosstab(
            long_df[whole_col],
            long_df['aspect_sentiment'],
            normalize='index'
        ) * 100
        
        return confusion_matrix
    
    @staticmethod
    def calculate_correlations(long_df: pd.DataFrame) -> dict:
        """Calculate various correlation metrics"""
        
        # Determine which sentiment column to use
        if 'whole_sentiment' in long_df.columns:
            whole_col = 'whole_sentiment'
        elif 'Sentiment_Label' in long_df.columns:
            whole_col = 'Sentiment_Label'
        elif 'sentiment' in long_df.columns:
            whole_col = 'sentiment'
        else:
            print(f"Available columns in calculate_correlations: {long_df.columns.tolist()}")
            raise KeyError("No sentiment column found in dataframe")
        
        # Create numeric columns if they don't exist
        if 'whole_sent_num' not in long_df.columns:
            long_df['whole_sent_num'] = long_df[whole_col].str.lower().map(SENT_MAP)
        
        if 'aspect_sent_num' not in long_df.columns:
            long_df['aspect_sent_num'] = long_df['aspect_sentiment'].str.lower().map(SENT_MAP)
        
        # Overall correlation
        corr_overall = long_df[["whole_sent_num", "aspect_sent_num"]].corr().iloc[0, 1]
        
        # Correlation by aspect
        corr_by_aspect = (
            long_df.dropna(subset=["whole_sent_num", "aspect_sent_num"])
            .groupby("aspect")
            .apply(lambda g: g["whole_sent_num"].corr(g["aspect_sent_num"]))
            .sort_values(ascending=False)
        )
        
        return {
            'overall': corr_overall,
            'by_aspect': corr_by_aspect.to_dict()
        }
    
    @staticmethod
    def analyze_emotion_sentiment_correlation(df: pd.DataFrame) -> dict:
        """Analyze correlation between emotion and sentiment"""
        
        # Determine which sentiment column to use
        if 'whole_sentiment' in df.columns:
            sentiment_col = 'whole_sentiment'
        elif 'Sentiment_Label' in df.columns:
            sentiment_col = 'Sentiment_Label'
        elif 'sentiment' in df.columns:
            sentiment_col = 'sentiment'
        else:
            print(f"Available columns in analyze_emotion_sentiment_correlation: {df.columns.tolist()}")
            raise KeyError("No sentiment column found in dataframe")
        
        corr_df = df[['review', sentiment_col, 'dominant_emotion']].copy()
        corr_df = corr_df.dropna()
        
        corr_df[sentiment_col] = corr_df[sentiment_col].str.lower().str.strip()
        corr_df['dominant_emotion'] = corr_df['dominant_emotion'].str.lower().str.strip()
        
        # Map emotions to sentiment
        corr_df['emotion_sentiment'] = corr_df['dominant_emotion'].map(EMOTION_TO_SENTIMENT)
        corr_df = corr_df.dropna()
        
        # Numeric encoding
        sent_to_num = {'negative': -1, 'neutral': 0, 'positive': 1}
        corr_df['whole_num'] = corr_df[sentiment_col].map(sent_to_num)
        corr_df['emotion_num'] = corr_df['emotion_sentiment'].map(sent_to_num)
        corr_df = corr_df.dropna()
        
        # Pearson and Spearman correlations
        pearson_r = corr_df['whole_num'].corr(corr_df['emotion_num'], method='pearson')
        spearman_rho = corr_df['whole_num'].corr(corr_df['emotion_num'], method='spearman')
        
        # Chi-square test
        ct = pd.crosstab(corr_df[sentiment_col], corr_df['emotion_sentiment'])
        chi2, p_value, dof, expected = chi2_contingency(ct.values)
        
        # Cramer's V
        n = ct.values.sum()
        r, k = ct.shape
        phi2 = chi2 / n
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        cramers_v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
        
        return {
            'pearson': pearson_r,
            'spearman': spearman_rho,
            'chi_square_p': p_value,
            'cramers_v': cramers_v
        }