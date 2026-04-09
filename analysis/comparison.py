import pandas as pd
from typing import Dict, Any, Optional, List

class ComparisonAnalyzer:
    """Handle year and section comparisons"""
    
    @staticmethod
    def get_course_summary(llm_ready_df: pd.DataFrame, analysis_long_df: pd.DataFrame, 
                           course_code: str, year: Optional[int] = None, 
                           section: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive summary for a specific course/section/year"""
        # Filter data
        filtered = llm_ready_df.copy()
        filter_desc = []

        if course_code:
            filtered = filtered[filtered['course_code'] == course_code]
            filter_desc.append(f"Course: {course_code}")
        if year:
            filtered = filtered[filtered['academic_year'] == year]
            filter_desc.append(f"Year: {year}")
        if section:
            filtered = filtered[filtered['section'] == section]
            filter_desc.append(f"Section: {section}")

        filter_str = ", ".join(filter_desc)

        # Basic stats
        n_reviews = len(filtered)
        sentiment_dist = filtered['Sentiment_Label'].value_counts().to_dict()
        emotion_dist = filtered['dominant_emotion'].value_counts().head(3).to_dict()

        # Aspect stats from long format
        filtered_long = analysis_long_df[analysis_long_df['review_clean'].isin(filtered['review_clean'])]

        top_aspects = {}
        if len(filtered_long) > 0:
            for aspect, group in filtered_long.groupby('aspect'):
                top_aspects[aspect] = {
                    'count': len(group),
                    'sentiment': group['aspect_sentiment'].value_counts().to_dict()
                }
            
            # Sort by count and take top 7
            top_aspects = dict(sorted(top_aspects.items(), 
                                     key=lambda x: x[1]['count'], 
                                     reverse=True)[:7])

        # Negative aspects and emotions
        negative_aspects = []
        if 'negative' in pd.concat([filtered_long['aspect_sentiment']]).values:
            negative_aspects = filtered_long[filtered_long['aspect_sentiment'] == 'negative']['aspect'].unique().tolist()
        
        negative_emotions = filtered[filtered['dominant_emotion'].isin(['anger', 'sadness', 'fear', 'disgust'])]['dominant_emotion'].value_counts().to_dict()

        return {
            'description': filter_str,
            'n_reviews': n_reviews,
            'sentiment_dist': sentiment_dist,
            'emotion_dist': emotion_dist,
            'top_aspects': top_aspects,
            'negative_aspects': negative_aspects[:5] if negative_aspects else [],
            'negative_emotions': negative_emotions
        }
    
    @staticmethod
    def compare_courses(llm_ready_df: pd.DataFrame, analysis_long_df: pd.DataFrame,
                        course_code: str, year1: int, year2: int) -> Dict[str, Any]:
        """Compare same course across different years"""
        data_year1 = ComparisonAnalyzer.get_course_summary(
            llm_ready_df, analysis_long_df, course_code, year1
        )
        data_year2 = ComparisonAnalyzer.get_course_summary(
            llm_ready_df, analysis_long_df, course_code, year2
        )
        
        # Find changes in aspects
        aspects1 = set(data_year1['top_aspects'].keys())
        aspects2 = set(data_year2['top_aspects'].keys())
        
        return {
            'type': f"{course_code} - {year1} vs {year2}",
            'year1': data_year1,
            'year2': data_year2,
            'new_aspects': list(aspects2 - aspects1),
            'disappeared_aspects': list(aspects1 - aspects2)
        }
    
    @staticmethod
    def compare_sections(llm_ready_df: pd.DataFrame, analysis_long_df: pd.DataFrame,
                         course_code: str, year: int, sections: List[str]) -> Dict[str, Any]:
        """Compare multiple sections within the same year"""
        section_data = {}
        
        for section in sections:
            section_data[section] = ComparisonAnalyzer.get_course_summary(
                llm_ready_df, analysis_long_df, course_code, year, section
            )
        
        return {
            'type': f"{course_code} ({year}) - Section Comparison",
            'sections': section_data
        }