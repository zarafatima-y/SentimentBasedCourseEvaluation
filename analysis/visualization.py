import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Tuple, Dict

class Visualizer:
    """Handle all visualizations"""

    _PALETTE = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    ]

    @staticmethod
    def create_radar_chart(
        categories: List[str],
        groups: Dict[str, List[float]],
        title: str,
        y_max: Optional[float] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        if not categories or not groups:
            print(f"Cannot create radar chart for '{title}': Missing data")
            return None, None

        palette = Visualizer._PALETTE
        N = len(categories)
        if N < 2:
            print(f"Cannot create radar chart for '{title}': Need at least 2 categories")
            return None, None

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]   # close loop

        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

        all_vals = []
        for i, (grp_name, values) in enumerate(groups.items()):
            if len(values) != N:
                print(f"Warning: value count mismatch for group '{grp_name}' — skipping")
                continue
            closed = list(values) + [values[0]]
            colour = palette[i % len(palette)]
            ax.plot(angles, closed, 'o-', linewidth=2, label=grp_name, color=colour)
            ax.fill(angles, closed, alpha=0.10, color=colour)
            all_vals.extend(values)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=9)

        arr = np.array(all_vals)
        ceiling = y_max if y_max is not None else (np.max(arr) * 1.15 if arr.size > 0 and np.max(arr) > 0 else 1)
        ax.set_ylim(0, ceiling)

        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
        plt.title(title, size=13, pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=9)
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def radar_from_aspect_df(
        aspect_df: pd.DataFrame,
        group_col: str,
        sentiment_col: str,
        groups: List,
        mode: str = 'counts',      
        top_n: int = 10,
        title: str = '',
    ) -> Tuple[plt.Figure, plt.Axes]:
        
        top_aspects = (
            aspect_df['aspect']
            .value_counts()
            .head(top_n)
            .index.tolist()
        )
        if len(top_aspects) < 2:
            return None, None

        groups_dict: Dict[str, List[float]] = {}
        for grp in groups:
            sub = aspect_df[aspect_df[group_col] == grp]
            if mode == 'counts':
                vc = sub['aspect'].value_counts()
                values = [float(vc.get(a, 0)) for a in top_aspects]
            elif mode == 'neg_pct':
                values = []
                for asp in top_aspects:
                    asp_sub = sub[sub['aspect'] == asp]
                    total = len(asp_sub)
                    neg   = (asp_sub[sentiment_col] == 'Negative').sum() if total > 0 else 0
                    values.append(round(neg / total * 100, 1) if total > 0 else 0.0)
            else:   # pos_pct
                values = []
                for asp in top_aspects:
                    asp_sub = sub[sub['aspect'] == asp]
                    total = len(asp_sub)
                    pos   = (asp_sub[sentiment_col] == 'Positive').sum() if total > 0 else 0
                    values.append(round(pos / total * 100, 1) if total > 0 else 0.0)
            groups_dict[str(grp)] = values

        y_max = 100.0 if mode in ('pos_pct', 'neg_pct') else None
        return Visualizer.create_radar_chart(top_aspects, groups_dict, title, y_max=y_max)

    @staticmethod
    def plot_section_comparison(analysis_long_df: pd.DataFrame, course_code: str, year: int):
        """Create comprehensive section comparison visualization"""
        data = analysis_long_df[
            (analysis_long_df['course_code'] == course_code) &
            (analysis_long_df['academic_year'] == year)
        ]

        if len(data) == 0:
            print(f"No data found for {course_code} in {year}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{course_code} ({year}) - Section Comparison', fontsize=16)

        sentiment_by_section = pd.crosstab(data['section'], data['Sentiment_Label'])
        sentiment_by_section.plot(kind='bar', ax=axes[0, 0],
                                  title='Sentiment Distribution by Section',
                                  color=['#ff9999', '#66b3ff', '#99ff99'])
        axes[0, 0].set_xlabel('Section')
        axes[0, 0].set_ylabel('Count')

        top_aspects = data['aspect'].value_counts().head(8)
        top_aspects.plot(kind='barh', ax=axes[0, 1],
                         title='Top 8 Aspects Mentioned (All Sections)',
                         color='skyblue')
        axes[0, 1].set_xlabel('Count')

        aspect_by_section = pd.crosstab(data['section'], data['aspect'])
        if len(aspect_by_section.columns) > 0:
            top_aspect_names = data['aspect'].value_counts().head(8).index.tolist()
            aspect_by_section_top = aspect_by_section[top_aspect_names]
            sns.heatmap(aspect_by_section_top, ax=axes[1, 0],
                        annot=True, fmt='g', cmap='YlOrRd')
            axes[1, 0].set_title('Top Aspect Mentions by Section')
            axes[1, 0].set_xlabel('Aspect')
            axes[1, 0].set_ylabel('Section')

        emotion_by_section = pd.crosstab(data['section'], data['dominant_emotion'])
        emotion_by_section.plot(kind='bar', ax=axes[1, 1],
                                title='Emotion Distribution by Section',
                                stacked=True, colormap='viridis')
        axes[1, 1].set_xlabel('Section')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()
        return fig

    @staticmethod
    def plot_disagreement_heatmap(confusion_matrix: pd.DataFrame, disagreement_rate: float):
        """Plot disagreement heatmap following Nashihin et al. 2025"""
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_matrix,
                    annot=True, fmt='.1f', cmap='RdYlGn_r',
                    center=50, vmin=0, vmax=100, square=True,
                    cbar_kws={'label': 'Percentage (%)'})
        plt.title(
            'Disagreement Heatmap: Whole-Text vs. Aspect Sentiment\n'
            '(Darker green = agreement, Darker red = disagreement)',
            fontsize=12, pad=20
        )
        plt.xlabel('Aspect Sentiment', fontsize=10)
        plt.ylabel('Whole-Text Sentiment', fontsize=10)
        plt.figtext(
            0.5, -0.15,
            f"Following Nashihin et al. (2025): Disagreement Rate = {disagreement_rate:.1%}\n"
            "Diagonal cells (Agreement) show where aspect matches whole-text.\n"
            "Off-diagonal cells (Disagreement) reveal the nuance captured.",
            ha='center', fontsize=9,
            bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=0.5')
        )
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        plt.show()

    @staticmethod
    def create_interactive_comparison(analysis_long_df: pd.DataFrame,
                                      course_code: str, year1: int, year2: int):
        """Create interactive Plotly comparison"""
        data1 = analysis_long_df[
            (analysis_long_df['course_code'] == course_code) &
            (analysis_long_df['academic_year'] == year1)
        ]
        data2 = analysis_long_df[
            (analysis_long_df['course_code'] == course_code) &
            (analysis_long_df['academic_year'] == year2)
        ]

        aspect_freq1 = data1['aspect'].value_counts().head(10)
        aspect_freq2 = data2['aspect'].value_counts().head(10)

        df_plot = pd.DataFrame({
            'Aspect': aspect_freq1.index,
            f'{year1}': aspect_freq1.values,
            f'{year2}': [aspect_freq2.get(aspect, 0) for aspect in aspect_freq1.index]
        })

        fig = go.Figure(data=[
            go.Bar(name=str(year1), x=df_plot['Aspect'], y=df_plot[str(year1)]),
            go.Bar(name=str(year2), x=df_plot['Aspect'], y=df_plot[str(year2)])
        ])
        fig.update_layout(
            title=f'{course_code}: Aspect Mentions Comparison ({year1} vs {year2})',
            xaxis_title='Aspect',
            yaxis_title='Number of Mentions',
            barmode='group',
            height=500
        )
        return fig