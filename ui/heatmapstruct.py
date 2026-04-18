import matplotlib.pyplot as plt
import seaborn as sns

COLOR_MAP = {'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'}


def build_aspect_heatmap(aspect_df, group_col, sentiment_col, title):
    """
    Build a seaborn heatmap where:
      - Y-axis  : <group_col>-<aspect>   (e.g. "2022-instructor")
      - X-axis  : sentiment values       (Negative | Neutral | Positive)
      - Values  : raw counts with annotations
    Returns a matplotlib Figure.
    """
    df = aspect_df.copy()
    df['y_label'] = df[group_col].astype(str) + '-' + df['aspect'].astype(str)

    pivot = (
        df.groupby(['y_label', sentiment_col])
          .size()
          .unstack(fill_value=0)
    )

    for col in ['Negative', 'Neutral', 'Positive']:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[['Negative', 'Neutral', 'Positive']]
    pivot = pivot.sort_index()

    fig_height = max(6, len(pivot) * 0.45)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='d',
        cmap='Blues',
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel(sentiment_col, fontsize=11)
    ax.set_ylabel(f'{group_col}-aspect', fontsize=11)
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    return fig
