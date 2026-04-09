import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import os

def ensure_dir(directory: str):
    """Ensure directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def clean_review_for_merge(df: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned review column for merging"""
    df['review_clean'] = df['review'].str.strip().str.lower()
    return df

def save_dataframes(df_dict: Dict[str, pd.DataFrame], output_dir: str = "data/output"):
    """Save multiple dataframes to CSV"""
    ensure_dir(output_dir)
    
    for name, df in df_dict.items():
        filename = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(filename, index=False)
        print(f"Saved {name} to {filename}")

def load_dataframes(file_dict: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Load multiple dataframes from CSV"""
    dfs = {}
    for name, filename in file_dict.items():
        if os.path.exists(filename):
            dfs[name] = pd.read_csv(filename)
            print(f"Loaded {name} from {filename}")
    return dfs