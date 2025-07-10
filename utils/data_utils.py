import os
import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger

def download_if_missing():
    if not os.path.exists('data/banglasenti.csv'):
        logger.warning("data/banglasenti.csv missing. Run python data/prepare.py")

def load_and_clean_csv(path, text_col='text', label_col='label'):
    df = pd.read_csv(path)
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df.dropna(subset=[text_col, label_col])
    return df

def stratified_split(df, label_col='label', test_size=0.2, seed=42):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df[label_col], random_state=seed)
    return train_df, test_df
