import os
import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger

def download_banglasenti():
    url = "https://github.com/niloydebbarma-code/banglasenti-dataset-prep/raw/main/banglasenti.csv"
    out_path = "data/banglasenti.csv"
    if not os.path.exists(out_path):
        import requests
        r = requests.get(url)
        with open(out_path, 'wb') as f:
            f.write(r.content)
        logger.info("Downloaded BanglaSenti dataset.")
    else:
        logger.info("BanglaSenti already exists.")

def split_banglasenti(seed=42):
    df = pd.read_csv("data/banglasenti.csv")
    df = df.dropna(subset=["text", "label"])
    train, temp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=seed)
    train.to_csv("data/train.csv", index=False)
    val.to_csv("data/val.csv", index=False)
    test.to_csv("data/test.csv", index=False)
    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

if __name__ == "__main__":
    download_banglasenti()
    split_banglasenti()
