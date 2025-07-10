# Data Directory

This directory contains the BanglaSenti dataset used for training and evaluation.

- The BanglaSenti CSV is automatically downloaded and split by `data/prepare.py`.

## Usage

Run `python -m data.prepare` to fetch the latest cleaned dataset and create train/val/test splits.

- **Dataset size:** 122,578 total rows (including header)
- **Class mapping:** 0 = Negative, 1 = Positive, 2 = Neutral
- **Split:** 97,863 train, 12,233 val, 12,233 test
