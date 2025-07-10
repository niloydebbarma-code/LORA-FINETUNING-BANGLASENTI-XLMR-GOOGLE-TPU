# LoRA Fine-Tuning of BanglaSenti on XLM-RoBERTa-Base Using Google TPUs

<!-- Logo Row: All logos in a flex row for responsive containment -->
<div align="center" style="display:flex;flex-wrap:wrap;justify-content:center;align-items:center;gap:16px;margin-bottom:8px;">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="120" style="margin:16px;filter: grayscale(100%) drop-shadow(0 0 8px #ccc);">
  <img src="https://pytorch.org/assets/images/pytorch-logo.png" alt="PyTorch" width="100" style="margin:16px;filter: grayscale(100%) drop-shadow(0 0 8px #ccc);">
  <span style="font-size:2.2em;margin:16px;vertical-align:middle;">🤗 PEFT</span>
  <img src="https://storage.googleapis.com/gweb-research2023-media/original_images/lockup_GoogleResearch_FullColor_Twitter.jpg" alt="Google Research" width="160" style="margin:16px;filter: grayscale(100%) drop-shadow(0 0 8px #ccc);">
  <img src="https://media.licdn.com/dms/image/v2/D5622AQEDpHkq60WtRg/feedshare-shrink_2048_1536/B56ZdV6KEGHUAs-/0/1749492999173?e=1755129600&v=beta&t=Af4O7y2CIGnIgSskmevlUsmPv2cjXhyN9sXkg0AjVI8" alt="TPU Research Cloud (unofficial)" width="120" style="margin:16px;filter: grayscale(100%) drop-shadow(0 0 8px #ccc);">
</div>

<!-- Badge Row: All badges in a horizontally scrollable container -->
<div align="center"style="overflow-x:auto; white-space:nowrap; text-align:center; margin-bottom:16px; padding-bottom:4px; border-bottom:1px solid #eee;">
  <a href="https://huggingface.co/niloydebbarma/BanglaSenti-XLM-RoBERTa-Experiment-Models"><img src="https://img.shields.io/badge/HF%20Experiment%20Model-blue?logo=huggingface&logoColor=yellow" alt="HF Experiment Model" style="margin:8px 16px;"></a>
  <a href="https://huggingface.co/niloydebbarma/banglasenti-lora-xlmr"><img src="https://img.shields.io/badge/HF%20LoRA%20Model-orange?logo=huggingface&logoColor=yellow" alt="HF LoRA Model" style="margin:8px 16px;"></a>
  <a href="https://huggingface.co/niloydebbarma/banglasenti-lora-xlmr"><img src="https://img.shields.io/badge/First%20LoRA%20for%20Bengali%20Sentiment-blueviolet?style=flat-square" alt="First LoRA for Bengali Sentiment" style="margin:8px 16px;"></a>
</div>

---

## Model Access

- **Experiment Model Repository:** [BanglaSenti-XLM-RoBERTa-Experiment-Models (Hugging Face)](https://huggingface.co/niloydebbarma/BanglaSenti-XLM-RoBERTa-Experiment-Models)
- **Published LoRA Model:** [banglasenti-lora-xlmr (Hugging Face)](https://huggingface.co/niloydebbarma/banglasenti-lora-xlmr)

## W&B Experiment Reports

- [BanglaSenti XLM-RoBERTa Training: Baseline vs LoRA (W&B Report)](https://wandb.ai/niloydebbarma-ai-youth-alliance/eval-bangla-lora-bn-tpu/reports/BanglaSenti-XLM-RoBERTa-Evaluation-Baseline-vs-LoRA--VmlldzoxMzQ5MTkxOA)
- [BanglaSenti XLM-RoBERTa Evaluation: Baseline vs LoRA (W&B Report)](https://wandb.ai/niloydebbarma-ai-youth-alliance/eval-bangla-lora-bn-tpu/reports/BanglaSenti-XLM-RoBERTa-Evaluation-Baseline-vs-LoRA--VmlldzoxMzQ5MTkxOA)

---

## 1. Introduction

This project provides the **first open-source implementation of LoRA (Low-Rank Adaptation) fine-tuning for Bengali sentiment analysis** using XLM-RoBERTa-Base and the largest available cleaned BanglaSenti dataset. The pipeline is designed for single-core Google Cloud TPUs. Both LoRA and baseline (no-LoRA) fine-tuning are supported, with all results and artifacts saved locally for transparency and reproducibility.

- **Dataset:** BanglaSenti (122,578 labeled reviews: 97,863 train, 12,233 val, 12,233 test)
- **Model:** XLM-RoBERTa-Base (12-layer multilingual transformer)
- **LoRA:** Parameter-efficient adapter training for transformer models

---

## 2. Objective

- Provide a reproducible codebase for LoRA and baseline fine-tuning on BanglaSenti using XLM-RoBERTa-Base
- Enable efficient training and evaluation on Google Cloud TPUs
- Support robust benchmarking and future research in Bengali sentiment analysis

---

## Features

- **First LoRA for Bengali Sentiment:** First open-source LoRA fine-tuning for Bengali sentiment on the largest available dataset
- **LoRA and Baseline Support:** Run both LoRA-adapted and full fine-tuning experiments
- **TPU-Ready:** Scripts are compatible with Google Cloud TPU v3-8 (single core)
- **Reproducible:** All configs, checkpoints, and results are saved locally; no external model hub calls required
- **Cleaned Dataset:** Uses the merged and cleaned BanglaSenti dataset
- **Open Source:** Apache 2.0 licensed code and data pipeline

---

## 3. Dataset & Model

This section lists the main dataset and model used in the experiments, with direct links for reference and reproducibility.

| Component   | Name & Link                                                                                                                                                                                                                                 |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset** | [Cleaned & merged BanglaSenti CSV](https://github.com/niloydebbarma-code/banglasenti-dataset-prep/blob/main/banglasenti.csv) ([repo](https://github.com/niloydebbarma-code/banglasenti-dataset-prep)) |
| **Model**   | [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) (12-layer multilingual transformer)                                                                                                                           |

---

## 4. Methodology

A step-by-step outline of the data preparation, model adaptation, training, and evaluation process.

1. **Data Preparation**
   - Run `python -m data.prepare` to prepare the dataset (train/val/test, stratified)
   - Tokenize using the XLM-RoBERTa tokenizer
2. **Adapter Setup**
   - Use Hugging Face `peft` to attach LoRA adapters to the base model
   - Sweep LoRA rank and alpha as needed
3. **TPU Training (Google Cloud)**
   - Single v3-8 TPU slice (1 chip) via PyTorch/XLA
   - Hyperparameters: 10 epochs, learning rate 5e-5, batch size 64
4. **Baseline**
   - Full fine-tuning of XLM-RoBERTa-Base on the same dataset
5. **Evaluation**
   - Compute accuracy and F1 on validation and test sets
   - All results are logged locally and linked to W&B logs for transparency.

---

## 5. Benchmark Results

**Table below compares the best LoRA model (Exp-TM-All) to the baseline. Bold indicates the best result for each metric.**

**How to interpret percentage changes:**
- For metrics where higher is better (Accuracy, F1):
  - Percentage Change = ((LoRA - Baseline) / Baseline) × 100
- For metrics where lower is better (Loss):
  - Percentage Change = ((Baseline - LoRA) / Baseline) × 100

| Experiment           | Train Acc | Train F1 | Train Loss | Val Acc  | Val F1  | Val Loss | During Acc | During F1 | Final Acc | Final F1 | W&B Train Link | W&B Eval Link |
|:---------------------|:----------|:---------|:-----------|:---------|:--------|:---------|:------------|:----------|:-----------|:---------|:---------------|:--------------|
| Baseline (No LoRA)   | 0.3378    | 0.33053  | 1.09993    | 0.44437  | 0.2051  | 0.01724  | 0.43497     | 0.34957   | 0.44429    | 0.27339  | [W&B Train](https://wandb.ai/niloydebbarma-ai-youth-alliance/bangla-lora-bn-tpu/runs/38qa7n35) | [W&B Eval](https://wandb.ai/niloydebbarma-ai-youth-alliance/eval-bangla-lora-bn-tpu/runs/bii81845) |
| **Exp-TM-All (LoRA)**    | **0.83005**   | **0.82504**  | **0.44704**    | **0.81068**  | **0.8032**  | **0.00808**  | **0.80528**     | **0.80415**   | **0.80724**    | **0.80645**  | **[W&B Train](https://wandb.ai/niloydebbarma-ai-youth-alliance/bangla-lora-bn-tpu/runs/nu1qo8cm)** | **[W&B Eval](https://wandb.ai/niloydebbarma-ai-youth-alliance/eval-bangla-lora-bn-tpu/runs/gfsmfb7f)** |
| Change (%)       | +145.7%   | +149.6%  | **–59.4%** | +82.4%   | +291.7% | **–53.1%** | +85.2%      | +129.9%   | +81.7%     | +194.9%  | —              | —              |

- LoRA adapters provide substantial improvements in accuracy and F1 over baseline full fine-tuning on the largest Bengali sentiment dataset.
- All results are fully reproducible and linked to W&B logs for transparency.

---

## 6. Full Experiment Details

See [`docs/experiment_report.md`](docs/experiment_report.md) for all experiment tables, metrics, and configuration details.

---

## 7. Repository Structure

An overview of the repository layout.

```
├── data/
│   ├── prepare.py        # dataset prep
│   ├── README.md         # data usage
│   └── banglasenti.csv   # (auto-downloaded)
├── models/
│   └── lora_adapter.py   # LoRA setup
├── train/
│   ├── train_lora.py     # LoRA training
│   └── train_xla.py      # Baseline training
├── eval/
│   ├── eval_all.py       # LoRA evaluation
│   └── eval_xla.py       # Baseline evaluation
├── configs/              # experiment configs
├── requirements.txt      # dependencies
├── README.md             # this file
└── utils/                # helper scripts
```

---

## 8. Data & Attribution

- The dataset is from [banglasenti-dataset-prep](https://github.com/niloydebbarma-code/banglasenti-dataset-prep)
- See [dataset_citations.md](docs/dataset_citations.md) for full citations and license info

---

## 9. Reproducibility & Deliverables

- **Apache 2.0 licensed GitHub repository:** All code, scripts, and configuration files are included for full transparency and reuse.
- **Google Cloud TPU compatibility:** All training and experiments are designed for single-core TPU v3-8, ensuring reproducibility on cloud hardware.
- **Experiment report:** Comprehensive markdown summary of results, metrics, and methodology for easy reference and benchmarking.

---

## Why Google Cloud TPU?

- **Reproducibility:** Consistent environment for benchmarking
- **Scalability:** Enables full dataset/model training
- **Open science:** Results can be reproduced by others with cloud access

---

## Data Preview

| text | label |
|------|-------|
| নাটক গুলির কথা দিনে দিনে খারাপের দিকে যাচ্ছে... | 0 |
| শেষের দুটি লাইন নাটকের পুরো সারমর্ম কে পরিবর্তন করে দিয়েছে... | 1 |
| টেনক্স ভাইয়া অনেক উপকৃত হলাম | 1 |
| আমার দেখা সবচেয়ে ভাল নাটক এর অন্যতম | 1 |
| এক কথায় অসাধারন | 1 |
| ... | ... |

- **Label mapping:** 0 = Negative, 1 = Positive, 2 = Neutral
- **Dataset size:** 122,578 total rows
- **Split:** 97,863 train, 12,233 val, 12,233 test
- **Training batch size:** 64
- **Epochs:** 10

---

## Optional: Advanced Logging with Loguru

For most users, standard logging is sufficient and is already integrated in the main scripts. However, if you need detailed, colorized, or file-based logging for debugging or analysis, you can enable advanced logging using the provided `utils/logging_config.py`:

1. **Import and activate in your script:**
   ```python
   from utils.logging_config import setup_logging
   setup_logging(log_level="DEBUG", log_file="logs/experiment.log")
   ```
   - This will log all messages (with rich formatting) to both the console and a rotating log file in the `logs/` directory.
   - You can customize the log level and file name as needed.
2. **Features:**
   - Colorized console output (with Rich)
   - Rotating file logs with detailed context (timestamp, module, function, line)
   - Intercepts standard Python logging and routes it through Loguru
   - Silences noisy library logs (e.g., transformers, datasets)

If you do not call `setup_logging`, only basic logging will be active. Use this feature for in-depth debugging or experiment traceability.

---

## 10. Acknowledgements

This research was supported by the [Google Research TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) program.

![TPU Research Cloud (unofficial illustration)](https://media.licdn.com/dms/image/v2/D5622AQEDpHkq60WtRg/feedshare-shrink_2048_1536/B56ZdV6KEGHUAs-/0/1749492999173?e=1755129600&v=beta&t=Af4O7y2CIGnIgSskmevlUsmPv2cjXhyN9sXkg0AjVI8)
*Image above is an unofficial illustration, not an official Google TRC logo.*

Special thanks to the TRC team at Google Research for providing free access to Google Cloud TPUs, which made this work possible.

The BanglaSenti dataset is from the open-source [banglasenti-dataset-prep](https://github.com/niloydebbarma-code/banglasenti-dataset-prep) project.

The base model [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) is provided by Facebook AI.

This project builds on the Hugging Face Transformers and PEFT libraries.

Thanks to the open-source community and all contributors to the code, data, and research.

---
For questions or contributions, create issues in the [GitHub issues](https://github.com/niloydebbarma-code/LORA-FINETUNING-BANGLASENTI-XLMR-GOOGLE-TPU/issues).
