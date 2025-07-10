# LoRA Fine-Tuning Experiment Report

## Overview

This report documents the experiments conducted for fine-tuning XLM-RoBERTa-base model on the BanglaSenti dataset using LoRA (Low-Rank Adaptation). Various configurations of LoRA parameters were tested to determine the optimal setup for Bangla sentiment analysis.


## Baseline Fine-Tuning (No LoRA)

This table summarizes the results of fine-tuning XLM-RoBERTa-base on BanglaSenti without any LoRA adaptation. It serves as the baseline for all subsequent comparisons.

| Experiment | Model | LoRA Rank | Alpha | Target Modules | Dropout | Batch Size | Learning Rate | Epochs | Steps | Train Acc | Train F1 | Train Loss | Val Acc | Val F1 | Val Loss | LoRA Weight Mean | During Accuracy | During F1 | Final Accuracy | Final F1 | Training Log File | Evaluation Log File | Configs Checkpoints Logs |
|:-----------|:------|:----------|:------|:---------------|:--------|:-----------|:-------------|:-------|:------|:----------|:---------|:-----------|:--------|:-------|:---------|:-----------------|:-----------------|:----------|:---------------|:---------|:------------------|:--------------------|:------------------------|
| **Baseline (No LoRA)** | xlm-roberta-base | - | - | - | -| 64 | 5e-5 | 10 (0-9) | 1529 | 0.3378 | 0.33053 | 1.09993 | 0.44437 | 0.2051 | 0.01724 | - | 0.43497 | 0.34957 | 0.44429 | 0.27339 | [W&B Train](https://wandb.ai/niloydebbarma-ai-youth-alliance/bangla-lora-bn-tpu/runs/38qa7n35) | [W&B Eval](https://wandb.ai/niloydebbarma-ai-youth-alliance/eval-bangla-lora-bn-tpu/runs/bii81845) | [HF Model](https://huggingface.co/niloydebbarma/BanglaSenti-XLM-RoBERTa-Experiment-Models/tree/main/ex-xlm-roberta-base) |


## LoRA Fine-Tuning Experiments

This table presents the results of LoRA fine-tuning experiments with different parameter settings. The best-performing configuration is highlighted in bold.

| Experiment | Model | LoRA Rank | Alpha | Target Modules | Dropout | Batch Size | Learning Rate | Epochs | Steps | Train Acc | Train F1 | Train Loss | Val Acc | Val F1 | Val Loss | LoRA Weight Mean | During Accuracy | During F1 | Final Accuracy | Final F1 | Training Log File | Evaluation Log File | Configs Checkpoints Logs |
|:-----------|:------|:----------|:------|:---------------|:--------|:-----------|:-------------|:-------|:------|:----------|:---------|:-----------|:--------|:-------|:---------|:-----------------|:-----------------|:----------|:---------------|:---------|:------------------|:--------------------|:------------------------|
| Exp-R4  | xlm-roberta-base | 4 | 8 | ["query", "value"] | 0.1 | 64 | 5e-5 | 10 (0-9) | 1529 | 0.76998 | 0.76314 | 0.58531 | 0.77585 | 0.76796 | 0.00904 | 0.00044 | 0.77945 | 0.77884 | 0.77618 | 0.77588 | [W&B Train](https://wandb.ai/niloydebbarma-ai-youth-alliance/bangla-lora-bn-tpu/runs/xqwohj2z) | [W&B Eval](https://wandb.ai/niloydebbarma-ai-youth-alliance/eval-bangla-lora-bn-tpu/runs/ot5nqa5k) | [HF Model](https://huggingface.co/niloydebbarma/BanglaSenti-XLM-RoBERTa-Experiment-Models/tree/main/ex-baseline-4-8-qv) |
| Exp-R16    | xlm-roberta-base | 16 | 32 | ["query", "value"] | 0.1 | 64 | 5e-5 | 10 (0-9) | 1529 | 0.78914 | 0.78218 | 0.55301 | 0.79425 | 0.78559 | 0.00862 | 2e-05 | 0.79343 | 0.79205 | 0.79343 | 0.79205 | [W&B Train](https://wandb.ai/niloydebbarma-ai-youth-alliance/bangla-lora-bn-tpu/runs/u6laj39m) | [W&B Eval](https://wandb.ai/niloydebbarma-ai-youth-alliance/eval-bangla-lora-bn-tpu/runs/frq17f5g) | [HF Model](https://huggingface.co/niloydebbarma/BanglaSenti-XLM-RoBERTa-Experiment-Models/tree/main/ex-16-32-qv) |
| Exp-R32    | xlm-roberta-base | 32 | 64 | ["query", "value"] | 0.1 | 64 | 5e-5 | 10 (0-9) | 1529 | 0.7959 | 0.78919 | 0.53123 | 0.79547 | 0.78675 | 0.00835 | -3e-05 | 0.79604 | 0.7951 | 0.79604 | 0.7951 | [W&B Train](https://wandb.ai/niloydebbarma-ai-youth-alliance/bangla-lora-bn-tpu/runs/8slt2m60) | [W&B Eval](https://wandb.ai/niloydebbarma-ai-youth-alliance/eval-bangla-lora-bn-tpu/runs/d5r8s7ti) | [HF Model](https://huggingface.co/niloydebbarma/BanglaSenti-XLM-RoBERTa-Experiment-Models/tree/main/ex-32-64-qv) |
| Exp-TM-QKV | xlm-roberta-base | 32 | 64 | ["query", "key", "value"] | 0.1 | 64 | 5e-5 | 10 (0-9) | 1529 | 0.80275 | 0.79637 | 0.51567 | 0.80234 | 0.79398 | 0.00822 | 0.0 | 0.79882 | 0.79759 | 0.79882 | 0.79759 | [W&B Train](https://wandb.ai/niloydebbarma-ai-youth-alliance/bangla-lora-bn-tpu/runs/e3g9xqet) | [W&B Eval](https://wandb.ai/niloydebbarma-ai-youth-alliance/eval-bangla-lora-bn-tpu/runs/4v56734e) | [HF Model](https://huggingface.co/niloydebbarma/BanglaSenti-XLM-RoBERTa-Experiment-Models/tree/main/ex-32-64-tm-qkv) |
| **Exp-TM-All** | **xlm-roberta-base** | **32** | **64** | **["query", "key", "value", "dense"]** | **0.1** | **64** | **5e-5** | **10 (0-9)** | **1529** | **0.83005** | **0.82504** | **0.44704** | **0.81068** | **0.8032** | **0.00808** | **-1e-05** | **0.80528** | **0.80415** | **0.80724** | **0.80645** | **[W&B Train](https://wandb.ai/niloydebbarma-ai-youth-alliance/bangla-lora-bn-tpu/runs/nu1qo8cm)** | **[W&B Eval](https://wandb.ai/niloydebbarma-ai-youth-alliance/eval-bangla-lora-bn-tpu/runs/gfsmfb7f)** | **[HF Model](https://huggingface.co/niloydebbarma/BanglaSenti-XLM-RoBERTa-Experiment-Models/tree/main/ex-32-64-tm-all)** |


## Comparison: Exp-TM-All (With LoRA) vs Baseline (Without LoRA)

The following table directly compares the best LoRA model (Exp-TM-All) to the baseline, showing absolute values and percentage changes for each metric. **Bold** indicates the best result for each metric. 

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


## W&B Reports

Interactive W&B reports provide detailed training and evaluation visualizations for both baseline and LoRA experiments.

### BanglaSenti XLM-RoBERTa Training: Baseline vs LoRA

- [BanglaSenti XLM-RoBERTa Training: Baseline vs LoRA](https://wandb.ai/niloydebbarma-ai-youth-alliance/bangla-lora-bn-tpu/reports/BanglaSenti-XLM-RoBERTa-Training-Baseline-vs-LoRA-q6hb0von)

<iframe src="https://api.wandb.ai/links/niloydebbarma-ai-youth-alliance/q6hb0von" style="border:none;height:1024px;width:100%"></iframe>

### BanglaSenti XLM-RoBERTa Evaluation: Baseline vs LoRA

- [BanglaSenti XLM-RoBERTa Evaluation: Baseline vs LoRA](https://wandb.ai/niloydebbarma-ai-youth-alliance/bangla-lora-bn-tpu/reports/BanglaSenti-XLM-RoBERTa-Evaluation-Baseline-vs-LoRA-fpjt1prn)

<iframe src="https://api.wandb.ai/links/niloydebbarma-ai-youth-alliance/fpjt1prn" style="border:none;height:1024px;width:100%"></iframe>


## Summary of Findings

- LoRA fine-tuning on XLM-RoBERTa-base yields a dramatic improvement in BanglaSenti sentiment analysis compared to the baseline.
- The best LoRA configuration (Exp-TM-All) achieves over 80% accuracy and F1 on validation and test sets, with a 145–291% relative improvement over the baseline.
- Loss is reduced by more than 50% compared to the baseline.
- The best results are obtained when LoRA is applied to all target modules (query, key, value, dense) with rank 32 and alpha 64.
- All results are fully reproducible, with code, configs, and checkpoints provided.
- W&B reports and Hugging Face links are included for transparency and further analysis.