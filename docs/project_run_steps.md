# BanglaSenti LoRA Fine-Tuning: Step-by-Step TPU Run Guide

Follow these steps to run BanglaSenti LoRA fine-tuning on a Google Cloud TPU. Each section covers only what you need to get started and finish a training run. 
---

## 1. Launch a Google Cloud TPU VM

1. Go to your Google Cloud project:
   - [Google Cloud TPU Console](https://console.cloud.google.com/compute/tpus?)
2. Click **Create TPU** and fill in:
   - **Name:** lora-bn-tpu
   - **Zone:** africa-south1-a (or your preferred region)
   - **TPU type:** v3-8 (or as needed)
   - **TPU software version:** (latest supported)
3. Click **Create**.

---

## 2. Connect to the TPU VM

- Click **SSH** or **SSH in browser window** next to your TPU VM in the console.

---

## 3. Start a tmux Session (Required)

You must use `tmux` for all work in this project to ensure your training continues even if your SSH session disconnects.

Start a new tmux session:
```bash
tmux
```
If you disconnect, SSH back in and reattach to your session:
```bash
tmux attach -t 0
```

---

## 4. Clone and Set Up the Project

```bash
git clone https://github.com/niloydebbarma-code/LoRA-Fine-Tuning-of-BanglaSenti-on-Bangla-BERT-Base-Using-Google-TPUs.git
cd LoRA-Fine-Tuning-of-BanglaSenti-on-Bangla-BERT-Base-Using-Google-TPUs
rm -rf logs/*
rm -rf checkpoints/*
mkdir logs
mkdir -p checkpoints/banglasenti-lora-xlmr/{lora_xlmr_transformers_model,lora_xlmr_base_model,lora_xlmr_tokenizer}
```

Install requirements:
```bash
pip install -r requirements.txt
```

---

## 5. Verify TPU Environment

```bash
pip list | grep torch
python -c "import torch_xla.core.xla_model as xm; print('Available TPU cores:', xm.get_xla_supported_devices()); print('Current device:', xm.xla_device())"
```

**Expected output:**
```
torch                    2.7.1
torch-xla                2.7.0
torchvision              0.22.1
WARNING:root:libtpu.so and TPU device found. Setting PJRT_DEVICE=TPU.
Available TPU cores: ['xla:0', 'xla:1', 'xla:2', 'xla:3', 'xla:4', 'xla:5', 'xla:6', 'xla:7']
Current device: xla:0
```

---

## 6. Prepare Data

Prepare the dataset (creates train/val/test splits):
```bash
python -m data.prepare
```

---

## 7. Check/Edit Configs Before Training/Evaluation

Open and review/edit the config files as needed:
```bash
nano configs/train.yaml
nano configs/eval.yaml
```
- To save changes: Press `Ctrl + O`, then `Enter`.
- To exit nano: Press `Ctrl + X`.

---

## 8. Train and Evaluate (With and Without LoRA)

**With LoRA:**
```bash
python -m train.train_lora --config configs/train.yaml
python -m eval.eval_all --config configs/eval.yaml
```

**Without LoRA (baseline):**
```bash
python -m train.train_xla --config configs/train.yaml
python -m eval.eval_xla --config configs/eval-xla.yaml
```

---

## 8.1 Monitoring and Detaching Training (W&B and tmux)

- If `wandb` is enabled in your config, after starting training you will see a W&B run link in the terminal. Visit this link in your browser to monitor live metrics and logs.
- To safely detach from your tmux session (training will continue in the background):
  - Press `Ctrl + B` (hold Ctrl, press B), then press `D`.
- You can now close your browser window or disconnect from the internet. Training will continue on the TPU VM.
- To check logs locally, run:
```bash
ls logs
nano logs/train_banglasenti.log
```
- If you see epoch progress in the log, training is running correctly.
- You can also monitor in W&B if enabled.

---

## 8.2 Resuming and Inspecting After Detach

- To reattach to your tmux session after disconnecting:
```bash
tmux ls
# Find your session (usually 0), then:
tmux attach -t 0
```
- Inspect logs or run evaluation as needed:
```bash
nano logs/train_banglasenti.log
nano logs/eval_run.log
```
- Run evaluation after training completes:
```bash
python -m eval.eval_all --config configs/eval.yaml
```
- Check evaluation logs in W&B (if enabled) or in the `logs/` folder.

---

## 9. Inspect Logs and Configs

```bash
nano logs/train_banglasenti.log
nano logs/train_banglasenti_xla.log
nano configs/train.yaml
nano configs/eval.yaml
nano configs/eval-xla.yaml
```

---

## 10. Clean Up (Optional)

To avoid unnecessary charges, delete your TPU VM when finished:

```bash
gcloud compute tpus tpu-vm delete lora-bn-tpu --zone=africa-south1-a
```

---

**Note:**
- After the log file is created, logs are not displayed in the terminal by default. This is intentional to keep the terminal output clean and does not indicate a fault or bug. To monitor training progress or check for issues, open the log file with `nano` or use `tail -f`. Logs are also available in Weights & Biases (W&B) if enabled.

---

**Tips:**
- Always monitor your logs for errors or warnings.
- Use the provided configs as templates for your own experiments.
- For long-running jobs, consider using `tmux` to keep your session alive.
- All results and checkpoints are saved locally for full reproducibility.

---

For questions or contributions, please refer to the repository.
