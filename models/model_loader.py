import torch
from transformers import AutoModelForSequenceClassification
from loguru import logger
import torch_xla.core.xla_model as xm

def load_pretrained_model(model_name, num_labels=3, device=None):
    # Always load on CPU, then move to XLA device (never use device_map or meta)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    if device is None:
        device = xm.xla_device()
    model = model.to(device)
    logger.info(f"Loaded {model_name} for sequence classification on {device}")
    return model
