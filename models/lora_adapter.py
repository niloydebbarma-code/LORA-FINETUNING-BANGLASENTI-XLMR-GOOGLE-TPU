import torch
import torch.nn as nn
import logging
from peft import get_peft_model, LoraConfig, TaskType

logger = logging.getLogger(__name__)

def add_lora_adapters(model, lora_config):
    # Add LoRA adapters to model, ensuring parameters are on the same device as base model
    logger.info(f"Initializing LoRA adapters with configuration: {lora_config}")
    
    # Extract the device of the base model to ensure consistent device placement
    device = next(model.parameters()).device
    logger.info(f"Ensuring LoRA adapters will be created on the same device as base model: {device}")
    
    # Configure LoRA parameters
    config = LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['alpha'],
        target_modules=lora_config['target_modules'],
        lora_dropout=lora_config['dropout'],
        bias=lora_config['bias'],
        task_type=TaskType[lora_config['task_type']]
    )
    
    # Apply LoRA to model and ensure it's on the correct device
    model = get_peft_model(model, config).to(device)
    logger.info(f"LoRA adapters successfully added to model. Configuration: {config}")
    return model

def freeze_base_layers(model):
    # Freeze all parameters except LoRA adapter parameters to reduce training complexity
    logger.info("Freezing base model layers to ensure only LoRA parameters are trainable")
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    logger.info("Base model layers have been successfully frozen")


def log_model_stats(model):
    # Calculate and log model parameter statistics for debugging and verification
    # Calculate parameter statistics
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable/total if total > 0 else 0
    
    # Log summary statistics
    logger.info("Model parameter statistics:")
    logger.info(f"  Total parameters: {total:,}")
    logger.info(f"  Trainable parameters: {trainable:,}")
    logger.info(f"  Trainable ratio: {ratio:.4f}")
    
    # Log details of trainable parameters
    logger.info("Trainable parameter details:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"  {name}: {param.shape}")

def load_lora_adapters(model, adapter_config_path, adapter_weights_path):
    import os
    logger.info(f"Loading LoRA adapters with config: {adapter_config_path} and weights: {adapter_weights_path}")

    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"Adapter configuration file not found at {adapter_config_path}")

    if not os.path.exists(adapter_weights_path):
        raise FileNotFoundError(f"Adapter weights file not found at {adapter_weights_path}")

    # Load the adapter configuration
    from peft import LoraConfig
    import json

    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)

    lora_config = LoraConfig(
        r=adapter_config['r'],
        lora_alpha=adapter_config['alpha'],
        target_modules=adapter_config['target_modules'],
        lora_dropout=adapter_config['dropout'],
        bias=adapter_config['bias'],
        task_type=adapter_config['task_type']
    )

    # Apply LoRA configuration to the model
    model = get_peft_model(model, lora_config)

    # Load the adapter weights
    state_dict = torch.load(adapter_weights_path, map_location=model.device)
    model.load_state_dict(state_dict, strict=False)

    logger.info(f"LoRA adapters successfully loaded with config: {adapter_config_path} and weights: {adapter_weights_path}")

    return model

