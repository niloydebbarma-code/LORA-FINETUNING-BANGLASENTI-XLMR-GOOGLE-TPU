# LoRA Fine-Tuning for Bangla Sentiment Analysis on XLM-RoBERTa-Base using TPUs

import os
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import wandb
import logging
from omegaconf import OmegaConf
from utils.argparse_config import get_arg_parser, load_config
from utils.data_utils import load_and_clean_csv
from utils.tokenization import get_tokenizer, tokenize_batch
from models.model_loader import load_pretrained_model
from models.lora_adapter import add_lora_adapters, freeze_base_layers, log_model_stats
from sklearn.metrics import f1_score
import numpy as np
import traceback


def append_main_log(msg):
    # Append a message to the main experiment log file
    main_log_file = os.path.join('logs', 'train_banglasenti_main.log')
    with open(main_log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


def log_param_devices(model, stage, logger):
    # Log the device location of model parameters
    device_counts = {}
    meta_params = []
    
    # Check each parameter's device
    for name, param in model.named_parameters():
        device = str(param.device)
        device_counts[device] = device_counts.get(device, 0) + 1
        
        # Track meta tensors
        if device == 'meta':
            meta_params.append(name)
    
    # Log results
    logger.info(f"[DEVICE CHECK] {stage}: parameter device counts: {device_counts}")
    
    # Warning for meta tensors
    if meta_params:
        logger.error(f"[DEVICE CHECK] {stage}: Found {len(meta_params)} parameters on 'meta' device: {meta_params}")
    else:
        logger.info(f"[DEVICE CHECK] {stage}: All parameters correctly placed")


def train_worker(flags):
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'train_banglasenti.log')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Remove all handlers associated with the root logger object
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Logger initialized and log file set up.")
    config = OmegaConf.to_container(flags, resolve=True)
    xm.set_rng_state(42)
    logger.info("Set RNG state.")
    train_path = config['dataset']['train_path']
    val_path = config['dataset']['val_path']
    logger.info(f"Loading train CSV: {train_path}")
    train_df = load_and_clean_csv(train_path)
    logger.info(f"Loading val CSV: {val_path}")
    val_df = load_and_clean_csv(val_path)
    logger.info(f"Loaded train/val CSVs. Train shape: {train_df.shape}, Val shape: {val_df.shape}")
    tokenizer = get_tokenizer(config['model_name'])
    logger.info("Tokenizer loaded.")
    train_enc = tokenize_batch(tokenizer, train_df['text'].tolist(), max_length=config['train']['max_length'])
    val_enc = tokenize_batch(tokenizer, val_df['text'].tolist(), max_length=config['train']['max_length'])
    logger.info("Tokenized train/val data.")
    num_labels = 3
    # Always load model on CPU, then move to XLA device
    model = load_pretrained_model(config['model_name'], num_labels=num_labels, device='cpu')
    log_param_devices(model, 'after model load', logger)
    model = add_lora_adapters(model, config['lora'])
    # Assign peft_config for saving (dict in this workflow)
    peft_config = config['lora']
    log_param_devices(model, 'after add_lora_adapters', logger)
    freeze_base_layers(model)
    log_model_stats(model)
    model = model.to(xm.xla_device())
    log_param_devices(model, 'after move to XLA device', logger)
    logger.info("Model loaded, LoRA adapters added, base layers frozen, model on XLA device.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['lr'])
    logger.info("Optimizer created.")
    label_counts = train_df['label'].value_counts().sort_index()
    class_weights = torch.tensor([1.0 / count for count in label_counts], dtype=torch.float32).to(xm.xla_device())
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    label_dtype = torch.long
    train_label_tensor = torch.tensor(train_df['label'].values, dtype=label_dtype)
    val_label_tensor = torch.tensor(val_df['label'].values, dtype=label_dtype)
    train_data = torch.utils.data.TensorDataset(train_enc['input_ids'], train_enc['attention_mask'], train_label_tensor)
    val_data = torch.utils.data.TensorDataset(val_enc['input_ids'], val_enc['attention_mask'], val_label_tensor)
    logger.info("TensorDatasets created.")
    # Set up data loaders
    train_sampler = torch.utils.data.RandomSampler(train_data)
    val_sampler = torch.utils.data.SequentialSampler(val_data)
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config['train']['batch_size'], sampler=train_sampler, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=config['train']['batch_size'], sampler=val_sampler, num_workers=0
    )
    logger.info("DataLoaders created.")
    # Label distribution check and log
    unique_labels, label_counts = np.unique(train_df['label'].values, return_counts=True)
    label_dist = dict(zip(unique_labels, label_counts))
    logger.info(f"Train label distribution: {label_dist}")
    if set(unique_labels) != {0, 1, 2}:
        logger.error(f"Training data does not contain all three labels 0, 1, 2. Found: {sorted(unique_labels)}")
        raise ValueError(f"Training data does not contain all three labels 0, 1, 2. Found: {sorted(unique_labels)}")
    unique_labels_val, label_counts_val = np.unique(val_df['label'].values, return_counts=True)
    label_dist_val = dict(zip(unique_labels_val, label_counts_val))
    logger.info(f"Val label distribution: {label_dist_val}")
    if set(unique_labels_val) != {0, 1, 2}:
        logger.error(f"Validation data does not contain all three labels 0, 1, 2. Found: {sorted(unique_labels_val)}")
        raise ValueError(f"Validation data does not contain all three labels 0, 1, 2. Found: {sorted(unique_labels_val)}")
    # Initialize wandb if enabled
    use_wandb = config['train']['wandb']
    if use_wandb:
        wandb.init(project=config['train']['wandb_project'], config=config)
        # Log the wandb run URL for easy access
        wandb_url = wandb.run.url
        logger.info(f"Weights & Biases run: {wandb_url}")
        append_main_log(f"Weights & Biases run: {wandb_url}")
    best_acc = 0.0
    checkpoint_dir = os.path.join(config['train']['checkpoint_dir'], 'banglasenti-lora-xlmr')
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info("Starting training loop...")
    logger.info("Using input tensor dtype: torch.int64 (long) for input_ids and attention_mask")
    try:
        logger.info("Entering training loop.")
        for epoch in range(config['train']['epochs']):
            logger.info(f"Epoch {epoch} started")
            model.train()
            step_count = 0
            train_pl = pl.ParallelLoader(train_loader, [xm.xla_device()]).per_device_loader(xm.xla_device())
            epoch_loss = 0.0
            lora_weight_means = []
            train_preds = []
            train_labels = []
            for step, (x, attn_mask, y) in enumerate(train_pl):
                optimizer.zero_grad()
                x = x.to(torch.long)
                attn_mask = attn_mask.to(torch.long)
                outputs = model(x, attention_mask=attn_mask)
                loss = criterion(outputs.logits, y)
                if not torch.isfinite(loss):
                    logger.error(f"Non-finite loss detected at step {step}, epoch {epoch}. Skipping update.")
                    continue
                loss.backward()
                xm.optimizer_step(optimizer)
                lora_params = [p for n, p in model.named_parameters() if 'lora' in n]
                if lora_params:
                    lora_mean = torch.cat([p.detach().flatten() for p in lora_params]).mean().item()
                    lora_weight_means.append(lora_mean)
                if step % 100 == 0 or step == 0 or step == len(train_pl) - 1:
                    logger.info(f"Epoch {epoch} Step {step} | Loss: {loss.item():.4f}")
                preds = torch.argmax(outputs.logits, dim=1)
                train_preds.append(preds.cpu())
                train_labels.append(y.cpu())
                step_count += 1
                epoch_loss += loss.item()
                if config['train']['wandb'] and (step % 100 == 0 or step == 0 or step == len(train_pl) - 1):
                    step_acc = (preds == y).float().mean().item()
                    step_f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average='macro')
                    wandb.log({'train/step_loss': loss.item(), 'train/step_acc': step_acc, 'train/step_f1': step_f1, 'step': step, 'epoch': epoch})
            avg_loss = epoch_loss / step_count if step_count > 0 else 0.0
            train_preds_cat = torch.cat(train_preds)
            train_labels_cat = torch.cat(train_labels)
            train_acc = (train_preds_cat == train_labels_cat).float().mean().item() if len(train_preds_cat) > 0 else 0.0
            train_f1 = f1_score(train_labels_cat.numpy(), train_preds_cat.numpy(), average='macro') if len(train_preds_cat) > 0 else 0.0
            # Validation
            model.eval()
            val_pl = pl.ParallelLoader(val_loader, [xm.xla_device()]).per_device_loader(xm.xla_device())
            correct, total = 0, 0
            all_preds = []
            all_labels = []
            val_loss = 0.0
            with torch.no_grad():
                for x, attn_mask, y in val_pl:
                    x = x.to(torch.long)
                    attn_mask = attn_mask.to(torch.long)
                    outputs = model(x, attention_mask=attn_mask)
                    loss = criterion(outputs.logits, y)
                    preds = torch.argmax(outputs.logits, dim=1)
                    all_preds.append(preds.cpu())
                    all_labels.append(y.cpu())
                    correct += (preds == y).sum().item()
                    total += y.size(0)
                    val_loss += loss.item()
            if total == 0:
                acc = 0.0
            else:
                acc = correct / total
            if not (0.0 <= acc <= 1.0) or not torch.isfinite(torch.tensor(acc)):
                logger.error(f"Non-finite or out-of-bounds accuracy detected at epoch {epoch}. Setting acc=0.0.")
                acc = 0.0
            all_preds_cat = torch.cat(all_preds)
            all_labels_cat = torch.cat(all_labels)
            val_f1 = f1_score(all_labels_cat.numpy(), all_preds_cat.numpy(), average='macro') if total > 0 else 0.0
            avg_val_loss = val_loss / total if total > 0 else 0.0
            logger.info(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {val_f1:.4f}")
            # Calculate LoRA weight mean for monitoring
            lora_mean = sum(lora_weight_means)/len(lora_weight_means) if lora_weight_means else 0.0
            
            if use_wandb:
                wandb.log({
                    'train/avg_loss': avg_loss,
                    'train/acc': train_acc,
                    'train/f1': train_f1,
                    'val/avg_loss': avg_val_loss,
                    'val/accuracy': acc,
                    'val/f1': val_f1,
                    'lora/mean_weight': lora_mean,
                    'epoch': epoch
                })
            if acc > best_acc:
                best_acc = acc
                logger.info(f"New best validation accuracy: {acc:.4f} > previous best: {best_acc:.4f}. Saving updated checkpoint.")
                # Save best model weights and HuggingFace-compatible model
                best_weights_path = os.path.join(checkpoint_dir, 'lora_xlmr_weights.pt')
                logger.info("Transferring best model weights from TPU to disk...")
                xm.save(model.state_dict(), best_weights_path)
                
                # Verify successful saving
                if os.path.exists(best_weights_path):
                    weights_size = os.path.getsize(best_weights_path) / (1024 * 1024)  # Size in MB
                    logger.info(f"Best model weights successfully saved to {best_weights_path} ({weights_size:.2f} MB)")
                try:
                    # Save the adapter config first
                    logger.info("Saving PEFT adapter config and state dict")
                    adapter_config_path = os.path.join(checkpoint_dir, 'lora_adapter_weights')
                    os.makedirs(adapter_config_path, exist_ok=True)
                    
                    # Save the model's config
                    model.config.save_pretrained(adapter_config_path)

                    # Simple, explicit PEFT config saving: always save as adapter_config.json using to_dict() or as dict
                    import json
                    adapter_config_file = os.path.join(adapter_config_path, "adapter_config.json")
                    if hasattr(peft_config, "to_dict"):
                        with open(adapter_config_file, "w") as f:
                            json.dump(peft_config.to_dict(), f, indent=2)
                        logger.info(f"PEFT config saved as JSON (from to_dict) to {adapter_config_file}")
                    elif isinstance(peft_config, dict):
                        with open(adapter_config_file, "w") as f:
                            json.dump(peft_config, f, indent=2)
                        logger.info(f"PEFT config saved as JSON (from dict) to {adapter_config_file}")
                    else:
                        logger.warning(f"PEFT config could not be saved: unknown type {type(peft_config)} to {adapter_config_file}")
                    
                    # Save state dict to CPU - this is safer for TPU models
                    # First save with XM, then load on CPU and save as regular pytorch file
                    state_dict_path = os.path.join(checkpoint_dir, 'lora_adapter_state_dict.pt')
                    
                    # Use xm.save which safely handles data movement from TPU to host memory
                    logger.info("Safely transferring model state from TPU to disk...")
                    xm.save(model.state_dict(), state_dict_path)
                    
                    # Verify the checkpoint file exists to confirm successful saving
                    if os.path.exists(state_dict_path):
                        checkpoint_size = os.path.getsize(state_dict_path) / (1024 * 1024)  # Size in MB
                        logger.info(f"Model state dict successfully saved to {state_dict_path} ({checkpoint_size:.2f} MB)")
                    
                    # Save tokenizer
                    tokenizer.save_pretrained(os.path.join(checkpoint_dir, 'lora_xlmr_tokenizer'))
                    
                    # Save additional info about the model
                    with open(os.path.join(checkpoint_dir, 'model_info.txt'), 'w') as f:
                        f.write(f"Base model: {config['model_name']}\n")
                        f.write(f"LoRA config: {config['lora']}\n")
                        f.write(f"Accuracy: {acc}\n")
                        f.write(f"F1 Score: {val_f1}\n")
                        
                    logger.info(f"Model adapter config, state dict, and tokenizer saved in {checkpoint_dir}")
                except Exception as e:
                    logger.error(f"Error saving with save_pretrained: {e}")
                    logger.error(traceback.format_exc())
        # Save final model state regardless of best accuracy
        final_checkpoint_dir = os.path.join(checkpoint_dir, 'final_state')
        os.makedirs(final_checkpoint_dir, exist_ok=True)
        
        # Save final weights using xm.save
        logger.info("Saving final model state after training completion")
        final_weights_path = os.path.join(final_checkpoint_dir, 'final_lora_xlmr_weights.pt')
        logger.info("Transferring final model weights from TPU to disk...")
        xm.save(model.state_dict(), final_weights_path)
        
        # Verify successful saving
        if os.path.exists(final_weights_path):
            weights_size = os.path.getsize(final_weights_path) / (1024 * 1024)  # Size in MB
            logger.info(f"Final model weights successfully saved to {final_weights_path} ({weights_size:.2f} MB)")
        
        try:
            # Save the adapter config first
            logger.info("Saving final PEFT adapter config and state dict")
            adapter_config_path = os.path.join(final_checkpoint_dir, 'final_lora_adapter_weights')
            os.makedirs(adapter_config_path, exist_ok=True)
            
            # Save the model's config
            model.config.save_pretrained(adapter_config_path)

            # Simple, explicit PEFT config saving: always save as adapter_config.json using to_dict() or as dict
            import json
            adapter_config_file = os.path.join(adapter_config_path, "adapter_config.json")
            if hasattr(peft_config, "to_dict"):
                with open(adapter_config_file, "w") as f:
                    json.dump(peft_config.to_dict(), f, indent=2)
                logger.info(f"PEFT config saved as JSON (from to_dict) to {adapter_config_file}")
            elif isinstance(peft_config, dict):
                with open(adapter_config_file, "w") as f:
                    json.dump(peft_config, f, indent=2)
                logger.info(f"PEFT config saved as JSON (from dict) to {adapter_config_file}")
            else:
                logger.warning(f"PEFT config could not be saved: unknown type {type(peft_config)} to {adapter_config_file}")
            
            # Save state dict to CPU - this is safer for TPU models
            # First save with XM, then load on CPU and save as regular pytorch file
            state_dict_path = os.path.join(final_checkpoint_dir, 'final_lora_adapter_state_dict.pt')
            
            # Use xm.save which safely handles data movement from TPU to host memory
            logger.info("Safely transferring final model state from TPU to disk...")
            xm.save(model.state_dict(), state_dict_path)
            
            # Verify the checkpoint file exists to confirm successful saving
            if os.path.exists(state_dict_path):
                checkpoint_size = os.path.getsize(state_dict_path) / (1024 * 1024)  # Size in MB
                logger.info(f"Final model state dict successfully saved to {state_dict_path} ({checkpoint_size:.2f} MB)")
            
            # Save tokenizer
            tokenizer.save_pretrained(os.path.join(final_checkpoint_dir, 'final_tokenizer'))
            
            # Save additional info about the final model
            with open(os.path.join(final_checkpoint_dir, 'model_info.txt'), 'w') as f:
                f.write(f"Base model: {config['model_name']}\n")
                f.write(f"LoRA config: {config['lora']}\n")
                f.write(f"Final training accuracy: {train_acc}\n")
                f.write(f"Final validation accuracy: {acc}\n")
                f.write(f"Final validation F1: {val_f1}\n")
                
            logger.info(f"Final model adapter config, state dict, and tokenizer saved in {final_checkpoint_dir}")
        except Exception as e:
            logger.error(f"Error saving final model state with save_pretrained: {e}")
            logger.error(traceback.format_exc())
        
        # Log final results
        logger.info(f"Final best validation accuracy: {best_acc:.4f}")
        logger.info(f"Run summary: config, best_acc, final model state saved in {checkpoint_dir}")
        append_main_log(f"Final best validation accuracy: {best_acc:.4f}")
        
        # Data safety: Clear any cached tensors to prevent leakage
        logger.info("Performing TPU data safety cleanup...")
        try:
            # Explicitly delete training data tensors from TPU memory
            del train_data, val_data, train_enc, val_enc
            del train_df, val_df
            
            # Clear model from TPU memory to free resources
            del model
            
            # For TPUs, mark step to ensure tensors are actually released
            xm.mark_step()
            
            # Force garbage collection
            import gc
            gc.collect()
            logger.info("TPU data safety cleanup completed successfully")
        except Exception as e:
            logger.warning(f"TPU data cleanup warning (non-critical): {str(e)}")
        
        if use_wandb:
            # Log the final accuracy before finishing the wandb run
            wandb.log({'final/best_val_accuracy': best_acc})
            wandb.finish()
            
        logger.info("Training complete.")
    except Exception as e:
        logger.error(f"Exception during training: {e}")
        logger.error(traceback.format_exc())
        raise

def _mp_fn(index, flags):
    # Entry point for the TPU process
    train_worker(flags)


if __name__ == '__main__':
    # Parse command line arguments and load configuration
    parser = get_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Set up logging directory
    os.makedirs('logs', exist_ok=True)
    
    print("Starting LoRA fine-tuning for Bangla sentiment analysis")
    
    # Launch training process
    xmp.spawn(_mp_fn, args=(config,), nprocs=1, start_method='fork')
