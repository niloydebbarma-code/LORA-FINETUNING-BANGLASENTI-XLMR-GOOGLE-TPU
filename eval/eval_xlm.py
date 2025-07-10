import os
import logging
import torch
import numpy as np
from omegaconf import OmegaConf
from utils.argparse_config import get_arg_parser, load_config
from utils.data_utils import load_and_clean_csv
from utils.tokenization import get_tokenizer_local, tokenize_batch
from models.model_loader import load_pretrained_model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch_xla.core.xla_model as xm
import wandb

logger = logging.getLogger(__name__)

def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels_tensor = batch[2].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(labels_tensor.cpu().numpy())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    cm = confusion_matrix(labels, preds, labels=list(range(model.config.num_labels)))
    logger.info(f"Confusion matrix (rows=true, cols=pred):\n{cm}")
    logger.info(f"Per-class support (true labels): {np.bincount(labels, minlength=model.config.num_labels)}")
    logger.info(f"Per-class predictions: {np.bincount(preds, minlength=model.config.num_labels) if len(preds) > 0 else []}")
    return acc, f1

def main():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'eval_run_xlm.log')
    logger.setLevel(logging.INFO)

    # Remove all handlers associated with the logger object
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Logger initialized and log file set up.")

    parser = get_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    config = OmegaConf.to_container(config, resolve=True)

    # Load evaluation-specific config
    eval_config = {
        'base_model_path_during': config['base_model_path_during'],
        'tokenizer_path_during': config['tokenizer_path_during'],
        'base_model_path_final': config['base_model_path_final'],
        'tokenizer_path_final': config['tokenizer_path_final'],
        'dataset': config['dataset'],
        'max_length': config['max_length'],
        'batch_size': config['batch_size'],
        'wandb': config.get('wandb', False),
        'wandb_project': config.get('wandb_project', None)
    }

    # Initialize WandB if enabled
    if eval_config.get('wandb', False):
        wandb.init(project=eval_config['wandb_project'], config=eval_config)
        logger.info("WandB initialized for evaluation.")
        logger.info(f"WandB run link: {wandb.run.get_url()}")
    else:
        logger.info("WandB logging is disabled.")

    # Paths for dual models
    paths = {
        "during": {
            "base_model_path": eval_config['base_model_path_during'],
            "tokenizer_path": eval_config['tokenizer_path_during']
        },
        "final": {
            "base_model_path": eval_config['base_model_path_final'],
            "tokenizer_path": eval_config['tokenizer_path_final']
        }
    }

    for model_type, model_paths in paths.items():
        logger.info(f"Evaluating {model_type} model...")

        # Ensure paths are absolute
        base_model_path = os.path.abspath(model_paths['base_model_path'])
        tokenizer_path = os.path.abspath(model_paths['tokenizer_path'])

        # Check for required files
        for path, desc in [
            (base_model_path, 'Base model weights'),
            (tokenizer_path, 'Tokenizer'),
        ]:
            if not os.path.exists(path):
                logger.error(f"{desc} not found at {path}")
                raise FileNotFoundError(f"{desc} not found at {path}")

        # Load tokenizer
        tokenizer = get_tokenizer_local(tokenizer_path)

        # Load dataset
        df = load_and_clean_csv(eval_config['dataset']['main'])
        enc = tokenize_batch(tokenizer, df['text'].tolist(), max_length=eval_config['max_length'])
        labels_tensor = torch.tensor(df['label'].values.astype(np.int64))
        data = torch.utils.data.TensorDataset(enc['input_ids'], enc['attention_mask'], labels_tensor)
        loader = torch.utils.data.DataLoader(data, batch_size=eval_config['batch_size'])

        # Load model
        num_labels = 3
        device = xm.xla_device()
        model = load_pretrained_model(config['model_name'], num_labels=num_labels, device='cpu')
        model.to(device)

        # Evaluate
        acc, f1 = evaluate(model, loader, device)
        if eval_config.get('wandb', False):
            wandb.log({f"{model_type}_accuracy": acc, f"{model_type}_f1": f1})
        logger.info(f"{model_type.capitalize()} model eval done. Accuracy={acc:.4f}, F1={f1:.4f}")
        print(f"{model_type.capitalize()} model eval done. Accuracy={acc:.4f}, F1={f1:.4f}")

    if eval_config.get('wandb', False):
        wandb.finish()

if __name__ == '__main__':
    main()
