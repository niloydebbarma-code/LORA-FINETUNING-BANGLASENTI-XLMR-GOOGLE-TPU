import click
import torch
from models.model_loader import load_pretrained_model
from utils.tokenization import get_tokenizer, tokenize_batch

@click.command()
@click.option('--model_path', required=True, help='Path to model checkpoint (e.g. checkpoints/banglasenti-lora-xlmr/hf_model)')
@click.option('--text', required=True, help='Input text for prediction')
@click.option('--max_length', default=256, help='Max sequence length')
def predict(model_path, text, max_length):
    # Make a prediction using the trained BanglaSenti LoRA XLM-R model
    model = load_pretrained_model(model_path)
    tokenizer = get_tokenizer(model_path)
    enc = tokenize_batch(tokenizer, [text], max_length=max_length)
    with torch.no_grad():
        logits = model(enc['input_ids'])
        pred = torch.argmax(logits.logits, dim=1).item()
    click.echo(f"Prediction: {pred}")

if __name__ == '__main__':
    predict()
