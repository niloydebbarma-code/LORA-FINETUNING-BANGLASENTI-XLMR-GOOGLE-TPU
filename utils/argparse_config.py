import argparse
from omegaconf import OmegaConf

def get_arg_parser():
    # Returns an argument parser for BanglaSenti LoRA XLM-R training and evaluation
    parser = argparse.ArgumentParser(description='BanglaSenti LoRA XLM-R CLI')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--log_file', type=str, default=None, help='Log file path')
    parser.add_argument('--device', type=str, default='xla', help='Device: xla/cpu')
    return parser

def load_config(config_path):
    # Loads a YAML config using OmegaConf
    return OmegaConf.load(config_path)
