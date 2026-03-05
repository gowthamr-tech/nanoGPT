import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import argparse
import torch
from src.utils.config import load_config
from src.utils.tokenizer import CharTokenizer
from src.models.transformer import GPT
from src.training.trainer import Trainer

parser = argparse.ArgumentParser(description="Train nanoGPT on a character-level dataset.")
parser.add_argument(
    "--config",
    default="configs/train.yaml",
    help="Path to training config YAML file."
)
parser.add_argument(
    "--data-path",
    default="data/input.txt",
    help="Path to plaintext dataset used for training."
)
args = parser.parse_args()

config = load_config(args.config)

with open(args.data_path, "r", encoding="utf-8") as f:
    text = f.read()
tokenizer = CharTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT(tokenizer.vocab_size, config)

trainer = Trainer(model, data, config, device)
trainer.train()
