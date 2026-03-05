import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
from src.utils.tokenizer import CharTokenizer
from src.models.transformer import GPT
from src.utils.config import load_config

parser = argparse.ArgumentParser(description="Generate text from a trained nanoGPT checkpoint.")
parser.add_argument(
    "--config",
    default="configs/train.yaml",
    help="Path to config YAML file used to create the model."
)
parser.add_argument(
    "--data-path",
    default="data/input.txt",
    help="Path to plaintext dataset used to build tokenizer vocabulary."
)
parser.add_argument(
    "--checkpoint",
    default="checkpoints/model.pt",
    help="Path to model checkpoint."
)
parser.add_argument(
    "--tokens",
    type=int,
    default=200,
    help="Number of tokens to generate."
)
args = parser.parse_args()

config = load_config(args.config)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = args.data_path
if not os.path.isabs(data_path):
    data_path = os.path.join(BASE_DIR, data_path)

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()
tokenizer = CharTokenizer(text)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT(tokenizer.vocab_size, config).to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
model.eval()

context = torch.zeros((1, 1), dtype=torch.long).to(device)

for _ in range(args.tokens):
    context_cond = context[:, -config["block_size"]:]
    logits = model(context_cond)
    logits = logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    context = torch.cat((context, next_token), dim=1)

print(tokenizer.decode(context[0].tolist()))
