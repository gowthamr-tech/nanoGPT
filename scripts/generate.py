import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.utils.tokenizer import CharTokenizer
from src.models.transformer import GPT
from src.utils.config import load_config

config = load_config("configs/train.yaml")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "data", "input.txt")

text = open(data_path).read()
tokenizer = CharTokenizer(text)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT(tokenizer.vocab_size, config).to(device)
model.load_state_dict(torch.load("checkpoints/model.pt"))
model.eval()

context = torch.zeros((1, 1), dtype=torch.long).to(device)

for _ in range(200):
    context_cond = context[:, -config["block_size"]:]
    logits = model(context_cond)
    logits = logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    context = torch.cat((context, next_token), dim=1)

print(tokenizer.decode(context[0].tolist()))
