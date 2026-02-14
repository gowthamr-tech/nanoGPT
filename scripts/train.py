import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch
from src.utils.config import load_config
from src.utils.tokenizer import CharTokenizer
from src.models.transformer import GPT
from src.training.trainer import Trainer

config = load_config("configs/train.yaml")

text = open("data/input.txt").read()
tokenizer = CharTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT(tokenizer.vocab_size, config)

trainer = Trainer(model, data, config, device)
trainer.train()
