import torch
import torch.nn.functional as F
from .dataset import get_batch

class Trainer:
    def __init__(self, model, data, config, device):
        self.model = model
        self.data = data
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    def train(self):
        self.model.to(self.device)

        for epoch in range(self.config["epochs"]):
            xb, yb = get_batch(
                self.data,
                self.config["block_size"],
                self.config["batch_size"]
            )

            xb, yb = xb.to(self.device), yb.to(self.device)

            logits = self.model(xb)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 50 == 0:
                print(f"Epoch {epoch} | Loss {loss.item()}")

        torch.save(self.model.state_dict(), "checkpoints/model.pt")
