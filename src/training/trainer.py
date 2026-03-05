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

    @torch.no_grad()
    def estimate_val_loss(self, val_data, eval_steps=20):
        self.model.eval()
        losses = []
        for _ in range(eval_steps):
            xb, yb = get_batch(val_data, self.config["block_size"], self.config["batch_size"])
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits = self.model(xb)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)

    def train(self):
        self.model.to(self.device)

        split = int(0.9 * len(self.data))
        train_data = self.data[:split]
        val_data = self.data[split:]

        steps_per_epoch = max(1, len(train_data) // (self.config["block_size"] * self.config["batch_size"]))

        for epoch in range(self.config["epochs"]):
            self.model.train()
            for step in range(steps_per_epoch):
                xb, yb = get_batch(
                    train_data,
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

            val_loss = self.estimate_val_loss(val_data)
            print(f"Epoch {epoch} | Val Loss {val_loss:.4f}")
            if val_loss < 1.5:
                print("Val loss below 1.5 — stopping early.")
                break

        torch.save(self.model.state_dict(), "checkpoints/model.pt")
