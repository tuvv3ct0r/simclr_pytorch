import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from model.model import SimCLR
from preprocessing import SimCLRDataModule
from utils.lars import LARS


class NTXentLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()

        self.temperature = temperature
        
    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]

        z = torch.cat([z_i, z_j], dim=0)
        z = z / torch.norm(z, dim=1, keepdim=True)

        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # [2B, 2B]
        sim /= self.temperature

        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels, labels], dim=0) # [2B]

        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, float('-inf'))  # remove self-similarity

        positives = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0)
        loss = -positives + torch.logsumexp(sim, dim=1)
        return loss.mean()


class Trainer:
    def __init__(self, config):
        # TODO: IMPLEMENT LR SCHEDULER
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = SimCLR(config).to(self.device)
        self.criterion = NTXentLoss(config["training"]["temperature"])

        if config["training"]["optimizer"] == "LARS":
            self.optimizer = LARS(
                self.model.parameters(),
                lr=config["training"]["learning_rate"],
                weight_decay=float(config["training"]["weight_decay"]),
                momentum=config["training"]["momentum"],
            )
        elif config["training"]["optimizer"] == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config["training"]["learning_rate"],
                weight_decay=float(config["training"]["weight_decay"])
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config['training']['optimizer']}")
        
        self.checkpoint_dir = config["checkpoint"]["save_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_interval = config["logging"]["log_interval"]

    def train(self, train_loader, val_loader=None):
        best_val_loss = float('inf')

        for epoch in range(self.config["training"]["epochs"]):
            self.model.train()
            running_loss = 0.0

            for batch_idx, (xi, xj) in enumerate(train_loader):
                xi, xj = xi.to(self.device), xj.to(self.device)
                zi = self.model(xi)
                zj = self.model(xj)
                loss = self.criterion(zi, zj)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                if (batch_idx + 1) % self.log_interval == 0:
                    print(f"Epoch {epoch+1} Batch {batch_idx+1}: Loss = {loss.item():.4f}")
            avg_train_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}: Avg Train Loss = {avg_train_loss:.4f}")

            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}")
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "best_model.pt"))
            # Checkpointing
            if (epoch + 1) % self.config["checkpoint"]["save_freq"] == 0:
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f"model_epoch_{epoch+1}.pt"))

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for xi, xj in val_loader:
                xi, xj = xi.to(self.device), xj.to(self.device)
                zi = self.model(xi)
                zj = self.model(xj)
                loss = self.criterion(zi, zj)
                total_loss += loss.item()
        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def test(self):
        pass
