"""
Comprehensive PyTorch Trainer for QueenNet.
Supports Adam optimizer, early stopping, CSV logging, checkpoint saving and resuming.
"""
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional

from queennet.utils import ensure_dir, setup_logger
from .early_stopping import EarlyStopping
from .losses import get_loss_fn

class Trainer:
    """
    Manages the training, validation, and checkpointing lifecycle of QueenNet.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        output_dir: str | Path = "outputs",
        device: torch.device | None = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        
        # Output subdirectories
        self.ckpt_dir = ensure_dir(self.output_dir / "checkpoints")
        self.log_dir = ensure_dir(self.output_dir / "logs")
        self.metrics_dir = ensure_dir(self.output_dir / "metrics")
        
        self.logger = setup_logger("trainer", self.log_dir / "training.log")
        
        # Training hyperparameters
        t_cfg = config.get("training", {})
        self.lr = t_cfg.get("learning_rate", 0.001)
        self.weight_decay = t_cfg.get("weight_decay", 1e-4)
        self.max_epochs = t_cfg.get("max_epochs", 300)
        self.patience = t_cfg.get("patience", 25)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.criterion = get_loss_fn(t_cfg.get("loss", "cross_entropy"))
        self.early_stopping = EarlyStopping(patience=self.patience, mode="min")
        
        self.history = []

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        """Train for a single epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(data)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * len(targets)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total_samples += len(targets)
            
        epoch_loss = total_loss / max(1, total_samples)
        epoch_acc = correct / max(1, total_samples)
        return epoch_loss, epoch_acc

    def validate(self) -> tuple[float, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(data)
                loss = self.criterion(logits, targets)
                
                total_loss += loss.item() * len(targets)
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total_samples += len(targets)
                
        val_loss = total_loss / max(1, total_samples)
        val_acc = correct / max(1, total_samples)
        return val_loss, val_acc

    def save_checkpoint(self, path: Path, epoch: int, val_loss: float, val_acc: float):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "config": self.config
        }, str(path))

    def load_checkpoint(self, path: Path):
        """Load weights from checkpoint."""
        ckpt = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt

    def fit(self) -> pd.DataFrame:
        """Run full training routine."""
        self.logger.info(f"Starting QueenNet Training for up to {self.max_epochs} epochs...")
        best_ckpt_path = self.ckpt_dir / "best_model.pt"
        
        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self.train_epoch(epoch)
            va_loss, va_acc = self.validate()
            elapsed = time.time() - t0
            
            record = {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "lr": self.optimizer.param_groups[0]["lr"],
                "time_sec": elapsed
            }
            self.history.append(record)
            
            self.logger.info(
                f"Epoch [{epoch:03d}/{self.max_epochs:03d}] "
                f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
                f"Val Loss: {va_loss:.4f} Acc: {va_acc*100:.2f}% | "
                f"Time: {elapsed:.2f}s"
            )
            
            if self.early_stopping.step(va_loss):
                self.logger.info(f"--> Saved best model checkpoint to {best_ckpt_path}")
                self.save_checkpoint(best_ckpt_path, epoch, va_loss, va_acc)
                
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch}.")
                break
                
        # Save training history to CSV
        df = pd.DataFrame(self.history)
        csv_path = self.metrics_dir / "training_history.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved training history to {csv_path}")
        
        # Load best weights before returning
        if best_ckpt_path.exists():
            self.load_checkpoint(best_ckpt_path)
            
        return df
