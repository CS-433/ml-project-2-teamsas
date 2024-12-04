from typing import Dict, List, Optional
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer(object):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        early_stopping_monitor: Optional[str],
        early_stopping_patience: Optional[int],
        best_model_path: Optional[Path],
        checkpoint_path: Optional[Path],
        checkpoint_frequency: Optional[int],
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.num_epochs = num_epochs

        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_patience = early_stopping_patience

        self.best_model_path = best_model_path

        self.checkpoint_path = checkpoint_path
        self.checkpoint_frequency = checkpoint_frequency

        self.device = device

    def _train_epoch(self):
        model = self.model.train()
        for data, target in tqdm(self.train_loader, leave=False):
            data, target = data, target.to(self.device)
            self.optimizer.zero_grad()
            output = model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

    def _validate_epoch(self, data_loader) -> Dict[str, float]:
        model = self.model.eval()
        sum_abs_error = 0
        sum_squared_error = 0
        num_samples = 0
        with torch.no_grad():
            for data, target in tqdm(data_loader, leave=False):
                data, target = data, target.to(self.device)
                output = model(data)
                sum_abs_error += torch.sum(torch.abs(output - target)).item()
                sum_squared_error += torch.sum((output - target) ** 2).item()
                num_samples += len(target)
        return {
            "mae": sum_abs_error / num_samples,
            "mse": sum_squared_error / num_samples,
            "rmse": (sum_squared_error / num_samples) ** 0.5,
        }

    def _compile_hist(self, history) -> Dict[str, List[float]]:
        return {
            metric: [measurement[metric] for measurement in history]
            for metric in history[0].keys()
        }

    def _save_model(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)

    def _load_model(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path))

    def train(self):
        train_hist = []
        val_hist = []

        best_measurement = float("inf")
        best_epoch = 0

        with tqdm(range(self.num_epochs)) as pbar:
            for epoch in pbar:
                self._train_epoch()
                train_metrics = self._validate_epoch(self.train_loader)
                val_metrics = self._validate_epoch(self.val_loader)
                train_hist.append(train_metrics)
                val_hist.append(val_metrics)
                pbar.set_postfix(
                    {
                        "train_mae": train_metrics["mae"],
                        "val_mae": val_metrics["mae"],
                    }
                )
                measurement = val_metrics[self.early_stopping_monitor]
                if measurement < best_measurement:
                    self._save_model(self.best_model_path)
                    best_measurement = measurement
                    best_epoch = epoch
                if epoch - best_epoch > self.early_stopping_patience:
                    break
                if self.checkpoint_path is not None:
                    if epoch % self.checkpoint_frequency == 0:
                        self._save_model(self.checkpoint_path)
        self._load_model(self.best_model_path)
        return self._compile_hist(train_hist), self._compile_hist(val_hist)
