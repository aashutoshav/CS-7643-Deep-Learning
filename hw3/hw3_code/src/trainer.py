import time
import logging

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from src.utils import get_best_device
from src.transformer import GenericTransformer
from dataclasses import dataclass
from typing import Tuple, Optional, Callable


@dataclass
class TrainerConfig:
    """Configuration for training."""

    learning_rate: float = 5e-4
    max_iters: int = 10000
    batch_size: int = 16
    device: str = "cpu"
    num_workers: int = 0
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_norm_clip: float = 1.0
    log_interval: int = 100


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        model: GenericTransformer,
        train_dataset,
        log_fn: Optional[Callable] = None,
    ):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.log_fn = log_fn  # Optional logging function
        self.logger = logging.getLogger(__name__)

        # determine the device we'll train on
        if config.device == "auto":
            self.device = get_best_device()
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.loss = None
        self.pbar = None

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = torch.optim.AdamW(
            model.configure_optimizers(config),
            lr=config.learning_rate,
            betas=config.betas,
        )

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        # Initialize progress bar
        self.pbar = tqdm(
            total=config.max_iters,
            desc="Training",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] loss: {postfix}",
        )

        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)

        try:
            while True:
                model.train()
                # fetch the next batch (x, y) and re-init iterator if needed
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)
                batch = [t.to(self.device) for t in batch]
                x, y = batch

                # forward the model
                logits, self.loss = model(x, targets=y)

                # backprop and update the parameters
                model.zero_grad(set_to_none=True)
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_norm_clip
                )
                self.optimizer.step()

                # Update timing and progress
                self.iter_num += 1
                tnow = time.time()
                self.iter_dt = tnow - self.iter_time
                self.iter_time = tnow

                # Update progress bar
                self.pbar.set_postfix_str(f"{self.loss.item():.4f}")
                self.pbar.update(1)

                # Call logging function if provided
                if self.log_fn and self.iter_num % config.log_interval == 0:
                    self.log_fn(self)

                # termination conditions
                if config.max_iters is not None and self.iter_num >= config.max_iters:
                    break
        finally:
            # Always close the progress bar
            if self.pbar:
                self.pbar.close()
