import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Callable

import torch
from tqdm import tqdm

from src.trainer import Trainer, TrainerConfig
from src.data import LMDataset, Tokenizer
from src.transformer import (
    TransformerConfig,
    Decoder,
    generate,
)
from src.utils import load_and_prepare_data, get_best_device

device = get_best_device()


@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    learning_rate: float = 5e-4
    batch_size: int = 16
    max_iters: int = 10000
    dec_n_layer: int = 4
    dec_n_embd: int = 64
    dec_n_head: int = 2
    log_interval: int = 100
    save_checkpoint: bool = True
    checkpoint_dir: str = "checkpoints"


def setup_logging() -> None:
    """Configure logging for training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
    )


def create_model(
    config: TrainingConfig, tokenizer: Tokenizer, data: List[str]
) -> torch.nn.Module:
    """Create and configure the model."""
    logger = logging.getLogger(__name__)

    # Configure decoder
    dec_config = TransformerConfig(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=max([len(tokenizer.encode(inp)) for inp in data]),
        n_layer=config.dec_n_layer,
        n_embd=config.dec_n_embd,
        n_head=config.dec_n_head,
    )

    model = Decoder(dec_config)
    logger.info(
        f"Created Decoder model with {config.dec_n_layer} layers, "
        f"{config.dec_n_embd} embedding dim, {config.dec_n_head} heads"
    )
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {param_count:,} parameters")
    return model


def save_checkpoint(
    model: torch.nn.Module,
    config: TrainingConfig,
    trainer: Trainer,
    checkpoint_dir: str,
) -> None:
    """Save model checkpoint."""
    Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_iter_{trainer.iter_num}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "iter_num": trainer.iter_num,
            "loss": trainer.loss.item() if hasattr(trainer, "loss") else None,
        },
        checkpoint_path,
    )

    logging.info(f"Checkpoint saved to {checkpoint_path}")


def create_log_function(tokenizer: Tokenizer, config: TrainingConfig) -> Callable:
    """Create the logging function for training progress."""
    logger = logging.getLogger(__name__)

    def log_training_progress(trainer):
        # Print to console (separate from TQDM)
        tqdm.write(
            f"iter {trainer.iter_num}: train loss {trainer.loss.item():.5f} "
            f"({trainer.iter_dt * 1000:.2f}ms/iter)"
        )

        # Generate sample output
        try:
            prefix = torch.tensor(
                [tokenizer.encode("translate this to piglatin[SEP]")],
                device=trainer.device,  # Fix: ensure tensor is on correct device
            )
            output = generate(trainer.model, prefix, 100, 0.1)
            sample_output = tokenizer.decode(output.cpu().numpy()[0])
            # Extract the generated part after [SEP] and stop at [END] if present
            if b"[SEP]" in sample_output:
                generated_part = sample_output.split(b"[SEP]", 1)[1]
                if b"[END]" in generated_part:
                    sample_output = generated_part.split(b"[END]")[0]
                else:
                    sample_output = generated_part
            else:
                sample_output = sample_output.split(b"[END]")[0]
            tqdm.write(f"Sample generation: {sample_output}")
        except Exception as e:
            tqdm.write(f"Failed to generate sample: {e}")

        # Save checkpoint periodically
        if (
            config.save_checkpoint
            and trainer.iter_num % (config.log_interval * 10) == 0
            and trainer.iter_num > 0
        ):
            try:
                save_checkpoint(trainer.model, config, trainer, config.checkpoint_dir)
                tqdm.write(f"Checkpoint saved at iteration {trainer.iter_num}")
            except Exception as e:
                tqdm.write(f"Failed to save checkpoint: {e}")

    return log_training_progress


def train_model(
    data: List[str],
    config: TrainingConfig,
    model: torch.nn.Module,
    tokenizer: Tokenizer,
) -> tuple:
    """Train the model with the given configuration."""
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

    # Training configuration
    train_config = TrainerConfig(
        learning_rate=config.learning_rate,
        max_iters=config.max_iters,
        batch_size=config.batch_size,
        num_workers=0,
        device=device,
        log_interval=config.log_interval,
    )

    # Create dataset
    train_ds = LMDataset("train", data, tokenizer)
    logger.info(f"Created training dataset with {len(train_ds)} samples")

    # Create logging function
    log_fn = create_log_function(tokenizer, config)

    # Create trainer with logging function
    trainer = Trainer(train_config, model, train_ds, log_fn=log_fn)

    # Start training
    trainer.run()

    logger.info("Training completed!")
    return model, trainer


def load_config_from_args() -> TrainingConfig:
    """Load configuration from command line arguments."""
    parser = argparse.ArgumentParser(description="Train transformer model")
    parser.add_argument(
        "--learning_rate", type=float, default=5e-4, help="Learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--max_iters", type=int, default=10000, help="Maximum iterations"
    )
    parser.add_argument("--dec_n_layer", type=int, default=4, help="Decoder layers")
    parser.add_argument(
        "--dec_n_embd", type=int, default=64, help="Decoder embedding dimension"
    )
    parser.add_argument(
        "--dec_n_head", type=int, default=2, help="Decoder attention heads"
    )
    parser.add_argument(
        "--log_interval", type=int, default=100, help="Logging interval"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--no_checkpoint", action="store_true", help="Disable checkpoint saving"
    )

    args = parser.parse_args()

    return TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        dec_n_layer=args.dec_n_layer,
        dec_n_embd=args.dec_n_embd,
        dec_n_head=args.dec_n_head,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
        save_checkpoint=not args.no_checkpoint,
    )


def main():
    """Main training function."""
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting transformer training script")

    # Load configuration
    config = load_config_from_args()
    logger.info(f"Training configuration loaded")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Max iterations: {config.max_iters}")
    logger.info(f"Using device: {device}")

    try:
        # Load data
        data = load_and_prepare_data("train")

        # Create model and tokenizer
        tokenizer = Tokenizer()
        model = create_model(config, tokenizer, data)

        # Train model
        model, trainer = train_model(data, config, model, tokenizer)

        # Save final checkpoint
        if config.save_checkpoint:
            save_checkpoint(model, config, trainer, config.checkpoint_dir)
            logger.info("Final checkpoint saved")
            logger.info(
                f"To evaluate this model, run: python evaluate.py {config.checkpoint_dir}/checkpoint_iter_{trainer.iter_num}.pt"
            )

        logger.info("Training script completed successfully!")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
