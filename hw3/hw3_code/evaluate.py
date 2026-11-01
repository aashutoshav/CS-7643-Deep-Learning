import argparse
import logging
from pathlib import Path
from typing import Dict, List

import torch
import tqdm
from sacrebleu.metrics import BLEU

from src.transformer import Decoder, generate, TransformerConfig
from src.utils import load_and_prepare_data, get_best_device
from src.data import Tokenizer
from train import TrainingConfig  # needed to unpickle checkpoints

device = get_best_device()


def load_checkpoint(checkpoint_path: str) -> tuple:
    """Load model checkpoint and configuration."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    iter_num = checkpoint.get("iter_num", "unknown")
    loss = checkpoint.get("loss", "unknown")

    logger.info(f"Loaded checkpoint from iteration {iter_num} with loss {loss}")
    return checkpoint, config


def create_model_from_checkpoint(
    checkpoint: dict, tokenizer: Tokenizer, data: List[str]
) -> torch.nn.Module:
    """Create and load model from checkpoint."""
    logger = logging.getLogger(__name__)
    config = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]

    pos_emb_key = "transformer.embedding.position_embeddings.weight"
    if pos_emb_key in state_dict:
        block_size = state_dict[pos_emb_key].shape[0]
    else:
        block_size = max([len(tokenizer.encode(inp)) for inp in data])

    # Configure decoder
    dec_config = TransformerConfig(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=block_size,
        n_layer=config.dec_n_layer,
        n_embd=config.dec_n_embd,
        n_head=config.dec_n_head,
    )

    model = Decoder(dec_config)
    logger.info(
        f"Created Decoder model with {config.dec_n_layer} layers, "
        f"{config.dec_n_embd} embedding dim, {config.dec_n_head} heads"
    )

    # Load model state
    model.load_state_dict(state_dict)
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded with {param_count:,} parameters")
    return model


def evaluate_model(
    model: torch.nn.Module,
    data: List[str],
    tokenizer: Tokenizer,
    temperature: float = 0.1,
) -> Dict[str, float]:
    """Evaluate the model on the dataset."""
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation...")

    model.eval()
    bleu = BLEU()
    results = []
    tgts = []
    cands = []

    logger.info(f"Evaluating on {len(data)} samples")

    with torch.no_grad():
        for sent in tqdm.tqdm(data, desc="Evaluating"):
            try:
                inp = torch.tensor(
                    [tokenizer.encode(sent.split("[SEP]")[0] + "[SEP]")]
                ).to(device)
                tgt = sent.split("[SEP]")[1].split("[END]")[0]

                cat = generate(model, inp, model.block_size - len(inp[0]), temperature)
                tgt_candidate = tokenizer.decode(cat.cpu().numpy()[0])
                # For Decoder models, we need to extract the part after [SEP] and before [END]
                if b"[SEP]" in tgt_candidate:
                    # Split at [SEP] and take the generated part
                    generated_part = tgt_candidate.split(b"[SEP]", 1)[1]
                    # If there's an [END] token, stop there
                    if b"[END]" in generated_part:
                        tgt_candidate = generated_part.split(b"[END]")[0]
                    else:
                        tgt_candidate = generated_part
                else:
                    # Fallback: just stop at [END] if no [SEP] found
                    tgt_candidate = tgt_candidate.split(b"[END]")[0]

                tgt_candidate_str = tgt_candidate.decode("utf-8", errors="replace")
                tgts.append(tgt)
                cands.append(tgt_candidate_str)
                correct = tgt == tgt_candidate_str
                results.append(correct)
            except Exception as e:
                logger.warning(f"Error evaluating sample: {e}")
                continue

    if not results:
        logger.error("No valid evaluation results")
        return {"exact_match": 0.0, "bleu_score": 0.0, "total_samples": 0}

    results_tensor = torch.tensor(results).type(torch.float)
    exact_match = torch.mean(results_tensor).item() * 100
    bleu_score = bleu.corpus_score(cands, [tgts]).score

    logger.info(
        f"Exact Match: {torch.sum(results_tensor)}/{len(results)} = {exact_match:.2f}%"
    )
    logger.info(f"BLEU Score: {bleu_score:.4f}")

    return {
        "exact_match": exact_match,
        "bleu_score": bleu_score,
        "total_samples": len(results),
    }


def generate_samples(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    num_samples: int = 5,
    temperature: float = 0.1,
) -> None:
    """Generate sample translations to inspect model quality."""
    logger = logging.getLogger(__name__)
    logger.info(f"Generating {num_samples} sample translations...")

    model.eval()
    sample_prompts = [
        "translate this to piglatin",
        "hello world",
        "the quick brown fox",
        "machine learning is fun",
        "transformer models are powerful",
    ]

    with torch.no_grad():
        for i, prompt in enumerate(sample_prompts[:num_samples]):
            try:
                prefix = torch.tensor([tokenizer.encode(f"{prompt}[SEP]")]).to(device)

                output = generate(model, prefix, 100, temperature)
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

                logger.info(
                    f"Sample {i+1}: {prompt} -> {sample_output.decode('utf-8', errors='replace')}"
                )
            except Exception as e:
                logger.warning(f"Failed to generate sample {i+1}: {e}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate transformer model from checkpoint"
    )
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file")
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Temperature for generation"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of sample generations to show",
    )
    parser.add_argument(
        "--no_samples", action="store_true", help="Skip sample generation"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting transformer evaluation script")
    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Using device: {device}")

    try:
        # Load checkpoint and data
        checkpoint, config = load_checkpoint(args.checkpoint_path)
        data = load_and_prepare_data("validation")

        # Create model and tokenizer
        tokenizer = Tokenizer()
        model = create_model_from_checkpoint(checkpoint, tokenizer, data)

        # Generate samples if requested
        if not args.no_samples:
            generate_samples(model, tokenizer, args.num_samples, args.temperature)

        # Evaluate model
        metrics = evaluate_model(
            model,
            data,
            tokenizer,
            args.temperature,
        )

        logger.info("=" * 50)
        logger.info("FINAL EVALUATION RESULTS")
        logger.info("=" * 50)
        logger.info(f"Exact Match Accuracy: {metrics['exact_match']:.2f}%")
        logger.info(f"BLEU Score: {metrics['bleu_score']:.4f}")
        logger.info(f"Total Samples Evaluated: {metrics['total_samples']}")
        logger.info("=" * 50)

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
