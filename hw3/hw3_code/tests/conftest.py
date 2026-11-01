import pytest
import torch
from src.utils import setup_deterministic
from src.transformer import TransformerConfig


@pytest.fixture(scope="function")
def deterministic_setup():
    """
    Set up deterministic environment for each test function.
    Forces CPU usage for consistent test results across different hardware.
    """
    # Force CPU for tests to ensure consistent results
    device = setup_deterministic(seed=3407, device="cpu")
    yield device
    # Cleanup: reset to default behavior after test
    torch.set_default_device("cpu")


@pytest.fixture
def transformer_config():
    """Default transformer configuration for tests."""
    return TransformerConfig(
        vocab_size=10, block_size=10, n_layer=2, n_embd=4, n_head=2
    )


@pytest.fixture
def embedding_config():
    """Configuration for embedding tests."""
    return TransformerConfig(
        vocab_size=10, block_size=10, n_layer=1, n_embd=1, n_head=1
    )


@pytest.fixture
def mha_config():
    """Configuration for multi-head attention tests."""
    return TransformerConfig(
        vocab_size=10,
        block_size=10,
        n_layer=1,
        n_embd=4,
        n_head=2,
        hidden_pdrop=0.25,
        attn_pdrop=0.1,
    )


@pytest.fixture
def mha_config_no_dropout():
    """Configuration for multi-head attention tests without dropout."""
    return TransformerConfig(
        vocab_size=10,
        block_size=10,
        n_layer=1,
        n_embd=4,
        n_head=2,
        hidden_pdrop=0.0,
        attn_pdrop=0.0,
    )
