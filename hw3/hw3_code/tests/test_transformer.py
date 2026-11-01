import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.transformer import (
    GenericTransformer,
    Embedding,
    GenericSelfAttention,
    TransformerBlock,
    Encoder,
    Decoder,
    generate,
)


def test_embedding(deterministic_setup, embedding_config):
    """Test embedding layer functionality."""
    embedding = Embedding(embedding_config)
    embedding.vocab_embeddings.weight = torch.nn.Parameter(
        torch.arange(0, 10, dtype=torch.float).reshape(10, 1)
    )
    embedding.position_embeddings.weight = torch.nn.Parameter(
        torch.arange(0, 10, dtype=torch.float).reshape(10, 1)
    )

    result = embedding(torch.tensor([[1, 2, 3]]))
    expected = torch.tensor([1, 3, 5], dtype=torch.float).reshape(1, 3, 1)
    assert torch.allclose(result, expected)


def test_mha(deterministic_setup, mha_config):
    """Test multi-head attention functionality."""
    attn = GenericSelfAttention(mha_config)
    # Set up specific weights for deterministic testing
    attn.q.weight = torch.nn.Parameter(torch.eye(2, 2).repeat(2, 2).flip(0))
    attn.q.bias = torch.nn.Parameter(torch.zeros(4))
    attn.k.weight = torch.nn.Parameter(torch.eye(2, 2).repeat(2, 2))
    attn.k.bias = torch.nn.Parameter(torch.zeros(4))
    attn.v.weight = torch.nn.Parameter(torch.eye(4, 4))
    attn.v.bias = torch.nn.Parameter(torch.zeros(4))
    attn.c_proj.weight = torch.nn.Parameter(torch.eye(4, 4))
    attn.c_proj.bias = torch.nn.Parameter(torch.zeros(4))

    embeddings = torch.tensor([[[1, 2, 3, 4], [4, 3, 2, 1]]], dtype=torch.float)
    mask = torch.ones(1, 1, 2, 2)

    result = attn(embeddings, mask)
    expected = torch.tensor(
        [[0.0000, 0.1654, 3.0456, 1.7295], [0.3307, 0.2480, 0.0000, 0.0000]],
        dtype=torch.float,
    )
    assert torch.allclose(result, expected, atol=1e-3, rtol=1)


def test_mha_mask(deterministic_setup, mha_config_no_dropout):
    """Test multi-head attention with masking."""
    attn = GenericSelfAttention(mha_config_no_dropout)

    # Set up specific weights for deterministic testing
    attn.v.weight = torch.nn.Parameter(torch.eye(4, 4))
    attn.v.bias = torch.nn.Parameter(torch.zeros(4))
    attn.c_proj.weight = torch.nn.Parameter(torch.eye(4, 4))
    attn.c_proj.bias = torch.nn.Parameter(torch.zeros(4))

    embeddings = torch.tensor([[[1, 2, 3, 4], [4, 3, 2, 1]]], dtype=torch.float)
    mask = torch.zeros(1, 1, 2, 2)
    mask[0, 0, 0, 0] = 1
    mask[0, 0, 1, 1] = 1

    result = attn(embeddings, mask)
    expected = torch.tensor([[[1, 2, 3, 4], [4, 3, 2, 1]]], dtype=torch.float)
    assert torch.allclose(result, expected, atol=1e-4)


def test_transformer_block(deterministic_setup, mha_config_no_dropout):
    """Test transformer block applies Pre-LN attention and MLP with residual connections."""
    block = TransformerBlock(mha_config_no_dropout)
    block.eval()

    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]])
    mask = torch.ones(1, 1, 2, 2)

    result = block(x, mask)
    expected = torch.tensor(
        [[[0.6172, 2.2894, 3.3411, 4.0958], [3.5078, 3.1687, 2.4068, 0.8504]]]
    )
    assert torch.allclose(result, expected, atol=1e-4)


def test_transformer(deterministic_setup, transformer_config):
    """Test full transformer functionality."""
    transformer = GenericTransformer(transformer_config)
    idx = torch.tensor([[1, 2, 3, 4, 5, -1, -1, -1, -1, -1]], dtype=torch.long)

    logits = transformer(idx)[0]
    s = F.softmax(logits[0, 0], dim=0)
    expected = torch.tensor(
        [
            0.1034,
            0.0960,
            0.1019,
            0.1022,
            0.1003,
            0.1040,
            0.0983,
            0.1072,
            0.0958,
            0.0910,
        ],
        dtype=torch.float,
    )
    assert torch.allclose(s, expected, atol=1e-5, rtol=1)


def test_transformer_loss(deterministic_setup, transformer_config):
    """Test transformer loss computation."""
    transformer = GenericTransformer(transformer_config)
    idx = torch.tensor([[1, 2, 3, 4, 5, -1, -1, -1, -1, -1]], dtype=torch.long)
    target = torch.arange(5).reshape(1, 5)

    _, loss = transformer(idx, targets=target)
    expected_loss = torch.tensor(2.2970)
    assert torch.allclose(loss, expected_loss, atol=1e-4)


def test_transformer_hidden(deterministic_setup, transformer_config):
    """Test transformer hidden state output."""
    transformer = GenericTransformer(transformer_config)
    idx = torch.tensor([[1, 2, 3, 4, 5, -1, -1, -1, -1, -1]], dtype=torch.long)
    target = torch.arange(5).reshape(1, 5)

    hidden = transformer(idx, targets=target, return_hidden=True)
    expected_hidden = torch.tensor([1.4134, -1.3609, 0.1767, -0.2292])
    assert torch.allclose(hidden[0, 0], expected_hidden, atol=1e-4)


def test_encoder(deterministic_setup, transformer_config):
    """Test encoder attention mask generation."""
    encoder = Encoder(transformer_config)
    mask = encoder.get_attention_mask(torch.tensor([5, 6]))

    assert mask.shape == (2, 1, 6, 6)
    mask_1 = [
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 0],
    ]
    mask_2 = [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ]
    expected = torch.tensor([[mask_1], [mask_2]], dtype=torch.int)
    assert torch.allclose(mask, expected)


def test_decoder(deterministic_setup, transformer_config):
    """Test decoder attention mask generation."""
    decoder = Decoder(transformer_config)
    mask = decoder.get_attention_mask(torch.tensor([5, 6]))

    assert mask.shape == (2, 1, 6, 6)

    mask_1 = [
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 0],
    ]
    mask_2 = [
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
    ]
    expected = torch.tensor([[mask_1], [mask_2]], dtype=torch.int)
    assert torch.allclose(mask, expected)


class DumbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.zeros(1, 1, 10)
        self.l[0, 0, 0] = 100
        self.l[0, 0, 5] = 90

    def forward(self, idx):
        return self.l.roll(idx[0, -1].item() + 1), None


def test_generate(deterministic_setup):
    dumb_model = DumbModel()
    assert torch.allclose(
        generate(dumb_model, torch.tensor([[0]]), 6),
        torch.tensor([0, 1, 2, 3, 4, 5, 6]),
    )
    temp_gen_1 = generate(dumb_model, torch.tensor([[0]]), 6, temperature=10)
    assert torch.allclose(temp_gen_1, torch.tensor([[0, 1, 7, 8, 4, 0, 1]]))
