import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float, Int  # type: ignore


@dataclass
class TransformerConfig:
    """Configuration for transformer models."""

    vocab_size: int = 10
    block_size: int = 10
    n_layer: int = 2
    n_embd: int = 4
    n_head: int = 2
    hidden_pdrop: float = 0.1
    attn_pdrop: float = 0.1


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = nn.Embedding(config.block_size, config.n_embd)

    def forward(
        self, idx: Int[torch.Tensor, "batch seq_len"]
    ) -> Float[torch.Tensor, "batch seq_len n_embd"]:
        """
        Implement the embedding lookup. You can add the positional embeddings to the token embeddings.

        This will take a few lines.

        :param idx: Token indices of shape (batch, seq_len)
        :returns embeddings: Token + positional embeddings of shape (batch, seq_len, n_embd)
        """
        B, T = idx.size()
        embeddings = None
        ### TODO: BEGIN SOLUTION ###
        tok_emb = self.vocab_embeddings(idx) # (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)
        pos_emb = self.position_embeddings(pos) # (T, n_embd)
        embeddings = tok_emb + pos_emb # (B, T, n_embd)
        ### END SOLUTION ###
        return embeddings


class GenericSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        #
        assert config.n_embd % config.n_head == 0

        # Note: These could be a single batched linear layer
        # but we separate them for simplicity of implementation.
        self.k = nn.Linear(config.n_embd, config.n_embd)
        self.v = nn.Linear(config.n_embd, config.n_embd)
        self.q = nn.Linear(config.n_embd, config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.hidden_dropout = nn.Dropout(config.hidden_pdrop)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq_len n_embd"],
        attention_mask: Int[torch.Tensor, "batch 1 seq_len seq_len"],
    ) -> Float[torch.Tensor, "batch seq_len n_embd"]:
        """
        Implement multi-headed self-attention in GPT-2 Style.

        Use the provided layers initialized in __init__.
        Apply attention dropout after the softmax and hidden dropout after the projection.

        Reference: https://jalammar.github.io/illustrated-gpt2/#part-2-illustrated-self-attention

        Note: Attention heads should be computed in parallel.

        For each item in the batch, if attention_mask[b, 0, i, j] = 0,
        then you should manually set the attention from token i to j to be -inf.
        Hint: See torch.masked_fill

        :param x: Input embeddings of shape (batch, seq_len, n_embd)
        :param attention_mask: Attention mask of shape (batch, 1, seq_len, seq_len)
                              The second dimension is 1 for broadcasting across attention heads
        :returns y: Output embeddings of shape (batch, seq_len, n_embd)
        """
        B, T, C = (
            x.size()
        )  
        y = None

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if attention_mask is not None:
            if attention_mask.dim() == 3:
                mask = attention_mask.unsqueeze(1)
            else:
                mask = attention_mask
            att = att.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.hidden_dropout(self.c_proj(y))
        ### END SOLUTION ###

        return y


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = GenericSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(config.hidden_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq_len n_embd"],
        attention_mask: Int[torch.Tensor, "batch 1 seq_len seq_len"],
    ) -> Float[torch.Tensor, "batch seq_len n_embd"]:
        """
        Apply one transformer block using Pre-LayerNorm residual connections.

        You can refer to section 2.3 of the original GPT-2 paper for the implementation details: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
        If you are unsure, you can also refer to the code release: https://github.com/openai/gpt-2

        This should be 2-3 lines.

        :param x: Input embeddings of shape (batch, seq_len, n_embd)
        :param attention_mask: Attention mask of shape (batch, 1, seq_len, seq_len)
        :returns x: Output embeddings of shape (batch, seq_len, n_embd)
        """
        ### TODO: BEGIN SOLUTION ###
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)        
        y = x + self.mlpf(self.ln_2(x))
        ### END SOLUTION ###
        return y


class GenericTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        self.transformer = nn.ModuleDict(
            dict(
                embedding=Embedding(config),
                h=nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters (note we only count transformer parameters, not lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the optimizer parameter groups for PyTorch optimizer initialization.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, (
            "parameters %s made it into both decay/no_decay sets!"
            % (str(inter_params),)
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        return optim_groups

    def get_attention_mask(
        self, num_tokens: Int[torch.Tensor, "batch"]
    ) -> Int[torch.Tensor, "batch 1 max_tokens max_tokens"]:
        """
        Base implementation - subclasses will override this for specific attention patterns.

        :param num_tokens: Number of tokens per batch element of shape (batch,)
        :returns attention_mask: Attention mask of shape (batch, 1, max_tokens, max_tokens)
        """
        B = num_tokens.shape[0]
        max_tokens = min(self.block_size, num_tokens.max().item())
        return torch.ones((B, 1, max_tokens, max_tokens), dtype=torch.int)

    def forward(
        self,
        idx: Int[torch.Tensor, "batch seq_len"],
        targets: Optional[Int[torch.Tensor, "batch seq_len"]] = None,
        return_hidden: bool = False,
    ) -> Tuple[
        Float[torch.Tensor, "batch seq_len vocab_size"],
        Optional[Float[torch.Tensor, ""]],
    ]:
        """
        Put all the modules of a Transformer together for inference

        All the modules you'll need are defined in self.transformer
        - You can iterate through a nn.ModuleList using a standard for loop.
        - Make sure to apply layer normalization (ln_f) after the final transformer block and before the language modeling head.
        - the hidden state is the output of the final transformer block after layer normalization but before the language modeling head

        This will take a few lines!

        :param idx: Token indices of shape (batch, seq_len)
        :param targets: Target token indices of shape (batch, seq_len), optional
        :param return_hidden: Whether to return the hidden state
        :returns logits: Output logits of shape (batch, seq_len, vocab_size)
        :returns loss: Cross-entropy loss (scalar) or None if targets not provided
        :returns hidden: Hidden state of shape (batch, seq_len, n_embd) if return_hidden is True, otherwise None
        """

        num_tokens = (idx != -1).type(torch.int).sum(dim=1)
        idx = idx.masked_fill(idx == -1, 0).type(torch.int)[
            :, : num_tokens.max().item()
        ]
        if targets is not None:
            targets = targets[:, : num_tokens.max().item()]
        attention_mask = self.get_attention_mask(num_tokens)
        ### TODO: BEGIN SOLUTION ###
        x = self.transformer.embedding(idx)
        
        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask)
            
        hidden = self.transformer.ln_f(x)
        
        logits = self.lm_head(hidden)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.reshape(-1), 
                ignore_index=-1
            )
        ### END SOLUTION ###
        if return_hidden:
            return hidden

        return logits, loss


class Encoder(GenericTransformer):
    """Encoder Style Transformer with Bidirectional Attention"""

    def get_attention_mask(
        self, num_tokens: Int[torch.Tensor, "batch"]
    ) -> Int[torch.Tensor, "batch 1 max_tokens max_tokens"]:
        """
        Implement a padding mask function.

        This allows batching sequences of different lengths.

        For example, for any row attention_mask[b, i] the following should be true:
                      For j < num_tokens[b], attention_mask[b, i, j] = 1
                      For j >= num_tokens[b],  attention_mask[b, i, j] = 0

        Reference: https://huggingface.co/docs/transformers/glossary#attention-mask

        This should be a 1-3 line function.

        :param num_tokens: Number of tokens per batch element of shape (batch,)
        :returns attention_mask: Padding mask of shape (batch, 1, max_tokens, max_tokens)
        """
        B = num_tokens.shape[0]
        max_tokens = num_tokens.max().item()
        ### TODO: BEGIN SOLUTION ###
        pos = torch.arange(max_tokens, device=num_tokens.device).unsqueeze(0) # (1, max_tokens)
        mask = (pos < num_tokens.unsqueeze(1)).type(torch.int) # (B, max_tokens)
        attention_mask = mask.unsqueeze(1).unsqueeze(2).expand(B, 1, max_tokens, max_tokens)
        
        ### END SOLUTION ###
        return attention_mask.reshape(B, 1, max_tokens, max_tokens)


class Decoder(Encoder):
    """Decoder Style model with a Causal Attention Mask"""

    def get_attention_mask(
        self, num_tokens: Int[torch.Tensor, "batch"]
    ) -> Int[torch.Tensor, "batch 1 max_tokens max_tokens"]:
        """
        Modify the output of the full encoder mask to create a "causal" mask
        such that tokens only attend to tokens which occured earlier in the input.

        For example, for any row attention_mask[b, i] the following should be true:
                      For j <= i, attention_mask[b, i, j] = 1
                      For j > i,  attention_mask[b, i, j] = 0

        This should be a one line function which modifies the full attention_mask.

        :param num_tokens: Number of tokens per batch element of shape (batch,)
        :returns attention_mask: Causal mask of shape (batch, 1, max_tokens, max_tokens)
        """
        full_attention_mask = super().get_attention_mask(num_tokens)
        ### TODO: BEGIN SOLUTION ###
        max_tokens = full_attention_mask.shape[-1]
        device = full_attention_mask.device
        causal_mask = torch.tril(torch.ones((max_tokens, max_tokens), device=device, dtype=torch.int))
        attention_mask = full_attention_mask * causal_mask.unsqueeze(0).unsqueeze(1)
        ### END SOLUTION ###
        return attention_mask


@torch.no_grad()
def generate(
    model: GenericTransformer,
    idx: Int[torch.Tensor, "batch seq_len"],
    max_new_tokens: int,
    temperature: float = 1.0,
) -> Int[torch.Tensor, "batch seq_len_plus_new"]:
    """
    Sample from your model max_new_tokens times using torch.multinomial.

    You should feed the predictions back into the model each time.

    Adjust the probability distribution to be more or less greedy using
    the temperature parameter.

    NOTE: torch.multinomial may not work on MPS devices. You will need to move the probabilities to the CPU for sampling.

    Reference: https://huggingface.co/blog/how-to-generate#sampling
    Temperature Reference: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture10-nlg.pdf

    :param model: The transformer model to generate from
    :param idx: Input token indices of shape (batch, seq_len)
    :param max_new_tokens: Number of new tokens to generate
    :param temperature: Sampling temperature (must be > 0)
    :returns idx: Extended token indices of shape (batch, seq_len + max_new_tokens)
    """
    assert temperature > 0, "Temperature must be positive"
    model.eval()
    ### TODO: BEGIN SOLUTION ###
    block_size = getattr(model, 'block_size', None)
    if block_size is None and hasattr(model, 'config'):
        block_size = getattr(model.config, 'block_size', None)

    for _ in range(max_new_tokens):
        if block_size is not None:
            idx_cond = idx[:, -block_size:]
        else:
            idx_cond = idx

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    ### END SOLUTION ###
    return idx
