"""The GPT: embeddings, causal self-attention, transformer blocks, training and generation.

Needs PyTorch (CPU is enough). The tokenizer and corpus code live in core.py.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64      # how many tokens of context the model can see
    n_layer: int = 4          # number of transformer blocks
    n_head: int = 4           # attention heads per block
    n_embd: int = 128         # size of each token's embedding vector
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Each token looks at itself and earlier tokens, never at later ones."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)     # queries, keys and values at once
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))        # how relevant is each token
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))   # hide the future
        weights = self.drop(F.softmax(scores, dim=-1))
        y = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class MLP(nn.Module):
    """The feed-forward layer that follows attention in each block."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    """Transformer block: attention + feed-forward, each with layer norm and a residual connection."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)     # token id -> vector
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)     # position -> vector
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)   # vector -> score per token
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def get_batch(data, block_size, batch_size, generator=None):
    """Random (input, target) batch; the target is the input shifted by one token."""
    if len(data) <= block_size + 1:
        raise ValueError(
            f"not enough text for a block size of {block_size}: add more text or lower --block-size"
        )
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, data, batch_size=32, batches=10, seed=0):
    """Average loss on fixed random batches (the same ones every time)."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(data, model.cfg.block_size, batch_size, generator)
        _, loss = model(x, y)
        total += loss.item()
    return total / batches


def lr_at(step, total_steps, base_lr, warmup=50, floor=0.1):
    """Learning rate: short linear warm-up, then cosine decay down to `floor` x base."""
    warmup = max(1, min(warmup, total_steps // 10))
    if step <= warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def train(model, train_data, val_data, steps, batch_size, lr, eval_every=100, seed=1337, log=print):
    """Train with AdamW and cross-entropy loss. Keeps the weights with the best validation loss.

    Small corpora are quickly memorised: validation loss stops improving and
    then rises. Restoring the best weights at the end avoids keeping an
    over-trained model.
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_val, best_state, history = float("inf"), None, []
    for step in range(1, steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, steps, lr)
        model.train()
        x, y = get_batch(train_data, model.cfg.block_size, batch_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            val = estimate_loss(model, val_data)
            history.append({"step": step, "train": loss.item(), "val": val})
            if val < best_val:
                best_val, best_state = val, copy.deepcopy(model.state_dict())
            log(f"step {step:>5}/{steps}  train {loss.item():.3f}  val {val:.3f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"best_val_loss": best_val, "history": history}


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

@torch.no_grad()
def generate_ids(model, prompt_ids, max_new_tokens, temperature=0.8, top_k=None, seed=None):
    """Sample `max_new_tokens` tokens after the prompt.

    Returns (new_ids, probs). probs[i] is the probability the model itself gave
    to new_ids[i] (at temperature 1), so it shows how confident the model was.

    temperature: below 1 is more conservative, above 1 more random, 0 always
    picks the single most likely token. top_k: only sample from the k most
    likely tokens.
    """
    if not prompt_ids:
        raise ValueError("the starting phrase is empty")
    if max_new_tokens < 0:
        raise ValueError("length cannot be negative")
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    if top_k is not None and top_k < 1:
        raise ValueError("top-k must be at least 1")

    model.eval()
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    idx = torch.tensor([list(prompt_ids)], dtype=torch.long)
    new_ids, probs = [], []
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.cfg.block_size :])
        logits = logits[:, -1, :]
        model_probs = F.softmax(logits, dim=-1)
        if temperature == 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            scaled = logits / temperature
            if top_k is not None:
                values, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                scaled = scaled.masked_fill(scaled < values[:, [-1]], float("-inf"))
            next_id = torch.multinomial(F.softmax(scaled, dim=-1), num_samples=1, generator=generator)
        token = int(next_id.item())
        new_ids.append(token)
        probs.append(float(model_probs[0, token]))
        idx = torch.cat([idx, next_id], dim=1)
    return new_ids, probs


@dataclass
class Completion:
    prompt: str
    text: str                              # the generated continuation only
    tokens: list[tuple[str, float]]        # (token text, probability the model gave it)


def complete(model, tokenizer, prompt, max_new_tokens=40, temperature=0.8, top_k=40, seed=None) -> Completion:
    """Complete a partial medical phrase or definition."""
    # Words in the training text carry their leading space, so a trailing space
    # typed by the user is dropped and the model supplies it with the next word.
    prompt = prompt.rstrip()
    new_ids, probs = generate_ids(model, tokenizer.encode(prompt), max_new_tokens, temperature, top_k, seed)
    pieces = [tokenizer.token_text(i) for i in new_ids]
    return Completion(prompt, tokenizer.decode(new_ids), list(zip(pieces, probs)))
