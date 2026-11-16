"""File I/O: .env, corpus download and loading, tokenizer and model checkpoints.

PyTorch is only imported inside the checkpoint functions, so the corpus and
tokenizer commands work without it.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from . import core

# "Structure and Functions of the Body", a hand-book of anatomy and physiology,
# Project Gutenberg eBook #64754 (listed there as public domain in the USA).
DEFAULT_SOURCE_NAME = "structure-and-functions-of-the-body"
DEFAULT_SOURCE_URL = "https://www.gutenberg.org/ebooks/64754.txt.utf-8"
USER_AGENT = "medical-terminology-helper/1.0 (student project)"


def load_env(path: str = ".env") -> None:
    """Read KEY=VALUE lines from a .env file into os.environ (existing vars win)."""
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------

def download_text(url: str, timeout: int = 30) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8-sig", errors="replace")


def fetch_corpus(url: str, name: str, data_dir: str, downloader=None):
    """Download a plain-text book, clean it, and save it as data_dir/<name>.txt."""
    downloader = downloader or download_text
    cleaned = core.prepare_corpus(downloader(url))
    if not cleaned:
        raise ValueError("the download contained no usable text")
    path = Path(data_dir) / f"{name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cleaned, encoding="utf-8")
    return path, len(cleaned)


def list_corpus_files(source: str) -> list[Path]:
    """`source` is one .txt file, or a folder whose .txt files are all used."""
    path = Path(source)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.txt"))
    raise FileNotFoundError(f"corpus not found: {source}")


def load_corpus(source: str, max_chars: int | None = None):
    """Read, clean and join the corpus. Returns (text, list of file names)."""
    files = list_corpus_files(source)
    if not files:
        raise FileNotFoundError(
            f"no .txt files in {source} - add some, or run `python -m app.cli fetch`"
        )
    parts = [core.prepare_corpus(f.read_text(encoding="utf-8")) for f in files]
    text = "\n\n".join(part.strip() for part in parts if part.strip()) + "\n"
    if not text.strip():
        raise ValueError(f"the corpus in {source} has no usable text")
    return core.limit_chars(text, max_chars), [f.name for f in files]


# --------------------------------------------------------------------------
# Tokenizer and model files
# --------------------------------------------------------------------------

def save_tokenizer(path: str, tokenizer: core.BPETokenizer) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(tokenizer.to_dict()), encoding="utf-8")


def load_tokenizer(path: str) -> core.BPETokenizer:
    if not Path(path).exists():
        raise FileNotFoundError(f"no tokenizer at {path} - run `python -m app.cli train` first")
    return core.BPETokenizer.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_checkpoint(path: str, model, tokenizer: core.BPETokenizer, meta: dict | None = None) -> None:
    import torch

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.cfg.to_dict(),
            "tokenizer": tokenizer.to_dict(),
            "state_dict": model.state_dict(),
            "meta": meta or {},
        },
        path,
    )


def load_checkpoint(path: str):
    """Return (model, tokenizer) from a checkpoint written by save_checkpoint."""
    if not Path(path).exists():
        raise FileNotFoundError(f"no model at {path} - run `python -m app.cli train` first")
    import torch

    from .llm import GPT, GPTConfig

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, core.BPETokenizer.from_dict(checkpoint["tokenizer"])
