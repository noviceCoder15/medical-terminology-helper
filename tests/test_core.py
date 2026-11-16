import random
from collections import Counter

import pytest

from app import core

GUTENBERG = """The Project Gutenberg eBook of Example

Some licence text that must not be learned.

*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***

CHAPTER I

The heart is a muscular organ that
pumps blood through the _vessels_ of the
body.[Pg 12]

[Illustration: Fig. 1. The heart.]

12

The lungs lie on either side of the heart.

*** END OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***

More licence text that must not be learned.
"""


# ---- milestone 1: corpus preparation ---------------------------------------

def test_strip_gutenberg_removes_header_and_footer():
    text = core.strip_gutenberg(GUTENBERG)
    assert "licence" not in text
    assert "The heart is a muscular organ" in text


def test_prepare_corpus_reflows_cleans_and_drops_noise():
    cleaned = core.prepare_corpus(GUTENBERG)
    assert "The heart is a muscular organ that pumps blood through the vessels of the body." in cleaned
    assert "The lungs lie on either side of the heart." in cleaned
    for unwanted in ("Illustration", "Pg 12", "_", "licence", "\n12\n"):
        assert unwanted not in cleaned
    assert all("\n" not in paragraph for paragraph in cleaned.strip().split("\n\n"))


def test_prepare_corpus_is_idempotent():
    once = core.prepare_corpus(GUTENBERG)
    assert core.prepare_corpus(once) == once


def test_prepare_corpus_of_nothing_is_empty():
    assert core.prepare_corpus("  \n\n \n") == ""


def test_prepare_corpus_normalises_punctuation():
    assert core.prepare_corpus("It\u2019s a \u201ctest\u201d \u2014 really fine.") == 'It\'s a "test" - really fine.\n'


def test_limit_chars_cuts_on_a_paragraph_boundary():
    text = "aaaa aaaa aaaa\n\nbbbb bbbb bbbb\n\ncccc cccc cccc\n"
    assert core.limit_chars(text, 35) == "aaaa aaaa aaaa\n\nbbbb bbbb bbbb\n"
    assert core.limit_chars(text, None) == text
    assert core.limit_chars(text, 10_000) == text


def test_split_paragraphs_holds_some_out_reproducibly():
    text = "\n\n".join(f"paragraph number {i} is here" for i in range(20)) + "\n"
    train, val = core.split_paragraphs(text, 0.2, seed=3)
    assert len(val.strip().split("\n\n")) == 4
    assert len(train.strip().split("\n\n")) == 16
    assert not set(train.split("\n\n")) & set(val.split("\n\n"))
    assert core.split_paragraphs(text, 0.2, seed=3) == (train, val)


def test_split_paragraphs_needs_at_least_two():
    with pytest.raises(ValueError):
        core.split_paragraphs("only one paragraph here\n")


# ---- milestone 2: byte-pair-encoding tokenizer --------------------------------

MEDICAL = (
    "Tachycardia is a heart rate faster than normal. Bradycardia is a heart rate slower than normal. "
    "Hypertension is high blood pressure. Hypotension is low blood pressure.\n\n"
) * 20


def naive_merges(text, vocab_size, min_count=2):
    """Reference trainer that recounts every pair each round (slow but obviously correct)."""
    words = Counter(tuple(chunk.encode("utf-8")) for chunk in core._PRETOKEN.findall(text))
    merges = []
    while 256 + len(merges) < vocab_size:
        pair_counts = Counter()
        for word, count in words.items():
            for pair in zip(word, word[1:]):
                pair_counts[pair] += count
        if not pair_counts:
            break
        best, best_count = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
        if best_count < min_count:
            break
        merges.append(best)
        new_id = 255 + len(merges)
        words = Counter({tuple(core._merge(list(w), best, new_id)): c for w, c in words.items()})
    return merges


def test_untrained_tokenizer_is_plain_bytes():
    tok = core.BPETokenizer()
    assert tok.vocab_size == 256
    assert tok.encode("Hi") == [72, 105]
    assert tok.decode([72, 105]) == "Hi"


def test_pretokenizer_covers_every_character():
    rng = random.Random(0)
    chars = [chr(c) for c in list(range(32, 127)) + [9, 10, 13, 160, 233, 8212, 8217, 0x4E2D, 0x1F600, 0x301, 0xB2]]
    for _ in range(500):
        text = "".join(rng.choice(chars) for _ in range(rng.randint(0, 30)))
        assert "".join(core._PRETOKEN.findall(text)) == text


def test_training_learns_merges_and_shortens_text():
    tok = core.BPETokenizer().train(MEDICAL, 300)
    assert 256 < tok.vocab_size <= 300
    assert len(tok.encode(MEDICAL)) < len(MEDICAL.encode("utf-8")) / 2


def test_common_words_become_single_tokens():
    tok = core.BPETokenizer().train(MEDICAL, 400)
    assert len(tok.encode(" normal")) == 1
    assert len(tok.encode(" heart")) == 1


def test_round_trip_for_any_text():
    tok = core.BPETokenizer().train(MEDICAL, 320)
    for text in ["Tachycardia is fast.", "caf\u00e9 \u4e2d\u6587 \U0001F600", "a  b\n\nc_d 12,5", "", "   "]:
        assert tok.decode(tok.encode(text)) == text


def test_fast_training_matches_the_reference_trainer():
    rng = random.Random(1)
    for _ in range(4):
        text = "".join(rng.choice("aeiou rstln,.\n") for _ in range(rng.randint(300, 2500)))
        assert list(core.BPETokenizer().train(text, 300).merges) == naive_merges(text, 300)


def test_training_is_reproducible():
    first = core.BPETokenizer().train(MEDICAL, 300)
    second = core.BPETokenizer().train(MEDICAL, 300)
    assert first.to_dict() == second.to_dict()


def test_training_stops_when_nothing_repeats():
    tok = core.BPETokenizer().train("abc", 500)
    assert tok.vocab_size == 256


def test_vocab_size_below_256_is_rejected():
    with pytest.raises(ValueError):
        core.BPETokenizer().train("hello", 100)


def test_saved_tokenizer_encodes_identically():
    tok = core.BPETokenizer().train(MEDICAL, 320)
    loaded = core.BPETokenizer.from_dict(tok.to_dict())
    text = "Hypertension and tachycardia"
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.vocab_size == tok.vocab_size


# ---- milestones 3-5: the GPT (needs PyTorch) --------------------------------

def torch_modules():
    torch = pytest.importorskip("torch")
    from app import llm

    return torch, llm


def tiny_model(llm, vocab_size=300):
    import torch

    torch.manual_seed(0)
    cfg = llm.GPTConfig(vocab_size=vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    return llm.GPT(cfg).eval()


def test_forward_shapes_and_loss():
    torch, llm = torch_modules()
    model = tiny_model(llm)
    x = torch.randint(0, 300, (3, 10))
    logits, loss = model(x, x)
    assert logits.shape == (3, 10, 300)
    assert loss.item() > 0


def test_model_is_causal():
    torch, llm = torch_modules()
    model = tiny_model(llm)
    x = torch.randint(0, 300, (1, 12))
    changed = x.clone()
    changed[0, -1] = (changed[0, -1] + 1) % 300
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(changed)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_heads_must_divide_the_embedding_size():
    _, llm = torch_modules()
    with pytest.raises(ValueError):
        llm.GPT(llm.GPTConfig(vocab_size=300, n_embd=30, n_head=4))


def test_generation_returns_ids_and_probabilities():
    _, llm = torch_modules()
    model = tiny_model(llm)
    ids, probs = llm.generate_ids(model, [1, 2, 3], 7, seed=1)
    assert len(ids) == len(probs) == 7
    assert all(0 <= i < 300 for i in ids)
    assert all(0 < p <= 1 for p in probs)


def test_generation_is_reproducible_with_a_seed():
    _, llm = torch_modules()
    model = tiny_model(llm)
    assert llm.generate_ids(model, [5], 10, seed=4) == llm.generate_ids(model, [5], 10, seed=4)


def test_temperature_zero_is_greedy_and_ignores_the_seed():
    _, llm = torch_modules()
    model = tiny_model(llm)
    first, _ = llm.generate_ids(model, [5, 6], 8, temperature=0, seed=1)
    second, _ = llm.generate_ids(model, [5, 6], 8, temperature=0, seed=2)
    assert first == second


def test_top_k_one_is_greedy():
    _, llm = torch_modules()
    model = tiny_model(llm)
    greedy, _ = llm.generate_ids(model, [5, 6], 8, temperature=0)
    top_one, _ = llm.generate_ids(model, [5, 6], 8, temperature=1.0, top_k=1, seed=3)
    assert greedy == top_one


def test_prompt_longer_than_the_context_is_handled():
    _, llm = torch_modules()
    model = tiny_model(llm)
    ids, _ = llm.generate_ids(model, list(range(40)), 3, seed=1)
    assert len(ids) == 3


def test_bad_generation_settings_are_rejected():
    _, llm = torch_modules()
    model = tiny_model(llm)
    for kwargs in ({"temperature": -1}, {"top_k": 0}, {"max_new_tokens": -1}):
        settings = {"max_new_tokens": 3, **kwargs}
        with pytest.raises(ValueError):
            llm.generate_ids(model, [1], **settings)
    with pytest.raises(ValueError):
        llm.generate_ids(model, [], 3)


def test_complete_ignores_a_trailing_space_in_the_prompt():
    _, llm = torch_modules()
    tok = core.BPETokenizer().train(MEDICAL, 300)
    model = tiny_model(llm, tok.vocab_size)
    result = llm.complete(model, tok, "The heart is ", 5, seed=1)
    assert result.prompt == "The heart is"
    assert len(result.tokens) == 5


def test_learning_rate_warms_up_then_decays():
    _, llm = torch_modules()
    assert llm.lr_at(1, 1000, 1e-3) < llm.lr_at(50, 1000, 1e-3)
    assert llm.lr_at(50, 1000, 1e-3) == pytest.approx(1e-3)
    assert llm.lr_at(1000, 1000, 1e-3) == pytest.approx(1e-4)


def test_training_lowers_the_validation_loss():
    torch, llm = torch_modules()
    tok = core.BPETokenizer().train(MEDICAL, 300)
    data = torch.tensor(tok.encode(MEDICAL), dtype=torch.long)
    model = tiny_model(llm, tok.vocab_size)
    result = llm.train(model, data, data, steps=80, batch_size=8, lr=3e-3, eval_every=40, log=lambda *_: None)
    assert result["best_val_loss"] < result["history"][0]["val"]
