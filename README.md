[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/vt3c8dtapg/badge.svg)](https://msrit.ask.thili.ai/b/vt3c8dtapg)
# Medical Terminology Helper

A command-line assistant that completes partial medical phrases and definitions. It is
powered by a small GPT built from scratch in PyTorch (byte-pair tokenizer, embeddings,
causal self-attention, transformer blocks, training loop, sampling) and trained on public
domain medical texts. It runs on a laptop CPU, with no API keys.

> **Use it for vocabulary practice only.** A model this small writes fluent text that is
> often wrong. Check anything that matters in a textbook, and never use it for medical
> decisions. The tool prints this reminder with every completion.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # skip if torch is installed
pip install -r requirements.txt
pytest                                   # check everything works

python -m app.cli fetch                  # download a public domain anatomy and physiology book
python -m app.cli train                  # train the tokenizer and the GPT (a few minutes)
python -m app.cli complete "The femur is"
python -m app.cli complete Tachycardia is --length 30 --temperature 0.5 --show-probs
python -m app.cli chat
python -m app.cli tokens "gastritis and hepatitis"
```

Run `fetch` before `train`. The bundled `data/glossary.txt` alone is too small to train on
well, so the model would only memorise it.

## Commands

| Command | What it does |
|---|---|
| `fetch` | Download a plain-text book (default: a Project Gutenberg anatomy and physiology text), clean it, and save it in `data/`. Use `--url` and `--name` for another text. |
| `train` | Train the BPE tokenizer and the GPT on everything in `data/`. |
| `complete PHRASE...` | Complete a partial phrase or definition. |
| `chat` | Enter phrases one after another. `/quit` leaves. |
| `tokens TEXT...` | Show how the trained tokenizer splits text. |

Options for `complete` and `chat`:

| Option | Meaning |
|---|---|
| `--length N` | Number of tokens to generate (default 40; a token is a word or part of one). |
| `--temperature T` | 0 gives the most likely token every time. Higher values give more varied text (default 0.8). |
| `--top-k K` | Only sample from the K most likely tokens (default 40). |
| `--show-probs` | Show the probability the model gave each generated token. |
| `--seed N` | Repeat the same output. |

Options for `train` (defaults in brackets): `--steps` (1500), `--vocab-size` (512),
`--block-size` (64), `--n-layer` (4), `--n-head` (4), `--n-embd` (128), `--dropout` (0.1),
`--batch-size` (32), `--lr` (1e-3), `--max-chars` (400000). For a quick test run, use
`--steps 300`.

Defaults can also be set in a `.env` file (copy `.env.example`).

## Example output

Real output from the tokenizer, trained on the bundled glossary only. Yours will differ a
little once it is trained on the full text. A `·` marks a space.

```
$ python -m app.cli tokens tachycardia
t|ach|y|cardia
11 characters -> 4 tokens   (· marks a space)

$ python -m app.cli tokens gastritis and hepatitis
g|ast|ritis|·and|·he|p|at|itis
23 characters -> 8 tokens   (· marks a space)
```

`complete` prints your phrase followed by the model's continuation, then the reminder.
With `--show-probs` it also prints a table of each generated token and its probability,
and the average.

## Milestones

| Milestone | Where |
|---|---|
| Curate a corpus of public domain medical texts | `fetch` (`storage.fetch_corpus`), `prepare_corpus` in `app/core.py`, `data/glossary.txt` |
| Process the corpus | `strip_gutenberg`, `prepare_corpus`, `split_paragraphs` in `app/core.py` |
| Byte-pair tokenizer and embeddings | `BPETokenizer` in `app/core.py`; `tok_emb` and `pos_emb` in `GPT` (`app/llm.py`) |
| Causal self-attention and the transformer block | `CausalSelfAttention`, `MLP`, `Block` in `app/llm.py` |
| Training loop | `train`, `get_batch`, `lr_at` in `app/llm.py` |
| Text generation | `generate_ids` and `complete` in `app/llm.py` |
| CLI | `app/cli.py` |
| Stretch: set the length | `--length` |
| Stretch: temperature | `--temperature` (and `--top-k`) |
| Stretch: show next-word probabilities | `--show-probs` |

## How it works

1. **Corpus.** Texts are cleaned (Project Gutenberg header and footer, illustration and page
   markers removed, hard-wrapped lines re-flowed into paragraphs). Paragraphs are shuffled
   and 10% are held out for validation, so both parts contain a bit of every source.
2. **Tokenizer.** Byte-pair encoding starts with the 256 byte values and repeatedly merges the
   most frequent adjacent pair into a new token, until the vocabulary has 512 tokens. Common
   words and word parts such as "itis" become single tokens.
3. **Embeddings.** Each token id becomes a vector, and a second vector encodes its position.
4. **Transformer blocks.** Each block applies causal self-attention (a token can only look at
   earlier tokens), then a feed-forward layer, each with layer norm and a residual connection.
5. **Training.** The model predicts the next token at every position. The loss is cross-entropy,
   the optimizer is AdamW with a warm-up and cosine learning-rate schedule. Validation loss is
   checked regularly and the best weights are kept.
6. **Generation.** The phrase is tokenized, then tokens are sampled one at a time from the
   model's predicted probabilities and added to the context.

## Things to know

- **Output can be wrong.** The model learns the style of the texts, not medical facts.
- **Training time.** With the defaults, expect roughly 5 to 15 minutes on a laptop CPU,
  depending on the machine. Training uses at most 400,000 characters of text.
- **Small corpora are memorised.** Validation loss stops improving and then rises, which is
  why the best weights are restored at the end. More text helps more than more steps.
- **Probabilities.** With `--show-probs`, each token's probability is what the model itself
  predicted (at temperature 1), so a low value marks a token the model was unsure about.
- **A trailing space is dropped.** Words in the training text carry their leading space, so
  `"The femur is "` is treated as `"The femur is"` and the model adds the space itself.
- **Texts and licences.** `fetch` downloads a text that Project Gutenberg lists as public
  domain in the USA. Check the terms for your own country before sharing it. The texts are
  old, so some terms are dated. `data/glossary.txt` was written for this project as a small
  seed of "Term: definition" entries and is not a medical reference.

## Tests

```bash
pytest
```

The tests cover corpus cleaning, the BPE tokenizer (including a check that the fast trainer
matches a simple reference version), the model's causal masking, generation settings, and
every CLI command. `.github/workflows/ci.yml` runs them on GitHub after each push.

## Layout

```
app/core.py      corpus preparation and the BPE tokenizer (no PyTorch needed)
app/llm.py       GPT model, training loop, generation
app/storage.py   .env, corpus download and loading, tokenizer and model files
app/cli.py       fetch / train / complete / chat / tokens
data/            glossary.txt, plus texts downloaded by `fetch`
tests/           unit tests and CLI tests
```
