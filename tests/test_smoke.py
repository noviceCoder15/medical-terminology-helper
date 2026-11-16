import pytest

from app import cli, core, storage

BOOK = """*** START OF THE PROJECT GUTENBERG EBOOK TEST BOOK ***

The heart is a muscular organ that pumps blood
through the vessels of the body.

The lungs lie on either side of the heart and
bring oxygen into the blood.

*** END OF THE PROJECT GUTENBERG EBOOK TEST BOOK ***
"""


def run(capsys, *argv):
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def write_corpus(tmp_path, paragraphs=60):
    """A small data folder with enough text to train a tiny model."""
    data = tmp_path / "data"
    data.mkdir()
    terms = ["tachycardia", "bradycardia", "hypertension", "hypotension", "ischemia", "dyspnea"]
    text = "\n\n".join(
        f"{terms[i % len(terms)].capitalize()} number {i}: a condition of the body, seen in the heart and lungs."
        for i in range(paragraphs)
    )
    (data / "terms.txt").write_text(text, encoding="utf-8")
    return str(data)


# ---- corpus commands (no PyTorch needed) -----------------------------------

def test_fetch_downloads_and_cleans_a_book(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(storage, "download_text", lambda url, timeout=30: BOOK)
    code, out, _ = run(capsys, "fetch", "--data", str(tmp_path / "data"), "--name", "book")
    saved = (tmp_path / "data" / "book.txt").read_text()
    assert code == 0 and "saved" in out
    assert "Gutenberg" not in saved
    assert "The heart is a muscular organ that pumps blood through the vessels of the body." in saved


def test_fetch_failure_is_reported_not_raised(tmp_path, capsys, monkeypatch):
    def broken(url, timeout=30):
        raise ConnectionError("no network")

    monkeypatch.setattr(storage, "download_text", broken)
    code, _, err = run(capsys, "fetch", "--data", str(tmp_path / "data"))
    assert code == 1 and "could not download" in err


def test_load_corpus_joins_every_file_in_the_folder(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "a.txt").write_text("First file has a paragraph of text.\n")
    (folder / "b.txt").write_text("Second file has another paragraph.\n")
    text, files = storage.load_corpus(str(folder))
    assert files == ["a.txt", "b.txt"]
    assert "First file" in text and "Second file" in text


def test_load_corpus_reports_an_empty_folder_and_a_missing_one(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        storage.load_corpus(str(empty))
    with pytest.raises(FileNotFoundError):
        storage.load_corpus(str(tmp_path / "nope"))


def test_the_bundled_glossary_is_usable():
    text, files = storage.load_corpus("data/glossary.txt")
    assert files == ["glossary.txt"]
    assert "Tachycardia:" in text and text.count("\n\n") > 50


def test_tokens_command_shows_the_split(tmp_path, capsys):
    tokenizer = core.BPETokenizer().train("tachycardia and bradycardia " * 30, 300)
    path = tmp_path / "tokenizer.json"
    storage.save_tokenizer(str(path), tokenizer)
    code, out, _ = run(capsys, "tokens", "tachycardia", "--tokenizer", str(path))
    assert code == 0
    assert "characters ->" in out and "tokens" in out


def test_tokens_without_a_trained_tokenizer_is_reported(tmp_path, capsys):
    code, _, err = run(capsys, "tokens", "heart", "--tokenizer", str(tmp_path / "none.json"))
    assert code == 1 and "train" in err


def test_complete_without_a_model_is_reported(tmp_path, capsys):
    code, _, err = run(capsys, "complete", "The", "heart", "--model", str(tmp_path / "none.pt"))
    assert code == 1 and "no model" in err


def test_train_with_a_missing_corpus_is_reported(tmp_path, capsys):
    pytest.importorskip("torch")
    code, _, err = run(capsys, "train", "--data", str(tmp_path / "nope"))
    assert code == 1 and "corpus not found" in err


# ---- train, complete, chat (needs PyTorch) ---------------------------------

def test_train_complete_and_chat_end_to_end(tmp_path, capsys):
    pytest.importorskip("torch")
    data = write_corpus(tmp_path)
    model = str(tmp_path / "models" / "gpt.pt")
    tokenizer = str(tmp_path / "models" / "tokenizer.json")
    paths = ["--data", data, "--model", model, "--tokenizer", tokenizer]

    code, out, _ = run(capsys, "train", *paths, "--steps", "6", "--eval-every", "3", "--batch-size", "4",
                       "--block-size", "16", "--n-layer", "1", "--n-head", "2", "--n-embd", "16",
                       "--vocab-size", "300", "--seed", "1")
    assert code == 0 and "best validation loss" in out

    code, out, _ = run(capsys, "complete", "Tachycardia", "is", *paths, "--length", "8", "--seed", "1")
    assert code == 0
    assert out.startswith("Tachycardia is")
    assert "small model" in out                       # the disclaimer is always shown

    code, out, _ = run(capsys, "complete", "Tachycardia", *paths, "--length", "5", "--show-probs", "--seed", "1")
    assert "probability" in out and "average probability" in out

    again = run(capsys, "complete", "Tachycardia", *paths, "--length", "5", "--seed", "7")[1]
    assert again == run(capsys, "complete", "Tachycardia", *paths, "--length", "5", "--seed", "7")[1]

    code, out, _ = run(capsys, "tokens", "tachycardia", *paths)
    assert code == 0 and "tokens" in out

    args = cli.build_parser().parse_args(["chat", *paths, "--length", "5", "--seed", "1"])
    model_obj, tok = storage.load_checkpoint(args.model)
    typed = iter(["", "Hypertension is", "/quit"])
    lines = []
    cli.run_chat(model_obj, tok, args, input_fn=lambda prompt="": next(typed), out=lines.append)
    assert len(lines) == 3                            # banner, disclaimer, one completion
    assert lines[2].startswith("Hypertension is")


def test_train_with_too_little_text_is_reported(tmp_path, capsys):
    pytest.importorskip("torch")
    data = tmp_path / "data"
    data.mkdir()
    (data / "tiny.txt").write_text("Only a short paragraph here.\n\nAnd one more short one.\n")
    code, _, err = run(capsys, "train", "--data", str(data), "--model", str(tmp_path / "m.pt"),
                       "--tokenizer", str(tmp_path / "t.json"), "--steps", "2")
    assert code == 1 and "too small" in err
