import json

from findmytext import utils


def _lines(*docs):
    return [json.dumps(doc) + "\n" for doc in docs]


def test_json_line_reader_skips_repeated_ids_with_a_warning(capsys):
    long_text = "word " * 50
    lines = _lines(
        {"id": "a", "text": long_text + "first"},
        {"id": "b", "text": long_text},
        {"id": "a", "text": long_text + "repeat"},
        {"id": "a", "text": long_text + "repeat again"},
    )

    docs = list(utils.json_line_reader(lines, min_length=10))

    assert [d["id"] for d in docs] == ["a", "b"]
    assert docs[0]["text"].endswith("first")
    captured = capsys.readouterr()
    assert captured.err.count("duplicate document ID a") == 2
    assert "line 3" in captured.err and "line 4" in captured.err
    assert "2 kept, 0 skipped, 2 duplicates" in captured.out


def test_get_filtered_files_matches_any_prefix_and_any_suffix(tmp_path):
    for name in ("b.jsonl.gz", "a.jsonl.zst", "a.txt", "partialindex(5,6)_a.jsonl"):
        (tmp_path / name).touch()
    (tmp_path / "partialindex(5,6)_b").mkdir()

    def names(**kwargs):
        paths = utils.get_filtered_files(str(tmp_path), **kwargs)
        assert all(p.startswith(str(tmp_path)) for p in paths)
        return [p[len(str(tmp_path)) + 1 :] for p in paths]

    # No constraints: the whole directory, sorted, directories included
    assert names() == sorted(
        ["b.jsonl.gz", "a.jsonl.zst", "a.txt", "partialindex(5,6)_a.jsonl", "partialindex(5,6)_b"]
    )
    assert names(file_suffixes=(".jsonl.zst", ".jsonl.gz")) == ["a.jsonl.zst", "b.jsonl.gz"]
    assert names(file_prefixes=("partialindex(5,6)_",)) == [
        "partialindex(5,6)_a.jsonl",
        "partialindex(5,6)_b",
    ]
    # Both given: an entry must satisfy one prefix AND one suffix
    assert names(file_prefixes=("a", "partialindex"), file_suffixes=(".jsonl",)) == [
        "partialindex(5,6)_a.jsonl"
    ]
    assert names(file_prefixes=["x"]) == []
