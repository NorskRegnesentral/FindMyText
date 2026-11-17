from html.parser import HTMLParser

import pytest

from experiments.review.export_pair_review import highlight_text
from findmytext.web import tokenize_with_offsets
from findmytext.winnower import Winnower


class TextCollector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text = []

    def handle_data(self, data):
        self.text.append(data)


def test_highlights_preserve_original_text_punctuation_and_whitespace():
    text = "  One-two, three & <four>!\n\nFem seks sju.\n"
    offsets = tokenize_with_offsets(text)
    assert Winnower().tokenize(text) == ["Onetwo", "three", "four", "Fem", "seks", "sju"]
    output = highlight_text(text, offsets, [0, 2], [2], 3, anchor_id="best")
    reader = TextCollector()
    reader.feed(output)
    assert "".join(reader.text) == text
    assert "<four>" not in output
    assert "&lt;" in output and "&gt;" in output
    assert '<mark id="best"><strong>' in output
    assert output.count('id="best"') == 1
    assert output.startswith("  <mark>")


def test_empty_and_unhighlighted_text_roundtrip():
    assert highlight_text("", [], [], [], 4) == ""
    assert highlight_text("<script> & untouched", [], [], [], 4) == "&lt;script&gt; &amp; untouched"


def test_invalid_highlight_positions_and_unrelated_cluster_are_explicit():
    offsets = tokenize_with_offsets("one two three four")
    with pytest.raises(ValueError, match="contained"):
        highlight_text("one two three four", offsets, [0], [1], 2)
    with pytest.raises(ValueError, match="outside"):
        highlight_text("one two three four", offsets, [2], [], 4)
