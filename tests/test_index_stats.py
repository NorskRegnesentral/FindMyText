import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("plotly")

from experiments.index_stats import index_statistics, resolve_indexes
from experiments import index_stats
from findmytext.indexing import POSTING_DTYPE


@pytest.mark.parametrize("include_zero", [False, True])
def test_histogram_annotations_use_axis_coordinates(include_zero):
    table = np.zeros(298, dtype=np.int64)
    table[1] = 800
    table[297] = 200
    if include_zero:
        table[0] = 100
    summary = index_stats._summarize(table, "docs")
    figure = index_stats._histogram(
        table, "Documents per fingerprint", "Documents", summary, "docs"
    )
    expected_values = sorted({
        summary[f"docs_p{level * 100:g}"] for level in index_stats.QUANTILES
    } | {summary["docs_max"]})
    shapes = sorted(figure.layout.shapes, key=lambda shape: shape.x0)
    annotations = sorted(figure.layout.annotations, key=lambda annotation: annotation.x)
    assert len(shapes) == len(annotations) == len(expected_values)
    for shape, annotation, value in zip(shapes, annotations, expected_values):
        assert shape.x0 == shape.x1 == value
        assert annotation.x == pytest.approx(
            value if include_zero else np.log10(value)
        )
        assert annotation.xref == "x"
        assert annotation.yref == "paper"
        assert annotation.xanchor == ("right" if value == 297 else "left")
    assert figure.layout.xaxis.type == ("linear" if include_zero else "log")
    assert figure.data[0].to_plotly_json()["x"][-1] == 297


def write_index(folder, *, forward=True, layout="interleaved", empty=False):
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text(json.dumps({
        "length": 4, "window_size": 6, "base": 257, "punctuation": False,
        "postings_layout": layout,
    }))
    fingerprints = [] if empty else [10, 20, 30]
    lengths = np.array([] if empty else [4, 2, 1], dtype=np.uint32)
    np.save(folder / "fingerprints.npy", np.array(fingerprints, dtype=np.uint64))
    np.save(folder / "lengths.npy", lengths)
    np.save(folder / "offsets.npy", np.array([] if empty else [0, 32, 48], dtype=np.uint64))
    np.save(folder / "doc_name_offsets.npy",
            np.array([0] if empty else [0, 1, 2, 3, 4], dtype=np.uint64))
    np.save(folder / "doc_name_bytes.npy",
            np.frombuffer(b"" if empty else b"abcd", dtype=np.uint8))
    entries = [] if empty else [(0, 0), (0, 1), (0, 2), (1, 0),
                                (0, 3), (0, 4), (2, 0)]
    postings = np.array(entries, dtype=POSTING_DTYPE)
    if layout == "columnar":
        np.concatenate([postings["doc_id"], postings["position"]]).tofile(folder / "postings.dat")
    else:
        postings.tofile(folder / "postings.dat")
    if forward:
        np.save(folder / "forward_offsets.npy",
                np.array([0] if empty else [0, 2, 3, 4, 4], dtype=np.uint64))
        np.save(folder / "forward_fingerprints.npy",
                np.array([] if empty else [10, 20, 10, 30], dtype=np.uint64))
    return folder


@pytest.mark.parametrize("forward", [True, False])
@pytest.mark.parametrize("layout", ["interleaved", "columnar"])
def test_statistics_count_distinct_documents(tmp_path, forward, layout):
    folder = write_index(tmp_path / "index", forward=forward, layout=layout)
    stats, figures = index_statistics("test", folder, batch_entries=2)
    expected = {
        "n_documents": 4,
        "n_documents_without_fingerprints": 1,
        "n_unique_fingerprints": 3,
        "n_postings": 7,
        "n_unique_doc_fingerprint_pairs": 4,
        "n_repeated_postings": 3,
        "singleton_fingerprints": 2,
        "shared_fingerprints": 1,
        "fingerprints_in_exactly_two_documents": 1,
        "fingerprints_in_all_documents": 0,
        "fingerprints_per_doc_min": 0,
        "fingerprints_per_doc_mean": 1,
        "fingerprints_per_doc_p50": 1,
        "fingerprints_per_doc_max": 2,
        "docs_per_fingerprint_max": 2,
        "postings_per_fingerprint_max": 4,
        "inverted_pair_updates": 1,
        "has_forward_index": forward,
        "postings_layout": layout,
    }
    for key, value in expected.items():
        assert stats[key] == value, key
    assert stats["singleton_fingerprints_fraction"] == pytest.approx(2 / 3)
    assert stats["repeated_postings_fraction"] == pytest.approx(3 / 7)
    assert stats["forward_topk_gb_read"] == pytest.approx(11 * 8 / 1e9)
    assert len(figures) == 3
    document_trace = figures[0].data[0].to_plotly_json()
    frequency_trace = figures[1].data[0].to_plotly_json()
    posting_trace = figures[2].data[0].to_plotly_json()
    assert list(document_trace["x"]) == [0, 1, 2]
    assert list(document_trace["y"]) == [1, 2, 1]
    assert figures[0].layout.xaxis.type == "linear"
    assert list(frequency_trace["x"]) == [1, 2]
    assert list(frequency_trace["y"]) == [2, 1]
    assert list(posting_trace["x"]) == [1, 2, 4]


@pytest.mark.parametrize("forward", [True, False])
def test_empty_statistics(tmp_path, forward):
    folder = write_index(tmp_path / "empty", forward=forward, empty=True)
    stats, figures = index_statistics("empty", folder, batch_entries=2)
    for key in ("n_documents", "n_unique_fingerprints", "singleton_fingerprints",
                "repeated_postings_fraction", "singleton_fingerprints_fraction",
                "forward_topk_mb_read_per_document", "fingerprints_per_doc_mean"):
        assert stats[key] == 0
    assert len(figures) == 3
    json.dumps(stats, allow_nan=False)


def test_folder_discovery_and_cli_reports(tmp_path):
    root = tmp_path / "corpora"
    names = ["hplt_nynorsk", "hplt_bokmal", "hplt_icelandic", "hplt_latvian"]
    for name in names:
        write_index(root / name / "findmytext_build" / "index", forward=False)
    (root / "unrelated").mkdir()
    discovered = resolve_indexes([str(root)])
    assert {name for name, _ in discovered} == set(names)
    index_dir = root / names[0] / "findmytext_build" / "index"
    assert resolve_indexes([str(root / names[0])]) == [(names[0], index_dir)]
    assert resolve_indexes([f"nynorsk={index_dir}"]) == [("nynorsk", index_dir)]
    output = tmp_path / "report"
    command = [sys.executable, "-m", "experiments.index_stats", str(root),
               "--output-dir", str(output), "--batch-entries", "2"]
    result = subprocess.run(command, capture_output=True, text=True, check=True,
                            cwd=Path(__file__).resolve().parents[1])
    assert "Found 4 index(es)" in result.stdout
    assert "singleton_fingerprints" in result.stdout
    saved = json.loads((output / "index_stats.json").read_text())
    assert {row["name"] for row in saved} == set(names)
    assert all(row["singleton_fingerprints"] == 2 for row in saved)
    assert len((output / "index_stats.csv").read_text().splitlines()) == 5
    for name in names:
        html = (output / f"index_stats_{name}.html").read_text()
        assert html.startswith("<!doctype html>")
        assert html.endswith("</body></html>")
        assert "plotly.js" in html
        assert '<script src="https://cdn.plot.ly/' not in html
    command[3] = f"{names[0]}={index_dir}"
    subprocess.run(command, capture_output=True, text=True, check=True,
                   cwd=Path(__file__).resolve().parents[1])
    assert len(json.loads((output / "index_stats.json").read_text())) == 4


def test_invalid_inputs(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        index_statistics("test", tmp_path, batch_entries=0)
    with pytest.raises(ValueError, match="No merged indexes"):
        resolve_indexes([str(tmp_path)])
    folder = write_index(tmp_path / "index")
    with pytest.raises(ValueError, match="unique"):
        resolve_indexes([f"a={folder}", f"a={folder}"])
    with pytest.raises(ValueError, match="simple file names"):
        resolve_indexes([f"../escape={folder}"])


def test_cli_defaults(tmp_path, monkeypatch):
    names = ("nynorsk", "bokmal", "icelandic", "latvian")
    expected = [
        f"{name}=/nr/project/stat/COPY.AI/data/Lars/hplt_{name}/findmytext_build/index"
        for name in names
    ]
    assert index_stats.DEFAULT_OUTPUT_DIR == Path(
        "/nr/project/stat/COPY.AI/data/Lars/index_analysis"
    )
    indexes = [(name, write_index(tmp_path / name)) for name in names]
    received = []

    def resolve_defaults(items):
        received.extend(items)
        return indexes

    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["index_stats"])
    monkeypatch.setattr(index_stats, "resolve_indexes", resolve_defaults)
    monkeypatch.setattr(index_stats, "DEFAULT_OUTPUT_DIR", output)
    index_stats.main()
    assert received == expected
    saved = json.loads((output / "index_stats.json").read_text())
    assert [row["name"] for row in saved] == list(names)
    assert (output / "index_stats.csv").is_file()


def test_comparison_axes_and_hover_metadata(tmp_path):
    folder = write_index(tmp_path / "index")
    stats, _ = index_statistics("test", folder, batch_entries=2)
    other = dict(stats, name="other", n_documents=12, n_unique_fingerprints=8,
                 singleton_fingerprints=2, singleton_fingerprints_fraction=0.25)
    fig = index_stats.comparison_figure([stats, other])
    trace = fig.data[0].to_plotly_json()
    assert list(trace["x"]) == [3, 8]
    assert list(trace["y"]) == [2, 2]
    assert list(trace["text"]) == ["test", "other"]
    assert list(trace["marker"]["color"]) == pytest.approx([2 / 3, 0.25])
    assert trace["customdata"][1][0:2] == ["other", 12]
    assert trace["customdata"][0][-1] == str(folder)
    for label in ("Documents:", "Singleton fraction:", "Positional postings:",
                  "Document/fingerprint pairs:", "Index size:", "Index:"):
        assert label in trace["hovertemplate"]
    fraction_fig = index_stats.singleton_fraction_figure([stats, other])
    fraction_trace = fraction_fig.data[0].to_plotly_json()
    assert list(fraction_trace["x"]) == [3, 8]
    assert list(fraction_trace["y"]) == pytest.approx([2 / 3, 0.25])
    assert fraction_fig.layout.yaxis.tickformat == ".0%"
    assert tuple(fraction_fig.layout.yaxis.range) == (0, 1)
    assert fraction_trace["customdata"][1][6] == 2
    assert "Singleton fingerprints: %{customdata[6]:,}" in fraction_trace["hovertemplate"]
    path = index_stats._write_comparison([stats, other], tmp_path)
    layouts = list(index_stats._saved_plot_layouts(path.read_text()))
    assert len(layouts) == 2
    assert layouts[1][2]["yaxis"]["tickformat"] == ".0%"
    assert layouts[1][2]["yaxis"]["range"] == [0, 1]


def test_repair_preserves_histograms_and_does_not_read_indexes(tmp_path, monkeypatch):
    folder = write_index(tmp_path / "index")
    stats, figures = index_statistics("test", folder, batch_entries=2)
    for figure in figures:
        for shape, annotation in zip(figure.layout.shapes, figure.layout.annotations):
            annotation.x = shape.x0
            annotation.xanchor = "center"
    output = tmp_path / "reports"
    output.mkdir()
    json_path = output / "index_stats.json"
    json_path.write_text(json.dumps([stats]))
    csv_path = output / "index_stats.csv"
    csv_path.write_text("Existing CSV must remain unchanged")
    path = output / "index_stats_test.html"
    original = "<html>" + "".join(
        fig.to_html(full_html=False, include_plotlyjs=False) for fig in figures
    ) + "</html>"
    path.write_text(original)

    def unexpected_index_access(*args, **kwargs):
        pytest.fail("Repair must not read any index")

    monkeypatch.setattr(index_stats, "DiskBasedIndex", unexpected_index_access)
    monkeypatch.setattr(index_stats, "resolve_indexes", unexpected_index_access)
    monkeypatch.setattr(sys, "argv", [
        "index_stats", "--repair-plots", "--output-dir", str(output),
    ])
    index_stats.main()
    repaired = path.read_text()
    assert repaired != original
    assert path.with_suffix(".html.bak").read_text() == original

    def without_layouts(html):
        for start, end, _ in reversed(list(index_stats._saved_plot_layouts(html))):
            html = html[:start] + "LAYOUT" + html[end:]
        return html

    assert without_layouts(repaired) == without_layouts(original)
    for _, _, layout in index_stats._saved_plot_layouts(repaired):
        assert layout["xaxis"]["autorange"] is True
        for shape, annotation in zip(layout["shapes"], layout["annotations"]):
            expected = (np.log10(shape["x0"]) if layout["xaxis"]["type"] == "log"
                        else shape["x0"])
            assert annotation["x"] == pytest.approx(expected)
    comparison = output / "index_stats_comparison.html"
    assert comparison.is_file()
    assert "Distinct versus singleton fingerprints by corpus" in comparison.read_text()
    assert json_path.read_text() == json.dumps([stats])
    assert csv_path.read_text() == "Existing CSV must remain unchanged"
    index_stats.main()
    assert path.read_text() == repaired
    assert path.with_suffix(".html.bak").read_text() == original


def test_repair_validates_all_reports_before_overwriting(tmp_path):
    folder = write_index(tmp_path / "index")
    stats, figures = index_statistics("valid", folder, batch_entries=2)
    (tmp_path / "index_stats.json").write_text(json.dumps([
        stats, dict(stats, name="broken"),
    ]))
    valid = tmp_path / "index_stats_valid.html"
    original = "".join(fig.to_html(full_html=False, include_plotlyjs=False)
                       for fig in figures)
    valid.write_text(original)
    (tmp_path / "index_stats_broken.html").write_text("No Plotly charts")
    with pytest.raises(ValueError, match="Expected three histogram charts"):
        index_stats.repair_saved_reports(tmp_path)
    assert valid.read_text() == original
    assert not valid.with_suffix(".html.bak").exists()
