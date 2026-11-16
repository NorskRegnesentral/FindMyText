"""Build HPLT positional inverted, forward, and optional overlap inverted indexes.

Example: python hplt_build.py lvs_Latn hplt_latvian
"""

import argparse
import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from findmytext import index_builder, winnower
from findmytext.indexing import validate_postings_layout
from findmytext.overlap_inverted import OverlapInvertedIndex, build_overlap_inverted_index

ROOT_PATH = Path("/nr/project/stat/COPY.AI/data/Lars")
MAP_URL = "https://data.hplt-project.org/three/sorted/{language}.map"


def download(language, corpus_path):
    corpus_path.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(MAP_URL.format(language=language)) as response:
        urls = response.read().decode().split()
    for url in urls:
        print(f"Downloading {url}", flush=True)
        # wget -c resumes partial downloads and skips complete ones.
        subprocess.run(
            ["wget", "-c", "-q", "--show-progress", "--progress=dot:giga", url],
            cwd=corpus_path,
            check=True,
        )


def build_index(
    corpus_path: Path,
    build_path: Path,
    nb_workers: int,
    max_nb_documents_before_flush: int = 100_000,
) -> Path:
    """Build rolling64 indexes, resuming only explicitly completed shards.

    A manifest binds this build directory to the source files and settings.
    The final index is published only after the positional and forward builds
    both succeed. Legacy or untracked build directories are never overwritten.
    """
    if nb_workers < 1 or max_nb_documents_before_flush < 1:
        raise ValueError("Worker and flush-document counts must be positive")
    corpus_files = sorted(corpus_path.glob("*.jsonl.zst"))
    if not corpus_files:
        raise FileNotFoundError(f"No corpus shards found in {corpus_path}")
    settings = winnower.Winnower(length=4, window_size=6).to_metadata()
    manifest = {
        "format": "findmytext-hplt-build-v1",
        "settings": settings,
        "min_length": 100,
        "max_length": 100_000,
        "corpus_files": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in corpus_files
        ],
    }
    manifest_path = build_path / "build.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                f"Source files or settings changed for {build_path}; use a new build directory."
            )
    else:
        if build_path.exists() and any(build_path.iterdir()):
            raise ValueError(
                f"Untracked/legacy build directory {build_path}; use a new build directory."
            )
        build_path.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    index_path = build_path / "index"
    if index_path.is_dir():
        meta = json.loads((index_path / "meta.json").read_text())
        if winnower.Winnower.from_metadata(meta).to_metadata() != settings:
            raise ValueError(f"Fingerprint settings do not match in {index_path}")
        validate_postings_layout(meta)
        if not all(
            (index_path / name).is_file()
            for name in (
                "fingerprints.npy",
                "offsets.npy",
                "lengths.npy",
                "postings.dat",
                "doc_name_offsets.npy",
                "doc_name_bytes.npy",
                "forward_offsets.npy",
                "forward_fingerprints.npy",
            )
        ):
            raise ValueError(f"Incomplete published index: {index_path}")
        print(f"Using existing index: {index_path}")
        return index_path

    intermediate_files = []
    for corpus_file in corpus_files:
        shard_name = corpus_file.name.removesuffix(".jsonl.zst")
        shard_path = build_path / "shards" / shard_name
        complete_path = shard_path / "complete.json"
        if complete_path.is_file():
            entries = json.loads(complete_path.read_text())
            for entry in entries:
                path = shard_path / entry["file"]
                if (
                    path.parent != shard_path
                    or not path.is_file()
                    or path.stat().st_size != entry["size"]
                ):
                    raise ValueError(
                        f"Completed shard file is missing or changed: {path}"
                    )
            print(f"Reusing indexed shard {shard_path}", flush=True)
            intermediate_files.extend(
                str(shard_path / entry["file"]) for entry in entries
            )
            continue
        print(f"Indexing {corpus_file}", flush=True)
        files = index_builder.index_file(
            str(corpus_file),
            str(shard_path),
            delete_existing=True,
            nb_workers=nb_workers,
            length=4,
            window_size=6,
            min_length=100,
            max_length=100_000,
            max_nb_documents_before_flush=max_nb_documents_before_flush,
        )
        entries = [
            {"file": Path(path).name, "size": Path(path).stat().st_size}
            for path in files
        ]
        partial = complete_path.with_suffix(".json.tmp")
        partial.write_text(json.dumps(entries, indent=2) + "\n")
        partial.replace(complete_path)
        intermediate_files.extend(files)

    with tempfile.TemporaryDirectory(prefix=".merge-", dir=build_path) as tmp:
        staging = Path(tmp) / "index"
        # Also builds the full, hash-sorted forward index, including singletons.
        index_builder.merge_indexes(intermediate_files, str(staging))
        staging.rename(index_path)
    print(f"Built index: {index_path}")
    return index_path


def build_overlap_inverted(index_path: Path, build_path: Path, batch_entries: int) -> Path:
    if batch_entries < 1:
        raise ValueError("batch_entries must be positive")
    sidecar_path = build_path / "overlap_inverted"
    if sidecar_path.exists():
        OverlapInvertedIndex(sidecar_path, index_path)
        print(f"Using existing overlap inverted index: {sidecar_path}", flush=True)
        return sidecar_path
    return build_overlap_inverted_index(index_path, sidecar_path, batch_entries=batch_entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", help="HPLT language code, e.g. nob_Latn.")
    parser.add_argument("folder", help=f"Dataset folder name under {ROOT_PATH}.")
    parser.add_argument("--nb-workers", type=int, default=8)
    parser.add_argument(
        "--skip-download", action="store_true", help="Use local corpus shards only."
    )
    parser.add_argument(
        "--overlap-inverted",
        action="store_true",
        help="Build the rare-first df >= 2 document-only overlap inverted index afterwards.",
    )
    parser.add_argument("--max-documents-before-flush", type=int, default=100_000)
    parser.add_argument(
        "--batch-entries",
        type=int,
        default=1_000_000,
        help="Overlap inverted scan/sort batch target.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="New/resumable build directory; never use legacy paths.",
    )
    args = parser.parse_args()
    if min(args.nb_workers, args.max_documents_before_flush, args.batch_entries) < 1:
        parser.error("Worker, flush-document, and batch-entry counts must be positive.")

    data_path = ROOT_PATH / args.folder
    corpus_path = data_path / args.language
    if not args.skip_download:
        download(args.language, corpus_path)
    build_path = args.build_dir or data_path / "findmytext_build"
    index_path = build_index(
        corpus_path, build_path, args.nb_workers, args.max_documents_before_flush
    )
    if args.overlap_inverted:
        build_overlap_inverted(index_path, build_path, args.batch_entries)


if __name__ == "__main__":
    main()
