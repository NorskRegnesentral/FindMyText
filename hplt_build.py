"""Download an HPLT v3 language and build its fingerprint index (inverted + forward).

Example: python hplt_build.py lvs_Latn hplt_latvian
"""

import argparse
import subprocess
import urllib.request
from pathlib import Path

from findmytext import index_builder

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


def build_index(corpus_path, build_path, nb_workers):
    index_path = build_path / "index"
    if index_path.is_dir():
        if not all(
            (index_path / name).is_file()
            for name in ("forward_offsets.npy", "forward_fingerprints.npy")
        ):
            index_builder.build_forward_index(str(index_path))
        print(f"Using existing index: {index_path}")
        return

    corpus_files = sorted(corpus_path.glob("*.jsonl.zst"))
    if not corpus_files:
        raise FileNotFoundError(f"No corpus shards found in {corpus_path}")

    intermediate_files = []
    for corpus_file in corpus_files:
        shard_name = corpus_file.name.removesuffix(".jsonl.zst")
        shard_path = build_path / "shards" / shard_name
        # final.mpk.gz is written last, so its presence means the shard is complete.
        if (shard_path / "final.mpk.gz").is_file():
            print(f"Reusing indexed shard {shard_path}", flush=True)
            intermediate_files.extend(map(str, sorted(shard_path.glob("*.mpk.gz"))))
            continue
        print(f"Indexing {corpus_file}", flush=True)
        intermediate_files.extend(
            index_builder.index_file(
                str(corpus_file),
                str(shard_path),
                delete_existing=True,
                nb_workers=nb_workers,
                length=4,
                window_size=6,
                min_length=100,
                max_length=100_000,
            )
        )

    # Also builds the forward index.
    index_builder.merge_indexes(intermediate_files, str(index_path))
    print(f"Built index: {index_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", help="HPLT language code, e.g. nob_Latn.")
    parser.add_argument("folder", help=f"Dataset folder name under {ROOT_PATH}.")
    parser.add_argument("--nb-workers", type=int, default=8)
    args = parser.parse_args()

    data_path = ROOT_PATH / args.folder
    corpus_path = data_path / args.language
    download(args.language, corpus_path)
    build_index(corpus_path, data_path / "findmytext_build", args.nb_workers)


if __name__ == "__main__":
    main()
