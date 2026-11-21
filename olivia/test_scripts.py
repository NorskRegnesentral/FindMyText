import os
import tempfile
import unittest
from unittest.mock import patch

from olivia import scripts


class IndexFileTests(unittest.TestCase):
    def test_index_file_nr_builds_the_partial_index_of_the_numbered_file(self):
        with tempfile.TemporaryDirectory() as work_path:
            corpus_dir = os.path.join(work_path, "corpus")
            os.mkdir(corpus_dir)
            for name in ("b.jsonl.gz", "a.jsonl.zst"):  # sorted: a is file 0, b is file 1
                open(os.path.join(corpus_dir, name), "w").close()
            with patch.object(scripts, "WORK_PATH", work_path), \
                    patch.object(scripts.index_builder, "build_index") as build_index:
                scripts.index_file_nr(
                    1, subdir="corpus", length=3, window_size=4,
                    stop_after=10, nb_workers=2, max_memory_gb=4.0,
                )
            build_index.assert_called_once_with(
                os.path.join(corpus_dir, "b.jsonl.gz"),
                os.path.join(corpus_dir, "partialindex(3,4)_b-first10"),
                3, 4, 2, 4.0, 10, delete_existing=True,
            )

    def test_index_file_nr_fails_clearly_when_the_directory_has_no_corpus_files(self):
        with tempfile.TemporaryDirectory() as work_path:
            os.mkdir(os.path.join(work_path, "corpus"))
            open(os.path.join(work_path, "corpus", "notes.txt"), "w").close()
            with patch.object(scripts, "WORK_PATH", work_path), \
                    patch.object(scripts.index_builder, "build_index") as build_index, \
                    self.assertRaisesRegex(ValueError, "No .jsonl"):
                scripts.index_file_nr(0, subdir="corpus")
            build_index.assert_not_called()

    def test_index_dir_keeps_everything_but_the_corpus_suffix(self):
        # 5_1.ab, 5_1.cc and 5_1.ia must not share one index directory.
        cases = [
            ("/elsewhere/5_1.ab.jsonl.zst", 5, 6, None, "/elsewhere/partialindex(5,6)_5_1.ab"),
            ("/elsewhere/5_1.cc.jsonl.gz", 5, 6, 10, "/elsewhere/partialindex(5,6)_5_1.cc-first10"),
            ("/elsewhere/plain.jsonl", 3, 4, None, "/elsewhere/partialindex(3,4)_plain"),
        ]
        for corpus_file, length, window_size, stop_after, expected in cases:
            self.assertEqual(scripts._index_dir(corpus_file, length, window_size, stop_after), expected)

    def test_merge_index_merges_matching_partial_indexes_into_fullindex(self):
        with tempfile.TemporaryDirectory() as work_path:
            corpus_dir = os.path.join(work_path, "corpus")
            os.mkdir(corpus_dir)
            partials = [os.path.join(corpus_dir, f"partialindex(5,6)_{name}") for name in ("b", "a")]
            for d in partials:
                os.mkdir(d)
                open(os.path.join(d, "meta.json"), "w").close()
            # Ignored: other parameters, leftover shard directory, directory without meta.json
            for name in ("partialindex(4,6)_c", "partialindex(5,6)_a_shards", "partialindex(5,6)_d"):
                os.mkdir(os.path.join(corpus_dir, name))
            open(os.path.join(corpus_dir, "partialindex(4,6)_c", "meta.json"), "w").close()

            with patch.object(scripts, "WORK_PATH", work_path), \
                    patch.object(scripts.index_builder, "Merger") as merger:
                scripts.merge_index(subdir="corpus", nb_workers=4, max_memory_gb=16)

            merger.assert_called_once_with(4, 16)
            merger.return_value.merge.assert_called_once_with(
                sorted(partials), os.path.join(corpus_dir, "fullindex(5,6)"),
            )
            with patch.object(scripts, "WORK_PATH", work_path), \
                    self.assertRaisesRegex(ValueError, "No partialindex"):
                scripts.merge_index(subdir="corpus", length=7)
            # A single partial index is copied to fullindex without merging.
            full_dir = os.path.join(corpus_dir, "fullindex(4,6)")
            with patch.object(scripts, "WORK_PATH", work_path), \
                    patch.object(scripts.index_builder, "Merger") as merger:
                scripts.merge_index(subdir="corpus", length=4)
            merger.assert_not_called()
            self.assertTrue(os.path.isfile(os.path.join(full_dir, "meta.json")))
            self.assertTrue(os.path.isdir(os.path.join(corpus_dir, "partialindex(4,6)_c")))

    def test_merge_shards_resumes_from_the_shards_of_a_numbered_file(self):
        with tempfile.TemporaryDirectory() as work_path:
            corpus_dir = os.path.join(work_path, "corpus")
            os.mkdir(corpus_dir)
            for name in ("b.jsonl.gz", "a.jsonl.zst"):
                open(os.path.join(corpus_dir, name), "w").close()
            index_dir = os.path.join(corpus_dir, "partialindex(3,4)_b")
            shards_dir = os.path.join(index_dir, "shards")
            os.makedirs(os.path.join(shards_dir, "shard-000000"))
            open(os.path.join(index_dir, "postings.dat"), "w").close()  # leftover of the interrupted merge

            with patch.object(scripts, "WORK_PATH", work_path), \
                    patch.object(scripts.index_builder, "Merger") as merger:
                scripts.merge_shards(1, subdir="corpus", length=3, window_size=4, nb_workers=2, max_memory_gb=4.0)

            merger.assert_called_once_with(2, 4.0)
            merger.return_value.merge_dir.assert_called_once_with(shards_dir, index_dir)
            self.assertFalse(os.path.exists(shards_dir))
            with patch.object(scripts, "WORK_PATH", work_path), \
                    self.assertRaisesRegex(ValueError, "No shards directory"):
                scripts.merge_shards(0, subdir="corpus", length=3, window_size=4)


if __name__ == "__main__":
    unittest.main()
