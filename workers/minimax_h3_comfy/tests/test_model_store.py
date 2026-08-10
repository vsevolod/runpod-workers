"""model_store: resolve HF cache snapshot, materialize /models via symlinks only."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import download_weights as dw  # noqa: E402
import model_store as ms  # noqa: E402

# Four relative paths (under snapshot or under models/)
WEIGHT_RELS = [rel for rel, _ in dw.WEIGHTS]


def _write_weight(path: Path, content: bytes = b"weight") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _make_cache_layout(
    cache_root: Path,
    model_id: str,
    *,
    snapshot_hash: str = "abc123",
    with_refs_main: bool = True,
    layout: str = "top",  # "top" | "under_models"
    weight_rels: list[str] | None = None,
    extra_snapshots: list[str] | None = None,
    via_blobs: bool = False,
) -> Path:
    """Build HF hub cache: models--org--name/snapshots/<hash>/...

    If via_blobs=True, write bytes under blobs/ and place relative symlinks
    in the snapshot (matches real huggingface_hub cache).
    """
    org, name = model_id.split("/", 1)
    model_root = cache_root / f"models--{org}--{name}"
    snap = model_root / "snapshots" / snapshot_hash
    snap.mkdir(parents=True, exist_ok=True)
    blobs = model_root / "blobs"
    if via_blobs:
        blobs.mkdir(parents=True, exist_ok=True)

    if with_refs_main:
        refs = model_root / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(snapshot_hash + "\n")

    rels = weight_rels if weight_rels is not None else WEIGHT_RELS
    for i, rel in enumerate(rels):
        if layout == "under_models":
            dest = snap / "models" / rel
            nest_depth = 1 + rel.count("/")  # models/ + parents of file
        else:
            dest = snap / rel
            nest_depth = rel.count("/")  # parent dirs of file only
        dest.parent.mkdir(parents=True, exist_ok=True)
        if via_blobs:
            blob_id = f"fakeblob{i:02d}"
            blob_path = blobs / blob_id
            blob_path.write_bytes(b"blob-payload")
            # snapshots/<hash>/ + nest_depth parent components → model_root
            depth = 2 + nest_depth
            rel_target = ("../" * depth) + f"blobs/{blob_id}"
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(rel_target)
        else:
            _write_weight(dest)

    if extra_snapshots:
        for h in extra_snapshots:
            (model_root / "snapshots" / h).mkdir(parents=True, exist_ok=True)

    return snap


def _make_volume_tree(volume_root: Path, rels: list[str] | None = None) -> Path:
    models = volume_root / "models"
    for rel in rels if rels is not None else WEIGHT_RELS:
        _write_weight(models / rel)
    return models


class TestResolveSnapshotPath(unittest.TestCase):
    def test_v0_1_refs_main(self):
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            expected = _make_cache_layout(cache, "Comfy-Org/MiniMax-H3", snapshot_hash="deadbeef")
            got = ms.resolve_snapshot_path("Comfy-Org/MiniMax-H3", cache)
            self.assertEqual(got.resolve(), expected.resolve())

    def test_v0_2_exactly_one_snapshot_no_refs(self):
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            expected = _make_cache_layout(
                cache,
                "org/name",
                snapshot_hash="onlyone",
                with_refs_main=False,
            )
            got = ms.resolve_snapshot_path("org/name", cache)
            self.assertEqual(got.resolve(), expected.resolve())

    def test_v0_3_two_snapshots_no_refs_fails(self):
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            _make_cache_layout(
                cache,
                "org/name",
                snapshot_hash="aaa",
                with_refs_main=False,
                extra_snapshots=["bbb"],
            )
            with self.assertRaises(ms.ModelStoreError) as ctx:
                ms.resolve_snapshot_path("org/name", cache)
            msg = str(ctx.exception)
            self.assertIn("aaa", msg)
            self.assertIn("bbb", msg)

    def test_v0_4_missing_hub(self):
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "empty_hub"
            cache.mkdir()
            with self.assertRaises(ms.ModelStoreError) as ctx:
                ms.resolve_snapshot_path("org/name", cache)
            msg = str(ctx.exception).lower()
            self.assertTrue("cache" in msg or "model" in msg or "not found" in msg)

    def test_reject_empty_model_id(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ms.ModelStoreError):
                ms.resolve_snapshot_path("", Path(td))

    def test_reject_pin_syntax(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ms.ModelStoreError) as ctx:
                ms.resolve_snapshot_path("org/name:abc123", Path(td))
            self.assertIn(":", str(ctx.exception))


class TestMaterialize(unittest.TestCase):
    def test_v0_5_top_level_layout_symlinks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src_root = root / "src"
            dest = root / "models_out"
            sources = {}
            for rel in WEIGHT_RELS:
                p = _write_weight(src_root / rel)
                sources[rel] = p
            ms.materialize_comfy_models(sources, dest)
            for rel in WEIGHT_RELS:
                target = dest / rel
                self.assertTrue(target.is_symlink(), f"{rel} should be symlink")
                self.assertTrue(target.resolve().is_file())

    def test_v0_6_under_models_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            snap = root / "snap"
            dest = root / "models_out"
            sources = {}
            for rel in WEIGHT_RELS:
                p = _write_weight(snap / "models" / rel)
                sources[rel] = p
            ms.materialize_comfy_models(sources, dest)
            for rel in WEIGHT_RELS:
                self.assertTrue((dest / rel).is_symlink())

    def test_v0_7_never_copies(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = _write_weight(root / "src" / WEIGHT_RELS[0], b"UNIQUE_BYTES_NO_COPY")
            dest = root / "out"
            ms.materialize_comfy_models({WEIGHT_RELS[0]: src}, dest)
            link = dest / WEIGHT_RELS[0]
            self.assertTrue(link.is_symlink())
            # Dest path must not be a regular file copy of the same inode path
            self.assertNotEqual(link, src)
            self.assertEqual(link.read_bytes(), b"UNIQUE_BYTES_NO_COPY")
            # Only symlink at dest; no second large write of content as regular file
            self.assertFalse(link.is_file() and not link.is_symlink())


class TestFindWeights(unittest.TestCase):
    def test_find_top_level(self):
        with tempfile.TemporaryDirectory() as td:
            snap = Path(td)
            for rel in WEIGHT_RELS:
                _write_weight(snap / rel)
            found = ms.find_weight_sources(snap)
            self.assertEqual(set(found), set(WEIGHT_RELS))

    def test_find_under_models(self):
        with tempfile.TemporaryDirectory() as td:
            snap = Path(td)
            for rel in WEIGHT_RELS:
                _write_weight(snap / "models" / rel)
            found = ms.find_weight_sources(snap)
            self.assertEqual(set(found), set(WEIGHT_RELS))

    def test_v0_8_missing_one_of_four(self):
        with tempfile.TemporaryDirectory() as td:
            snap = Path(td)
            for rel in WEIGHT_RELS[:3]:
                _write_weight(snap / rel)
            with self.assertRaises(ms.ModelStoreError) as ctx:
                ms.find_weight_sources(snap)
            self.assertIn(WEIGHT_RELS[3], str(ctx.exception))


class TestPrepareModels(unittest.TestCase):
    def test_v0_9_cache_over_empty_volume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache = root / "hub"
            dest = root / "models"
            model_id = "Comfy-Org/MiniMax-H3"
            _make_cache_layout(cache, model_id)
            # empty/missing volume
            env = {
                "MODEL_NAME": model_id,
                "HF_CACHE_ROOT": str(cache),
                "MODEL_DIR": str(root / "no_volume"),
            }
            buf = io.StringIO()
            with mock.patch.dict(os.environ, env, clear=False), redirect_stdout(buf):
                ms.prepare_models(dest=dest)
            out = buf.getvalue()
            self.assertIn("source=cache", out)
            for rel in WEIGHT_RELS:
                self.assertTrue((dest / rel).is_symlink())

    def test_v0_10_volume_when_no_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache = root / "hub"
            cache.mkdir()
            vol = root / "volume"
            _make_volume_tree(vol)
            dest = root / "models"
            env = {
                "MODEL_NAME": "org/unused",
                "HF_CACHE_ROOT": str(cache),
                "MODEL_DIR": str(vol),
            }
            buf = io.StringIO()
            with mock.patch.dict(os.environ, env, clear=False), redirect_stdout(buf):
                ms.prepare_models(dest=dest)
            out = buf.getvalue()
            self.assertIn("source=volume", out)
            for rel in WEIGHT_RELS:
                self.assertTrue((dest / rel).is_symlink())

    def test_v0_11_neither_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache = root / "hub"
            cache.mkdir()
            dest = root / "models"
            env = {
                "MODEL_NAME": "org/missing",
                "HF_CACHE_ROOT": str(cache),
                "MODEL_DIR": str(root / "nope"),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                with self.assertRaises(ms.ModelStoreError):
                    ms.prepare_models(dest=dest)

    def test_cache_preferred_over_volume(self):
        """V4.2 optional preference: complete cache wins."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache = root / "hub"
            model_id = "slim/t2v"
            _make_cache_layout(cache, model_id)
            vol = root / "volume"
            _make_volume_tree(vol)
            dest = root / "models"
            env = {
                "MODEL_NAME": model_id,
                "HF_CACHE_ROOT": str(cache),
                "MODEL_DIR": str(vol),
            }
            buf = io.StringIO()
            with mock.patch.dict(os.environ, env, clear=False), redirect_stdout(buf):
                ms.prepare_models(dest=dest)
            self.assertIn("source=cache", buf.getvalue())

    def test_v0_13_comfy_models_root_env_ignored(self):
        """COMFY_MODELS_ROOT must not override hard-coded COMFY_MODELS CLI path."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache = root / "hub"
            dest = root / "models_cli"
            wrong = root / "wrong"
            wrong.mkdir()
            model_id = "org/name"
            _make_cache_layout(cache, model_id)
            env = {
                "MODEL_NAME": model_id,
                "HF_CACHE_ROOT": str(cache),
                "MODEL_DIR": str(root / "no_vol"),
                "COMFY_MODELS_ROOT": str(wrong),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch.object(ms, "COMFY_MODELS", dest):
                    rc = ms.main()
            self.assertEqual(rc, 0)
            for rel in WEIGHT_RELS:
                self.assertTrue((dest / rel).is_symlink(), rel)
                self.assertFalse(
                    (wrong / rel).exists(),
                    f"must not write under COMFY_MODELS_ROOT: {rel}",
                )
            # Module has no COMFY_MODELS_ROOT override; prod constant is hard-coded
            self.assertFalse(hasattr(ms, "COMFY_MODELS_ROOT"))
            self.assertEqual(ms.COMFY_MODELS, Path("/models"))

    def test_snapshot_via_blobs_symlinks(self):
        """Real HF cache: snapshot entries are relative symlinks into blobs/."""
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            model_id = "Comfy-Org/MiniMax-H3"
            snap = _make_cache_layout(cache, model_id, via_blobs=True)
            sample = snap / WEIGHT_RELS[0]
            self.assertTrue(sample.is_symlink())
            sources = ms.find_weight_sources(snap)
            self.assertEqual(set(sources), set(WEIGHT_RELS))
            # resolve() must land on real blob bytes
            self.assertTrue(sources[WEIGHT_RELS[0]].is_file())
            self.assertIn("blobs", str(sources[WEIGHT_RELS[0]]))
            dest = cache / "out"
            ms.materialize_comfy_models(sources, dest)
            for rel in WEIGHT_RELS:
                self.assertTrue((dest / rel).is_symlink())
                self.assertEqual((dest / rel).read_bytes(), b"blob-payload")

    def test_cli_main_exit_nonzero_on_fail(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = {
                "MODEL_NAME": "org/x",
                "HF_CACHE_ROOT": str(root / "hub"),
                "MODEL_DIR": str(root / "vol"),
            }
            (root / "hub").mkdir()
            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch.object(ms, "COMFY_MODELS", root / "models"):
                    rc = ms.main()
            self.assertNotEqual(rc, 0)


class TestSharedWeightsConstant(unittest.TestCase):
    def test_weight_list_matches_download_weights(self):
        self.assertEqual(ms.WEIGHT_RELS, WEIGHT_RELS)
        self.assertEqual(len(ms.WEIGHT_RELS), 4)


if __name__ == "__main__":
    unittest.main()
