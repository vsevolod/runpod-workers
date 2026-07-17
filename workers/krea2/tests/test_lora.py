import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

_MISSING_MODULE = object()
_CANONICAL_PACKAGE_BEFORE = sys.modules.get("krea2_infer", _MISSING_MODULE)
_CANONICAL_LORA_BEFORE = sys.modules.get("krea2_infer.lora", _MISSING_MODULE)

PACKAGE_ROOT = Path(__file__).parents[1] / "krea2_infer"
_LORA_MODULE_NAME = "_krea2_lora_contract_tests"


def _load_lora_module():
    spec = importlib.util.spec_from_file_location(
        _LORA_MODULE_NAME,
        PACKAGE_ROOT / "lora.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the LoRA contract module")

    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(_LORA_MODULE_NAME, _MISSING_MODULE)
    sys.modules[_LORA_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is _MISSING_MODULE:
            sys.modules.pop(_LORA_MODULE_NAME, None)
        else:
            sys.modules[_LORA_MODULE_NAME] = previous
    return module


_lora = _load_lora_module()
LoRACatalog = _lora.LoRACatalog
LoRAError = _lora.LoRAError
normalize_lora_requests = _lora.normalize_lora_requests


class TestModuleIsolationTest(unittest.TestCase):
    def test_loader_preserves_canonical_package_bindings(self):
        self.assertIs(
            sys.modules.get("krea2_infer", _MISSING_MODULE),
            _CANONICAL_PACKAGE_BEFORE,
        )
        self.assertIs(
            sys.modules.get("krea2_infer.lora", _MISSING_MODULE),
            _CANONICAL_LORA_BEFORE,
        )


class LoRACatalogTest(unittest.TestCase):
    def test_scan_uses_top_level_safetensors_stems_as_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.safetensors"
            second = root / "second.safetensors"
            first.touch()
            second.touch()
            (root / "ignored.txt").touch()
            (root / "nested").mkdir()
            (root / "nested" / "nested.safetensors").touch()
            (root / "not-a-file.safetensors").mkdir()

            catalog = LoRACatalog.scan(root)

            self.assertEqual(catalog.names, ("first", "second"))
            self.assertEqual(catalog.resolve("first"), first)
            self.assertEqual(catalog.resolve("second"), second)

    def test_scan_ignores_top_level_safetensors_symlinks(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as external_directory,
        ):
            root = Path(directory)
            external = Path(external_directory) / "external.safetensors"
            external.touch()
            (root / "linked.safetensors").symlink_to(external)

            catalog = LoRACatalog.scan(root)

            self.assertEqual(catalog.names, ())

    def test_catalog_copies_the_source_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            alpha = root / "alpha.safetensors"
            beta = root / "beta.safetensors"
            paths = {"alpha": alpha}
            catalog = LoRACatalog(root=root, _paths=paths)

            paths["beta"] = beta

            self.assertEqual(catalog.names, ("alpha",))
            self.assertEqual(catalog.resolve("alpha"), alpha)

    def test_catalog_mapping_cannot_be_mutated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            alpha = root / "alpha.safetensors"
            beta = root / "beta.safetensors"
            catalog = LoRACatalog(root=root, _paths={"alpha": alpha})

            with self.assertRaises(TypeError):
                catalog._paths["beta"] = beta

            self.assertEqual(catalog.names, ("alpha",))
            self.assertEqual(catalog.resolve("alpha"), alpha)

    def test_scan_of_missing_directory_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"

            with self.assertLogs(_LORA_MODULE_NAME, level="WARNING"):
                catalog = LoRACatalog.scan(missing)

            self.assertEqual(catalog.names, ())

    def test_scan_rejects_a_non_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "catalog"
            file_path.touch()

            with self.assertRaises(RuntimeError):
                LoRACatalog.scan(file_path)

    def test_scan_rejects_invalid_ids(self):
        for filename in (
            ".safetensors",
            "..safetensors",
            "...safetensors",
            "bad\\name.safetensors",
        ):
            with (
                self.subTest(filename=filename),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                (root / filename).touch()

                with self.assertRaises(RuntimeError):
                    LoRACatalog.scan(root)


class NormalizeLoRARequestsTest(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_directory.name)
        self.alpha_path = self.root / "alpha.safetensors"
        self.beta_path = self.root / "beta.safetensors"
        self.alpha_path.touch()
        self.beta_path.touch()
        self.catalog = LoRACatalog.scan(self.root)

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_none_returns_an_empty_tuple(self):
        self.assertEqual(normalize_lora_requests(None, self.catalog), ())

    def test_defaults_strength_and_omits_zero_strength(self):
        selections = normalize_lora_requests(
            [{"name": "alpha"}, {"name": "beta", "strength": 0.0}],
            self.catalog,
        )

        self.assertEqual(len(selections), 1)
        self.assertEqual(selections[0].name, "alpha")
        self.assertEqual(selections[0].strength, 1.0)
        self.assertEqual(selections[0].path, self.alpha_path)
        self.assertEqual(selections[0].as_dict(), {"name": "alpha", "strength": 1.0})

    def test_rejects_more_than_four_items_before_filtering_zero_strength(self):
        requests = [{"name": "alpha", "strength": 0.0} for _ in range(5)]

        with self.assertRaisesRegex(LoRAError, "4"):
            normalize_lora_requests(requests, self.catalog)

    def test_rejects_duplicate_names(self):
        with self.assertRaises(LoRAError):
            normalize_lora_requests(
                [{"name": "alpha", "strength": 0.0}, {"name": "alpha"}],
                self.catalog,
            )

    def test_unknown_name_error_does_not_expose_catalog_path(self):
        with self.assertRaises(LoRAError) as raised:
            normalize_lora_requests([{"name": "unknown"}], self.catalog)

        self.assertEqual(str(raised.exception), "Unknown LoRA: unknown")
        self.assertNotIn(str(self.root), str(raised.exception))

    def test_zero_strength_still_resolves_the_name(self):
        with self.assertRaisesRegex(LoRAError, "Unknown LoRA: unknown"):
            normalize_lora_requests(
                [{"name": "unknown", "strength": 0.0}],
                self.catalog,
            )

    def test_rejects_path_components_and_safetensors_suffix(self):
        for name in (
            "nested/alpha",
            "nested\\alpha",
            ".",
            "..",
            "alpha.safetensors",
        ):
            with self.subTest(name=name), self.assertRaises(LoRAError):
                normalize_lora_requests([{"name": name}], self.catalog)

    def test_rejects_extra_object_fields(self):
        with self.assertRaises(LoRAError):
            normalize_lora_requests(
                [{"name": "alpha", "strength": 1.0, "extra": True}],
                self.catalog,
            )

    def test_rejects_invalid_strengths(self):
        for strength in (True, False, math.nan, math.inf, -math.inf, -0.1, 2.1):
            with self.subTest(strength=strength), self.assertRaises(LoRAError):
                normalize_lora_requests(
                    [{"name": "alpha", "strength": strength}],
                    self.catalog,
                )

    def test_rejects_oversized_integer_strength_with_safe_error(self):
        lora_path = self.root / "lora_a.safetensors"
        lora_path.touch()
        catalog = LoRACatalog.scan(self.root)

        with self.assertRaises(LoRAError):
            normalize_lora_requests(
                [{"name": "lora_a", "strength": 10**400}],
                catalog,
            )

    def test_accepts_strength_boundaries(self):
        self.assertEqual(
            normalize_lora_requests(
                [{"name": "alpha", "strength": 0}, {"name": "beta", "strength": 2}],
                self.catalog,
            )[0].as_dict(),
            {"name": "beta", "strength": 2.0},
        )

    def test_rejects_non_list_input(self):
        for raw in ({}, "alpha", 1):
            with self.subTest(raw=raw), self.assertRaises(LoRAError):
                normalize_lora_requests(raw, self.catalog)

    def test_rejects_non_object_list_items(self):
        for item in ("alpha", 1, None, []):
            with self.subTest(item=item), self.assertRaises(LoRAError):
                normalize_lora_requests([item], self.catalog)

    def test_rejects_missing_or_invalid_names(self):
        for item in ({}, {"strength": 1.0}, {"name": ""}, {"name": 1}):
            with self.subTest(item=item), self.assertRaises(LoRAError):
                normalize_lora_requests([item], self.catalog)


if __name__ == "__main__":
    unittest.main()
