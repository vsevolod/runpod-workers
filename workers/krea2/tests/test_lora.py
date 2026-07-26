import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

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
LoRALoader = _lora.LoRALoader
LoRASelection = _lora.LoRASelection
normalize_lora_requests = _lora.normalize_lora_requests


class _Node(nn.Module):
    pass


def _linear(in_features=3, out_features=2):
    return nn.Linear(in_features, out_features, bias=False)


class _TinyKreaModel(nn.Module):
    def __init__(self):
        super().__init__()
        block = _Node()
        block.attn = _Node()
        block.attn.wq = _linear()
        block.attn.wk = _linear()
        block.attn.wv = _linear()
        block.attn.gate = _linear()
        block.attn.wo = _linear()
        block.mlp = _Node()
        block.mlp.gate = _linear()
        block.mlp.up = _linear()
        block.mlp.down = _linear()
        self.blocks = nn.ModuleList([block])

        self.txtfusion = _Node()
        self.txtfusion.layerwise_blocks = nn.ModuleList([self._fusion_block()])
        self.txtfusion.refiner_blocks = nn.ModuleList([self._fusion_block()])
        self.txtfusion.projector = _linear()
        self.first = _linear()
        self.tmlp = nn.Sequential(_linear(), nn.Identity(), _linear())
        self.tproj = nn.Sequential(nn.Identity(), _linear())
        self.txtmlp = nn.Sequential(nn.Identity(), _linear(), nn.Identity(), _linear())
        self.last = _Node()
        self.last.linear = _linear()

    @staticmethod
    def _fusion_block():
        block = _Node()
        block.attn = _Node()
        block.attn.wq = _linear()
        block.attn.wk = _linear()
        block.attn.wv = _linear()
        block.attn.gate = _linear()
        block.attn.wo = _linear()
        block.mlp = _Node()
        block.mlp.gate = _linear()
        block.mlp.up = _linear()
        block.mlp.down = _linear()
        return block


def _selection(path, name="public-adapter", strength=0.75, type="lora"):
    return LoRASelection(name=name, strength=strength, path=path, type=type)


def _pair(base, *, rank=1, in_features=3, out_features=2, native=False):
    if native:
        return {
            f"{base}.lora_down.weight": torch.ones(rank, in_features),
            f"{base}.lora_up.weight": torch.ones(out_features, rank),
        }
    return {
        f"{base}.lora_A.weight": torch.ones(rank, in_features),
        f"{base}.lora_B.weight": torch.ones(out_features, rank),
    }


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
        self.assertEqual(selections[0].type, "lora")
        self.assertEqual(
            selections[0].as_dict(),
            {"name": "alpha", "strength": 1.0, "type": "lora"},
        )

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
            {"name": "beta", "strength": 2.0, "type": "lora"},
        )

    def test_defaults_type_to_lora(self):
        selections = normalize_lora_requests([{"name": "alpha"}], self.catalog)
        self.assertEqual(selections[0].type, "lora")
        self.assertEqual(selections[0].as_dict()["type"], "lora")

    def test_accepts_weight_diff_type_and_higher_strength(self):
        selections = normalize_lora_requests(
            [
                {
                    "name": "alpha",
                    "type": "weight_diff",
                    "strength": 4.0,
                }
            ],
            self.catalog,
        )
        self.assertEqual(
            selections[0].as_dict(),
            {"name": "alpha", "strength": 4.0, "type": "weight_diff"},
        )

    def test_weight_diff_accepts_strength_five(self):
        selections = normalize_lora_requests(
            [{"name": "alpha", "type": "weight_diff", "strength": 5.0}],
            self.catalog,
        )
        self.assertEqual(selections[0].strength, 5.0)

    def test_rejects_strength_above_ceiling_for_type(self):
        with self.assertRaises(LoRAError):
            normalize_lora_requests(
                [{"name": "alpha", "type": "lora", "strength": 4.0}],
                self.catalog,
            )
        with self.assertRaises(LoRAError):
            normalize_lora_requests(
                [{"name": "alpha", "type": "weight_diff", "strength": 5.1}],
                self.catalog,
            )

    def test_rejects_unknown_type(self):
        with self.assertRaisesRegex(LoRAError, "type"):
            normalize_lora_requests(
                [{"name": "alpha", "type": "loha"}],
                self.catalog,
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


class LoRALoaderTest(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_directory.name)
        self.model = _TinyKreaModel()
        self.loader = LoRALoader(self.model)

    def tearDown(self):
        self.temp_directory.cleanup()

    def _save(self, tensors, filename="private-file-name.safetensors"):
        path = self.root / filename
        save_file(tensors, path)
        return path

    def _assert_rejected(self, tensors, *, name="public-adapter", type="lora"):
        path = self._save(tensors)
        with self.assertRaises(LoRAError) as raised:
            self.loader.load((_selection(path, name=name, type=type),))
        message = str(raised.exception)
        self.assertIn(name, message)
        self.assertNotIn(str(path), message)
        self.assertNotIn(str(self.root), message)
        return message

    def test_loads_diffusers_pair_with_alpha_and_preserves_request_fields(self):
        base = "transformer.transformer_blocks.0.attn.to_q"
        tensors = _pair(base, rank=2)
        tensors[f"{base}.alpha"] = torch.tensor(6.0)
        path = self._save(tensors)

        prepared = self.loader.load((_selection(path, name="cinematic", strength=1.25),))

        self.assertIsInstance(prepared, tuple)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0].name, "cinematic")
        self.assertEqual(prepared[0].strength, 1.25)
        self.assertEqual(len(prepared[0].layers), 1)
        layer = prepared[0].layers[0]
        self.assertEqual(layer.target, "blocks.0.attn.wq")
        self.assertEqual(layer.scale, 3.0)
        torch.testing.assert_close(layer.down, tensors[f"{base}.lora_A.weight"])
        torch.testing.assert_close(layer.up, tensors[f"{base}.lora_B.weight"])

    def test_loads_native_pair_with_default_scale(self):
        base = "blocks.0.attn.wq"
        tensors = _pair(base, rank=2, native=True)
        path = self._save(tensors)

        prepared = self.loader.load((_selection(path),))

        self.assertEqual(prepared[0].layers[0].target, base)
        self.assertEqual(prepared[0].layers[0].scale, 1.0)

    def test_resolves_all_supported_diffusers_target_families_and_prefixes(self):
        mappings = (
            ("transformer_blocks.0.attn.to_q", "blocks.0.attn.wq"),
            ("transformer_blocks.0.attn.to_k", "blocks.0.attn.wk"),
            ("transformer_blocks.0.attn.to_v", "blocks.0.attn.wv"),
            ("transformer_blocks.0.attn.to_gate", "blocks.0.attn.gate"),
            ("transformer_blocks.0.attn.to_out.0", "blocks.0.attn.wo"),
            ("transformer_blocks.0.attn.to_out", "blocks.0.attn.wo"),
            ("transformer_blocks.0.ff.gate", "blocks.0.mlp.gate"),
            ("transformer_blocks.0.ff.up", "blocks.0.mlp.up"),
            ("transformer_blocks.0.ff.down", "blocks.0.mlp.down"),
            (
                "text_fusion.layerwise_blocks.0.attn.to_q",
                "txtfusion.layerwise_blocks.0.attn.wq",
            ),
            (
                "text_fusion.refiner_blocks.0.ff.down",
                "txtfusion.refiner_blocks.0.mlp.down",
            ),
            ("img_in", "first"),
            ("time_embed.linear_1", "tmlp.0"),
            ("time_embed.linear_2", "tmlp.2"),
            ("time_mod_proj", "tproj.1"),
            ("txt_in.linear_1", "txtmlp.1"),
            ("txt_in.linear_2", "txtmlp.3"),
            ("text_fusion.projector", "txtfusion.projector"),
            ("final_layer.linear", "last.linear"),
        )
        prefixes = ("", "transformer.", "diffusion_model.")

        for source, expected in mappings:
            for prefix in prefixes:
                with self.subTest(source=source, prefix=prefix):
                    base = f"{prefix}{source}"
                    path = self._save(_pair(base), filename="mapping.safetensors")

                    prepared = self.loader.load((_selection(path),))

                    self.assertEqual(prepared[0].layers[0].target, expected)

    def test_returns_multiple_prepared_adapters_in_request_order(self):
        first_path = self._save(
            _pair("blocks.0.attn.wq", native=True),
            filename="first.safetensors",
        )
        second_path = self._save(
            _pair("blocks.0.attn.wk", native=True),
            filename="second.safetensors",
        )

        prepared = self.loader.load(
            (
                _selection(first_path, name="first", strength=0.5),
                _selection(second_path, name="second", strength=1.5),
            )
        )

        self.assertEqual(tuple(item.name for item in prepared), ("first", "second"))
        self.assertEqual(tuple(item.strength for item in prepared), (0.5, 1.5))

    def test_layers_are_sorted_by_resolved_target(self):
        tensors = {}
        tensors.update(_pair("blocks.0.attn.wv", native=True))
        tensors.update(_pair("blocks.0.attn.wq", native=True))
        path = self._save(tensors)

        prepared = self.loader.load((_selection(path),))

        self.assertEqual(
            tuple(layer.target for layer in prepared[0].layers),
            ("blocks.0.attn.wq", "blocks.0.attn.wv"),
        )

    def test_rejects_unknown_target(self):
        self._assert_rejected(_pair("transformer.unknown_target"))

    def test_rejects_incomplete_pair(self):
        self._assert_rejected(
            {"blocks.0.attn.wq.lora_down.weight": torch.ones(1, 3)}
        )

    def test_rejects_duplicate_half_across_suffix_conventions(self):
        tensors = _pair("blocks.0.attn.wq", native=True)
        tensors["blocks.0.attn.wq.lora_A.weight"] = torch.ones(1, 3)
        self._assert_rejected(tensors)

    def test_rejects_pairs_mixed_across_suffix_conventions(self):
        mixed_pairs = (
            {
                "blocks.0.attn.wq.lora_A.weight": torch.ones(1, 3),
                "blocks.0.attn.wq.lora_up.weight": torch.ones(2, 1),
            },
            {
                "blocks.0.attn.wq.lora_down.weight": torch.ones(1, 3),
                "blocks.0.attn.wq.lora_B.weight": torch.ones(2, 1),
            },
        )
        for tensors in mixed_pairs:
            with self.subTest(keys=tuple(tensors)):
                self._assert_rejected(tensors)

    def test_rejects_non_2d_weights(self):
        for key, value in (
            ("blocks.0.attn.wq.lora_down.weight", torch.ones(3)),
            ("blocks.0.attn.wq.lora_up.weight", torch.ones(2)),
        ):
            with self.subTest(key=key):
                tensors = _pair("blocks.0.attn.wq", native=True)
                tensors[key] = value
                self._assert_rejected(tensors)

    def test_rejects_non_floating_point_weights(self):
        base = "blocks.0.attn.wq"
        for dtype in (torch.int64, torch.bool):
            for suffix, shape in (
                (".lora_A.weight", (1, 3)),
                (".lora_B.weight", (2, 1)),
            ):
                with self.subTest(dtype=dtype, suffix=suffix):
                    tensors = _pair(base)
                    tensors[f"{base}{suffix}"] = torch.ones(shape, dtype=dtype)
                    self._assert_rejected(tensors)

    def test_rejects_zero_rank_weights(self):
        self._assert_rejected(
            _pair("blocks.0.attn.wq", rank=0, native=True)
        )

    def test_rejects_dimension_or_rank_mismatch(self):
        invalid_pairs = (
            {
                "blocks.0.attn.wq.lora_down.weight": torch.ones(1, 4),
                "blocks.0.attn.wq.lora_up.weight": torch.ones(2, 1),
            },
            {
                "blocks.0.attn.wq.lora_down.weight": torch.ones(1, 3),
                "blocks.0.attn.wq.lora_up.weight": torch.ones(3, 1),
            },
            {
                "blocks.0.attn.wq.lora_down.weight": torch.ones(1, 3),
                "blocks.0.attn.wq.lora_up.weight": torch.ones(2, 2),
            },
        )
        for tensors in invalid_pairs:
            with self.subTest(shapes=tuple(t.shape for t in tensors.values())):
                self._assert_rejected(tensors)

    def test_rejects_non_scalar_or_non_finite_alpha(self):
        base = "blocks.0.attn.wq"
        for alpha in (torch.ones(2), torch.tensor(float("nan")), torch.tensor(float("inf"))):
            with self.subTest(alpha=alpha):
                tensors = _pair(base, native=True)
                tensors[f"{base}.alpha"] = alpha
                self._assert_rejected(tensors)

    def test_rejects_boolean_alpha(self):
        base = "blocks.0.attn.wq"
        tensors = _pair(base, native=True)
        tensors[f"{base}.alpha"] = torch.tensor(True)

        self._assert_rejected(tensors)

    def test_accepts_integral_alpha(self):
        base = "blocks.0.attn.wq"
        tensors = _pair(base, rank=2, native=True)
        tensors[f"{base}.alpha"] = torch.tensor(6, dtype=torch.int64)
        path = self._save(tensors)

        prepared = self.loader.load((_selection(path),))

        self.assertEqual(prepared[0].layers[0].scale, 3.0)

    def test_rejects_unmatched_alpha(self):
        self._assert_rejected({"blocks.0.attn.wq.alpha": torch.tensor(1.0)})

    def test_rejects_no_layers(self):
        self._assert_rejected({})

    def test_rejects_any_unconsumed_tensor_key(self):
        tensors = _pair("blocks.0.attn.wq", native=True)
        tensors["metadata.unexpected"] = torch.tensor(1.0)
        self._assert_rejected(tensors)

    def test_file_controlled_keys_are_not_exposed_in_errors(self):
        secret_key = "/private/volume/secret.tensor"
        long_key = f"private-{'x' * 10_000}.tensor"
        for key in (secret_key, long_key):
            with self.subTest(key_length=len(key)):
                tensors = _pair("blocks.0.attn.wq", native=True)
                tensors[key] = torch.tensor(1.0)

                message = self._assert_rejected(tensors)

                self.assertNotIn(key, message)
                self.assertNotIn("/private/volume", message)
                self.assertLess(len(message), 256)

    def test_file_controlled_target_bases_are_not_exposed_in_errors(self):
        secret_base = "/private/volume/secret-target"
        malformed_files = (
            _pair(secret_base, native=True),
            {f"{secret_base}.alpha": torch.tensor(1.0)},
            {f"{secret_base}.lora_down.weight": torch.ones(1, 3)},
        )
        for tensors in malformed_files:
            with self.subTest(keys=tuple(tensors)):
                message = self._assert_rejected(tensors)

                self.assertNotIn(secret_base, message)
                self.assertNotIn("/private/volume", message)
                self.assertLess(len(message), 256)

    def test_duplicate_resolved_target_error_does_not_expose_tensor_targets(self):
        first_base = "transformer_blocks.0.attn.to_out"
        second_base = "transformer_blocks.0.attn.to_out.0"
        tensors = _pair(first_base)
        tensors.update(_pair(second_base))

        message = self._assert_rejected(tensors)

        self.assertNotIn(first_base, message)
        self.assertNotIn(second_base, message)
        self.assertNotIn("blocks.0.attn.wo", message)
        self.assertLess(len(message), 256)

    def test_wraps_corrupted_safetensors_error_without_exposing_path(self):
        path = self.root / "secret-corrupted-file.safetensors"
        path.write_bytes(b"not safetensors")

        with self.assertRaises(LoRAError) as raised:
            self.loader.load((_selection(path, name="public-adapter"),))

        message = str(raised.exception)
        self.assertIn("public-adapter", message)
        self.assertNotIn(str(path), message)
        self.assertNotIn(str(self.root), message)

    def test_loads_weight_diff_projector_with_diffusion_model_prefix(self):
        delta = torch.zeros(2, 3)
        delta[0, 1] = -0.5117
        delta[0, 2] = -0.8906
        tensors = {"diffusion_model.txtfusion.projector.diff": delta}
        path = self._save(tensors)

        prepared = self.loader.load(
            (
                _selection(
                    path,
                    name="fedor_bypass",
                    strength=4.0,
                    type="weight_diff",
                ),
            )
        )

        self.assertEqual(prepared[0].name, "fedor_bypass")
        self.assertEqual(prepared[0].type, "weight_diff")
        self.assertEqual(prepared[0].strength, 4.0)
        self.assertEqual(len(prepared[0].layers), 1)
        layer = prepared[0].layers[0]
        self.assertEqual(layer.target, "txtfusion.projector")
        torch.testing.assert_close(layer.delta, delta)

    def test_loads_weight_diff_via_text_fusion_projector_alias(self):
        delta = torch.ones(2, 3)
        path = self._save({"text_fusion.projector.diff": delta})

        prepared = self.loader.load(
            (_selection(path, type="weight_diff"),)
        )

        self.assertEqual(prepared[0].layers[0].target, "txtfusion.projector")

    def test_rejects_weight_diff_shape_mismatch(self):
        tensors = {
            "diffusion_model.txtfusion.projector.diff": torch.ones(1, 12)
        }
        self._assert_rejected(tensors, type="weight_diff")

    def test_rejects_weight_diff_when_type_is_lora(self):
        tensors = {
            "diffusion_model.txtfusion.projector.diff": torch.ones(2, 3)
        }
        self._assert_rejected(tensors, type="lora")

    def test_rejects_lora_pairs_when_type_is_weight_diff(self):
        self._assert_rejected(
            _pair("blocks.0.attn.wq", native=True),
            type="weight_diff",
        )

    def test_prepared_lora_type_defaults_on_rank_load(self):
        path = self._save(_pair("blocks.0.attn.wq", native=True))
        prepared = self.loader.load((_selection(path),))
        self.assertEqual(prepared[0].type, "lora")


class LoRASchemaTests(unittest.TestCase):
    def test_loras_is_an_optional_list_with_empty_default(self):
        # Import from the worker package root (PYTHONPATH=workers/krea2).
        from schemas import INPUT_SCHEMA

        self.assertIs(INPUT_SCHEMA["loras"]["type"], list)
        self.assertFalse(INPUT_SCHEMA["loras"]["required"])
        self.assertEqual(INPUT_SCHEMA["loras"]["default"], [])


if __name__ == "__main__":
    unittest.main()
