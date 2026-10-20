"""Pinned local cache behavior; Hub transfers use controlled fixture files."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def fixtures():
    return {
        'config.json': json.dumps({'model_type': 'qwen3_5', 'architectures': ['Qwen3_5ForConditionalGeneration']}).encode(),
        'processor_config.json': b'{}', 'tokenizer_config.json': b'{}',
        'tokenizer.json': b'{}', 'chat_template.jinja': b'caption template',
        'generation_config.json': b'{}',
        'model.safetensors.index.json': json.dumps({'weight_map': {
            'first': 'model-00001-of-00002.safetensors',
            'last': 'model-00002-of-00002.safetensors',
        }}).encode(),
        'model-00001-of-00002.safetensors': b'first weights',
        'model-00002-of-00002.safetensors': b'last weights',
    }


def store():
    return importlib.import_module('model_store')


def hub(files, transfers):
    class API:
        def __init__(self, **kwargs):
            pass

        def model_info(self, repo, **kwargs):
            assert kwargs['revision'] == 'ead05a2ba36c2e68cc03669414210bde75988185'
            return SimpleNamespace(siblings=[SimpleNamespace(rfilename=k, size=len(v)) for k, v in files.items()])

    def download(*, filename, local_dir, revision, **kwargs):
        assert revision == 'ead05a2ba36c2e68cc03669414210bde75988185'
        transfers.append(filename)
        path = Path(local_dir) / filename
        path.write_bytes(files[filename])
        return str(path)

    return patch.multiple('huggingface_hub', HfApi=API, hf_hub_download=download)


def test_metadata_ready_before_weights_and_shards_published_individually(tmp_path):
    transfers, ready = [], []

    class Publisher:
        def ready(self, name, path):
            ready.append(name)
            if name == 'processor':
                assert not list(tmp_path.glob('*.safetensors'))
            if name == 'shard_00000':
                assert not (tmp_path / 'model-00002-of-00002.safetensors').exists()

    with hub(fixtures(), transfers):
        store().download_model(tmp_path, publisher=Publisher())
    assert ready == ['processor', 'shard_00000', 'shard_00001', 'all']
    assert store().snapshot_ready(tmp_path)


def test_warm_cache_and_truncated_shard_repair(tmp_path):
    transfers = []
    with hub(fixtures(), transfers):
        store().download_model(tmp_path)
        transfers.clear()
        store().download_model(tmp_path)
        assert transfers == []
        (tmp_path / 'model-00002-of-00002.safetensors').write_bytes(b'partial')
        assert not store().snapshot_ready(tmp_path)
        store().download_model(tmp_path)
    assert transfers == ['model-00002-of-00002.safetensors']
    assert store().snapshot_ready(tmp_path)


def test_retry_reuses_first_shard_after_later_download_failure(tmp_path):
    import pytest

    files, transfers = fixtures(), []
    original = files['model-00002-of-00002.safetensors']
    with hub(files, transfers):
        files['model-00002-of-00002.safetensors'] = b'broken'
        # Metadata reports a complete file but the transfer returns a partial one.
        from huggingface_hub import hf_hub_download

        def truncated(**kwargs):
            result = hf_hub_download(**kwargs)
            if kwargs['filename'] == 'model-00002-of-00002.safetensors':
                Path(result).write_bytes(b'x')
            return result

        with patch('huggingface_hub.hf_hub_download', truncated):
            with pytest.raises(ValueError, match='Incomplete'):
                store().download_model(tmp_path)
        assert not store().snapshot_ready(tmp_path)
    files['model-00002-of-00002.safetensors'] = original
    transfers.clear()
    with hub(files, transfers):
        store().download_model(tmp_path)
    assert transfers == ['model-00002-of-00002.safetensors']


def test_interrupted_revision_change_never_blesses_old_same_sized_files(tmp_path):
    import pytest

    files, transfers = fixtures(), []
    with hub(files, transfers):
        store().download_model(tmp_path)
    marker = tmp_path / '.caption-model.json'
    data = json.loads(marker.read_text())
    data['revision'] = 'old revision'
    marker.write_text(json.dumps(data))
    with hub(files, transfers), patch('huggingface_hub.hf_hub_download', side_effect=OSError('interrupted')):
        with pytest.raises(OSError, match='interrupted'):
            store().download_model(tmp_path)
    transfers.clear()
    with hub(files, transfers):
        store().download_model(tmp_path)
    assert set(transfers) == set(files)

