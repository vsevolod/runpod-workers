"""Pinned Gliese files on local disk, coordinated by the shared downloader."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sys

from shared.model_download import ModelDownloader
from shared.prefetch import PrefetchSession

MODEL_ID = 'prithivMLmods/Gliese-Qwen3.5-9B-Abliterated-Caption'
MODEL_REVISION = 'ead05a2ba36c2e68cc03669414210bde75988185'
DEFAULT_MODEL_DIR = '/models/gliese_qwen35_9b'
MANIFEST = '.caption-model.json'
METADATA = (
    'config.json', 'processor_config.json', 'tokenizer_config.json',
    'tokenizer.json', 'chat_template.jinja', 'generation_config.json',
    'model.safetensors.index.json',
)
SHARD = re.compile(r'model-\d{5}-of-\d{5}\.safetensors\Z')


def model_directory():
    return Path(os.environ.get('MODEL_DIR') or DEFAULT_MODEL_DIR)


def _manifest(root):
    try:
        data = json.loads((root / MANIFEST).read_text())
        if data['repo'] != MODEL_ID or data['revision'] != MODEL_REVISION:
            return None
        files = data['files']
        if not files or not set(METADATA).issubset(files):
            return None
        if any(name not in METADATA and not SHARD.fullmatch(name) for name in files):
            return None
        if any(type(size) is not int or size <= 0 for size in files.values()):
            return None
        if not isinstance(data['completed'], list) or not set(data['completed']).issubset(files):
            return None
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _matches(root, name, size):
    path = root / name
    return not path.is_symlink() and path.is_file() and path.stat().st_size == size


def shard_names(root):
    index = json.loads((Path(root) / 'model.safetensors.index.json').read_text())
    names = sorted(set(index['weight_map'].values()))
    if not names or any(not SHARD.fullmatch(name) for name in names):
        raise ValueError('Invalid Gliese checkpoint shard index')
    return names


def snapshot_ready(root):
    root = Path(root)
    saved = _manifest(root)
    if saved is None:
        return False
    try:
        return (set(saved['completed']) == set(saved['files'])
                and all(_matches(root, name, size) for name, size in saved['files'].items())
                and set(shard_names(root)) == {name for name in saved['files'] if SHARD.fullmatch(name)})
    except (OSError, ValueError, KeyError, TypeError):
        return False


def download_model(root, *, hf_home=None, token=None, publisher=None):
    root = Path(root)
    previous = _manifest(root)
    if snapshot_ready(root):
        if publisher:
            publisher.ready('processor', root)
            for index, name in enumerate(shard_names(root)):
                publisher.ready(f'shard_{index:05}', root / name)
            publisher.ready('all', root)
        return
    from huggingface_hub import HfApi

    entries = HfApi(token=token).model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True).siblings
    files = {entry.rfilename: entry.size for entry in entries
             if entry.rfilename in METADATA or SHARD.fullmatch(entry.rfilename)}
    if (not set(METADATA).issubset(files) or not any(SHARD.fullmatch(name) for name in files)
            or any(type(size) is not int or size <= 0 for size in files.values())):
        raise ValueError('Incomplete metadata for pinned Gliese checkpoint')
    root.mkdir(parents=True, exist_ok=True)
    known = {name: previous['files'][name] for name in previous['completed']} if previous else {}
    saved = {'repo': MODEL_ID, 'revision': MODEL_REVISION, 'files': files,
             'completed': [name for name, size in files.items()
                           if known.get(name) == size and _matches(root, name, size)]}

    def save_progress():
        pending = root / (MANIFEST + '.tmp')
        pending.write_text(json.dumps(saved, indent=2))
        pending.replace(root / MANIFEST)

    # Record successful transfers separately from expected sizes: otherwise a
    # crash could bless same-sized files left over from a different revision.
    save_progress()
    with ModelDownloader(root, hf_home=hf_home, token=token) as downloads:
        def fetch(name):
            if name in saved['completed']:
                return
            path = root / name
            if path.is_file() or path.is_symlink():
                path.unlink()
            downloads.file(MODEL_ID, name, revision=MODEL_REVISION)
            if not _matches(root, name, files[name]):
                raise ValueError(f'Incomplete Gliese file: {name}')
            saved['completed'].append(name)
            save_progress()

        for name in METADATA:
            fetch(name)
        names = shard_names(root)
        if set(names) != {name for name in files if SHARD.fullmatch(name)}:
            raise ValueError('Gliese shard index does not match pinned file metadata')
        if publisher:
            publisher.ready('processor', root)
        for index, name in enumerate(names):
            fetch(name)
            if publisher:
                publisher.ready(f'shard_{index:05}', root / name)
    if publisher:
        publisher.ready('all', root)


@contextmanager
def prefetch_model(root):
    if snapshot_ready(root):
        yield None
        return
    command = [sys.executable, '-u', str(Path(__file__).with_name('download_weights.py')), '--output', str(root)]
    with PrefetchSession(command) as pending:
        yield pending
