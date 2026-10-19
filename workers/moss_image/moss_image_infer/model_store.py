"""Fetch pinned data files only; publish completeness after checking file sizes."""

import json
from pathlib import Path

from shared.model_download import ModelDownloader
from .config import MODEL_ID, MODEL_REVISION, LORA_ID, LORA_REVISION, LORA_FILENAME

MODEL_MANIFEST = '.moss-model.json'
LORA_MANIFEST = '.moss-lora.json'


def remote_files(repo, revision, token):
    from huggingface_hub import HfApi
    return HfApi(token=token).model_info(repo, revision=revision, files_metadata=True).siblings


def _manifest(root, name, repo, revision):
    try:
        data = json.loads((root / name).read_text())
        if data['repo'] != repo or data['revision'] != revision or not data['files']:
            return None
        for item in data['files']:
            path = Path(item['path'])
            if path.is_absolute() or '..' in path.parts or type(item['size']) is not int or item['size'] <= 0:
                return None
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _matches(root, item):
    path = root / item['path']
    try:
        return not path.is_symlink() and path.is_file() and path.stat().st_size == item['size']
    except OSError:
        return False


def snapshot_ready(root):
    root = Path(root)
    data = _manifest(root, MODEL_MANIFEST, MODEL_ID, MODEL_REVISION)
    return data is not None and all(_matches(root, item) for item in data['files'])


def _download(root, *, repo, revision, manifest, select, hf_home=None, token=None):
    root = Path(root)
    previous = _manifest(root, manifest, repo, revision)
    if previous and all(_matches(root, item) for item in previous['files']):
        return
    with ModelDownloader(root, hf_home=hf_home, token=token) as downloads:
        files = [{'path': entry.rfilename, 'size': entry.size}
                 for entry in remote_files(repo, revision, token) if select(entry.rfilename)]
        if not files or any(type(item['size']) is not int or item['size'] <= 0 for item in files):
            raise ValueError(f'Missing file sizes for pinned snapshot {repo}@{revision}')
        known = {item['path']: item for item in previous['files']} if previous else {}
        for item in files:
            if known.get(item['path']) == item and _matches(root, item):
                continue
            path = root / item['path']
            # A truncated local-dir file with valid Hub metadata can otherwise be reused.
            if path.is_file() or path.is_symlink():
                path.unlink()
            downloads.file(repo, item['path'], revision=revision)
            if not _matches(root, item):
                raise ValueError(f'Incomplete MOSS file: {path}')
        pending = root / (manifest + '.tmp')
        pending.write_text(json.dumps({'repo': repo, 'revision': revision, 'files': files}, indent=2))
        pending.replace(root / manifest)


def download_model(root, *, hf_home=None, token=None):
    def select(path):
        return path.split('/')[0] in {'transformer', 'text_encoder', 'tokenizer', 'vae', 'scheduler'} and (
            path.endswith(('.json', '.safetensors', '.jinja')))
    _download(root, repo=MODEL_ID, revision=MODEL_REVISION, manifest=MODEL_MANIFEST,
              select=select, hf_home=hf_home, token=token)


def ensure_turbo_lora(root, *, hf_home=None, token=None):
    directory = Path(root) / 'turbo'
    _download(directory, repo=LORA_ID, revision=LORA_REVISION, manifest=LORA_MANIFEST,
              select=lambda path: path == LORA_FILENAME, hf_home=hf_home, token=token)
    return directory / LORA_FILENAME
