"""Fetch pinned data files only; publish completeness after checking file sizes."""

import json
from contextlib import contextmanager
from pathlib import Path
import sys

from shared.model_download import ModelDownloader
from shared.prefetch import PrefetchSession
from .config import (MODEL_ID, MODEL_REVISION, LORA_ID, LORA_REVISION, LORA_FILENAME,
                     EDIT_ID, EDIT_REVISION, EDIT_FILES)

MODEL_MANIFEST = '.moss-model.json'
LORA_MANIFEST = '.moss-lora.json'
EDIT_MANIFEST = '.moss-edit.json'
COMPONENTS = {'transformer': ('transformer',),
              'text_encoder': ('text_encoder', 'tokenizer'), 'vae': ('vae', 'scheduler')}


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


def _download(root, *, repo, revision, manifest, select, hf_home=None, token=None, publisher=None, required=()):
    root = Path(root)
    previous = _manifest(root, manifest, repo, revision)
    if previous and set(required) <= {item['path'] for item in previous['files']} and all(
        _matches(root, item) for item in previous['files']
    ):
        if publisher:
            for component in COMPONENTS:
                publisher.ready(component, root / component)
        return
    with ModelDownloader(root, hf_home=hf_home, token=token) as downloads:
        files = [{'path': entry.rfilename, 'size': entry.size}
                 for entry in remote_files(repo, revision, token) if select(entry.rfilename)]
        if not files or any(type(item['size']) is not int or item['size'] <= 0 for item in files):
            raise ValueError(f'Missing file sizes for pinned snapshot {repo}@{revision}')
        if not set(required) <= {item['path'] for item in files}:
            raise ValueError(f'Missing required files for pinned snapshot {repo}@{revision}')
        known = {item['path']: item for item in previous['files']} if previous else {}
        groups = COMPONENTS.items() if publisher else [(None, {item['path'].split('/')[0] for item in files})]
        for component, prefixes in groups:
            selected = [item for item in files if item['path'].split('/')[0] in prefixes]
            if not selected:
                raise ValueError(f'Missing MOSS component: {component}')
            component_manifest = f'.moss-{component}.json'
            if component is not None:
                saved = _manifest(root, component_manifest, repo, revision)
                if saved:
                    known.update({item['path']: item for item in saved['files']})
            for item in selected:
                if known.get(item['path']) == item and _matches(root, item):
                    continue
                path = root / item['path']
                # A truncated local-dir file with valid Hub metadata can otherwise be reused.
                if path.is_file() or path.is_symlink():
                    path.unlink()
                downloads.file(repo, item['path'], revision=revision)
                if not _matches(root, item):
                    raise ValueError(f'Incomplete MOSS file: {path}')
            if publisher:
                _write_manifest(root, component_manifest, repo, revision, selected)
                publisher.ready(component, root / component)
        _write_manifest(root, manifest, repo, revision, files)


def _write_manifest(root, name, repo, revision, files):
    pending = root / (name + '.tmp')
    pending.write_text(json.dumps({'repo': repo, 'revision': revision, 'files': files}, indent=2))
    pending.replace(root / name)


def download_model(root, *, hf_home=None, token=None, publisher=None):
    def select(path):
        return path.split('/')[0] in {'transformer', 'text_encoder', 'tokenizer', 'vae', 'scheduler'} and (
            path.endswith(('.json', '.safetensors', '.jinja')))
    _download(root, repo=MODEL_ID, revision=MODEL_REVISION, manifest=MODEL_MANIFEST,
              select=select, hf_home=hf_home, token=token, publisher=publisher)


def ensure_turbo_lora(root, *, hf_home=None, token=None):
    directory = Path(root) / 'turbo'
    _download(directory, repo=LORA_ID, revision=LORA_REVISION, manifest=LORA_MANIFEST,
              select=lambda path: path == LORA_FILENAME, hf_home=hf_home, token=token)
    return directory / LORA_FILENAME


def turbo_ready(root):
    root = Path(root) / 'turbo'
    data = _manifest(root, LORA_MANIFEST, LORA_ID, LORA_REVISION)
    return data is not None and all(_matches(root, item) for item in data['files'])


def ensure_edit_lora(root, *, hf_home=None, token=None):
    directory = Path(root) / 'edit'
    _download(directory, repo=EDIT_ID, revision=EDIT_REVISION, manifest=EDIT_MANIFEST,
              select=lambda path: path in EDIT_FILES, required=EDIT_FILES,
              hf_home=hf_home, token=token)
    return directory


def edit_ready(root):
    root = Path(root) / 'edit'
    data = _manifest(root, EDIT_MANIFEST, EDIT_ID, EDIT_REVISION)
    return (data is not None and {item['path'] for item in data['files']} == set(EDIT_FILES)
            and all(_matches(root, item) for item in data['files']))


@contextmanager
def prefetch_model(root, *, turbo=False, lora_only=False, edit=False):
    if turbo and edit:
        raise ValueError('MOSS Turbo and Edit adapters cannot be combined')
    if (lora_only or snapshot_ready(root)) and (not turbo or turbo_ready(root)) and (not edit or edit_ready(root)):
        yield None
        return
    command = [sys.executable, '-u', str(Path(__file__).parents[1] / 'download_weights.py'), '--output', str(root)]
    if lora_only:
        command.append('--lora-only')
    if edit:
        command.append('--edit')
    elif turbo and not lora_only:
        command.append('--turbo')
    with PrefetchSession(command) as pending:
        yield pending
