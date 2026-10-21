"""Pinned files and atomic progress, using the common CPU downloader lifecycle."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys

from shared.model_download import ModelDownloader
from shared.prefetch import PrefetchSession
from .config import CHECKPOINTS

MANIFEST = '.fibo-model.json'
COMPONENTS = {
    'transformer': ('model_index.json', 'transformer/'),
    'text_encoder': ('text_encoder/', 'processor/', 'fonts/'),
    'vae': ('vae/', 'scheduler/'),
}
REQUIRED = {
    'model_index.json', 'transformer/config.json',
    'transformer/diffusion_pytorch_model.safetensors.index.json',
    'text_encoder/config.json', 'text_encoder/generation_config.json', 'text_encoder/model.safetensors',
    'processor/processor_config.json', 'processor/tokenizer_config.json', 'processor/tokenizer.json',
    'processor/chat_template.jinja', 'fonts/Aileron-Regular.otf', 'fonts/DejaVuSans.ttf',
    'fonts/LICENSE-Aileron.txt', 'fonts/LICENSE-DejaVu.txt',
    'vae/config.json', 'vae/diffusion_pytorch_model.safetensors', 'scheduler/scheduler_config.json',
}


def _selected(name):
    path = Path(name)
    return (not path.is_absolute() and '..' not in path.parts
            and any(name == prefix or name.startswith(prefix) for prefixes in COMPONENTS.values() for prefix in prefixes)
            and name.endswith(('.json', '.safetensors', '.jinja', '.otf', '.ttf', '.txt')))


def _matches(root, name, size):
    path = root / name
    try:
        return not path.is_symlink() and path.is_file() and path.stat().st_size == size
    except OSError:
        return False


def _manifest(root, variant):
    try:
        data = json.loads((root / MANIFEST).read_text())
        if (data['repo'], data['revision']) != CHECKPOINTS[variant]:
            return None
        files = data['files']
        if not REQUIRED.issubset(files) or any(not _selected(name) for name in files):
            return None
        if any(type(size) is not int or size <= 0 for size in files.values()):
            return None
        if not isinstance(data['completed'], list) or not set(data['completed']).issubset(files):
            return None
        return data
    except (OSError, ValueError, TypeError, KeyError):
        return None


def snapshot_ready(root, *, variant):
    root = Path(root)
    data = _manifest(root, variant)
    return data is not None and set(data['completed']) == set(data['files']) and all(
        _matches(root, name, size) for name, size in data['files'].items())


def download_model(root, *, variant, hf_home=None, token=None, publisher=None):
    root = Path(root)
    repo, revision = CHECKPOINTS[variant]
    previous = _manifest(root, variant)
    if snapshot_ready(root, variant=variant):
        if publisher:
            for name in (*COMPONENTS, 'all'):
                publisher.ready(name, root)
        return
    from huggingface_hub import HfApi

    entries = HfApi(token=token).model_info(repo, revision=revision, files_metadata=True).siblings
    files = {entry.rfilename: entry.size for entry in entries if _selected(entry.rfilename)}
    if not REQUIRED.issubset(files) or any(type(size) is not int or size <= 0 for size in files.values()):
        raise ValueError(f'Incomplete pinned Fibo snapshot metadata: {repo}@{revision}')
    if not any(name.startswith('transformer/') and name.endswith('.safetensors') for name in files):
        raise ValueError('Missing Fibo transformer weights')
    known = {name: previous['files'][name] for name in previous['completed']} if previous else {}
    saved = {'repo': repo, 'revision': revision, 'files': files,
             'completed': [name for name, size in files.items()
                           if known.get(name) == size and _matches(root, name, size)]}
    root.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = root / (MANIFEST + '.tmp')
        temporary.write_text(json.dumps(saved, indent=2))
        temporary.replace(root / MANIFEST)

    save()  # Never bless old same-sized files after an interrupted revision change.
    with ModelDownloader(root, hf_home=hf_home, token=token) as downloads:
        for component, prefixes in COMPONENTS.items():
            selected = sorted(name for name in files if any(name == p or name.startswith(p) for p in prefixes))
            for name in selected:
                if name in saved['completed']:
                    continue
                path = root / name
                if path.is_file() or path.is_symlink():
                    path.unlink()
                downloads.file(repo, name, revision=revision)
                if not _matches(root, name, files[name]):
                    raise ValueError(f'Incomplete Fibo file: {name}')
                saved['completed'].append(name)
                save()
            if publisher:
                publisher.ready(component, root / component)
    if publisher:
        publisher.ready('all', root)


@contextmanager
def prefetch_model(root, *, variant):
    if snapshot_ready(root, variant=variant):
        yield None
        return
    command = [sys.executable, '-u', str(Path(__file__).parents[1] / 'download_weights.py'),
               '--output', str(root), '--variant', variant]
    with PrefetchSession(command) as pending:
        yield pending
