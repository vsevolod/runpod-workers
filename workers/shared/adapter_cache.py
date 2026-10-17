"""Copy only requested adapters to local disk before safetensors maps them."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import stat
import tempfile
import threading
import time

logger = logging.getLogger(__name__)


class AdapterCache:
    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self._lock = threading.Lock()

    @contextmanager
    def prefetch(self, paths):
        """Start I/O now; join on exit even if inference fails before activation."""
        paths = tuple(Path(path) for path in paths)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='lora-download') as executor:
            yield executor.submit(lambda: tuple(self._ensure(path) for path in paths))

    def _ensure(self, source: Path) -> Path:
        # Serialize publication and validation across concurrent requests.
        with self._lock:
            before = source.lstat()
            if not stat.S_ISREG(before.st_mode):
                raise OSError(f'LoRA source is not a regular file: {source.name}')
            signature = [str(source.absolute()), before.st_size, before.st_mtime_ns, before.st_ctime_ns]
            identity = hashlib.sha256(str(source.absolute()).encode()).hexdigest()[:24]
            folder = self.directory / identity
            destination = folder / source.name
            marker = folder / 'complete.json'
            try:
                saved = json.loads(marker.read_text())
                local = destination.lstat()
                if (stat.S_ISREG(local.st_mode) and saved['source'] == signature
                        and saved['local'] == [local.st_size, local.st_mtime_ns]):
                    logger.info('stage=lora_download cache=hit name=%s path=%s', source.stem, destination)
                    return destination
            except (OSError, ValueError, KeyError, TypeError):
                pass
            folder.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            logger.info('stage=lora_download begin name=%s bytes=%d path=%s', source.stem, before.st_size, destination)
            fd, temporary_name = tempfile.mkstemp(dir=folder, suffix='.part')
            os.close(fd)
            temporary = Path(temporary_name)
            marker_tmp = folder / 'complete.json.part'
            try:
                # An actual copy: never hardlink/symlink a network-backed mapping.
                shutil.copyfile(source, temporary)
                after = source.lstat()
                if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != tuple(signature[1:]):
                    raise OSError(f'LoRA source changed during copy: {source.name}')
                local = temporary.stat()
                if local.st_size != before.st_size:
                    raise OSError(f'Incomplete LoRA copy: {source.name}')
                marker_tmp.write_text(json.dumps({'source': signature, 'local': [local.st_size, local.st_mtime_ns]}))
                temporary.replace(destination)
                marker_tmp.replace(marker)
                logger.info('stage=lora_download done name=%s elapsed_s=%.2f path=%s', source.stem, time.monotonic() - started, destination)
                return destination
            except BaseException:
                logger.exception('stage=lora_download failed name=%s', source.stem)
                raise
            finally:
                temporary.unlink(missing_ok=True)
                marker_tmp.unlink(missing_ok=True)
