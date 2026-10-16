"""A CPU-only download process and per-component barriers for model loaders."""

from __future__ import annotations

import json
from contextlib import contextmanager
import subprocess
import tempfile
import time
from pathlib import Path


_active_session = None


class DownloadError(RuntimeError):
    pass


class ComponentPublisher:
    def __init__(self, state_dir: str | Path | None):
        self.state_dir = Path(state_dir) if state_dir else None

    def ready(self, component: str, path: str | Path) -> None:
        if self.state_dir is None:
            return
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Downloaded component {component} is missing: {path}")
        target = self.state_dir / f"{component}.json"
        pending = target.with_suffix(".tmp")
        pending.write_text(json.dumps({"path": str(path)}))
        pending.replace(target)
        print(f"stage=component_ready component={component} path={path}", flush=True)


class PrefetchSession:
    def __init__(self, command: list[str], *, env: dict | None = None):
        self.command = command
        self.env = env

    def __enter__(self):
        global _active_session
        self._temp = tempfile.TemporaryDirectory(prefix="model-prefetch-")
        self.state_dir = Path(self._temp.name)
        try:
            self.process = subprocess.Popen(
                [*self.command, "--prefetch-state-dir", str(self.state_dir)], env=self.env,
            )
        except BaseException:
            self._temp.cleanup()
            raise
        self._previous = _active_session
        _active_session = self
        return self

    def wait(self, component: str, *, timeout: float = 1800) -> Path:
        started = time.monotonic()
        deadline = started + timeout
        ready = self.state_dir / f"{component}.json"
        print(f"stage=wait_component begin component={component}", flush=True)
        while True:
            if ready.is_file():
                # The final barrier also verifies a clean process exit. An
                # early ready marker must never hide a later downloader crash.
                if component == "all":
                    try:
                        status = self.process.wait(timeout=max(.001, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired as err:
                        raise TimeoutError("Downloader did not finish") from err
                    if status != 0:
                        raise DownloadError(f"Downloader exited with code {status}")
                path = Path(json.loads(ready.read_text())["path"])
                print(f"stage=wait_component done component={component} elapsed_s={time.monotonic() - started:.2f}", flush=True)
                return path
            status = self.process.poll()
            if status is not None:
                # Recheck after poll: the marker may have appeared between the
                # first check and the process exiting.
                if ready.is_file():
                    continue
                raise DownloadError(f"Downloader exited with code {status} before component {component} was ready")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for component {component}")
            time.sleep(.025)

    def __exit__(self, exc_type, exc, traceback):
        global _active_session
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        finally:
            _active_session = self._previous
            self._temp.cleanup()


def wait_for_component(component: str, *, timeout: float = 1800) -> Path | None:
    """Direct/manual pipeline loading keeps its existing behavior without a session."""
    if _active_session is None:
        return None
    return _active_session.wait(component, timeout=timeout)


@contextmanager
def loading_component(component: str):
    started = time.monotonic()
    print(f"stage=component_load begin component={component}", flush=True)
    try:
        yield
    except BaseException:
        print(f"stage=component_load failed component={component} elapsed_s={time.monotonic() - started:.2f}", flush=True)
        raise
    else:
        print(f"stage=component_load done component={component} elapsed_s={time.monotonic() - started:.2f}", flush=True)
