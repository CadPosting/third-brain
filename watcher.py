"""
File watcher — monitors the vault for changes and auto-indexes.
Run alongside server.py: python watcher.py
Anything you write in Obsidian gets indexed within seconds.
"""
import time
import logging
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from indexer import index_file

VAULT_PATH = Path.home() / "vault"
DEBOUNCE_SECONDS = 2.0  # wait for file to stop changing before indexing

logging.basicConfig(level=logging.INFO, format="[watcher] %(message)s")
log = logging.getLogger(__name__)


class VaultHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self._timers: dict[str, threading.Timer] = {}

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self._debounce(event.src_path)

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self._debounce(event.src_path)

    def _debounce(self, path: str):
        if path in self._timers:
            self._timers[path].cancel()
        t = threading.Timer(DEBOUNCE_SECONDS, self._index, args=[path])
        self._timers[path] = t
        t.start()

    def _index(self, path: str):
        self._timers.pop(path, None)
        try:
            chunks = index_file(path)
            log.info(f"indexed {Path(path).name} ({chunks} chunks)")
        except Exception as e:
            log.error(f"failed to index {path}: {e}")


def run():
    VAULT_PATH.mkdir(parents=True, exist_ok=True)
    observer = Observer()
    observer.schedule(VaultHandler(), str(VAULT_PATH), recursive=True)
    observer.start()
    log.info(f"watching {VAULT_PATH}")
    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    run()
