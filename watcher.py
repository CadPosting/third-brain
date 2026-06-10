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
from indexer import index_file, remove_file

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

    def on_deleted(self, event):
        # Without this, chunks of deleted notes stay searchable forever.
        if event.is_directory or event.src_path.endswith(".md"):
            self._cancel(event.src_path)
            self._remove(event.src_path)

    def on_moved(self, event):
        # Covers Obsidian's "delete" (a move into .trash/) and renames: drop
        # chunks at the old path, re-index the destination if it's still an
        # indexable note (index_file itself skips .trash/.obsidian).
        if event.is_directory or event.src_path.endswith(".md"):
            self._cancel(event.src_path)
            self._remove(event.src_path)
        dest = getattr(event, "dest_path", "") or ""
        if event.is_directory and dest:
            for p in Path(dest).rglob("*.md"):
                self._debounce(str(p))
        elif dest.endswith(".md"):
            self._debounce(dest)

    def _debounce(self, path: str):
        if path in self._timers:
            self._timers[path].cancel()
        t = threading.Timer(DEBOUNCE_SECONDS, self._index, args=[path])
        self._timers[path] = t
        t.start()

    def _cancel(self, path: str):
        timer = self._timers.pop(path, None)
        if timer:
            timer.cancel()

    def _remove(self, path: str):
        try:
            remove_file(path)
            log.info(f"removed {Path(path).name} from index")
        except Exception as e:
            log.error(f"failed to remove {path}: {e}")

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
