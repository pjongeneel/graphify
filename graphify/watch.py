# monitor a folder and auto-trigger --update when files change
from __future__ import annotations
import json
import time
from pathlib import Path


_WATCHED_EXTENSIONS = {
    ".py", ".ts", ".js", ".go", ".rs", ".java", ".cpp", ".c", ".rb", ".swift", ".kt",
    ".cs", ".scala", ".php", ".cc", ".cxx", ".hpp", ".h", ".kts",
    ".md", ".txt", ".rst", ".pdf",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg",
}

_CODE_EXTENSIONS = {
    ".py", ".ts", ".js", ".go", ".rs", ".java", ".cpp", ".c", ".rb", ".swift", ".kt",
    ".cs", ".scala", ".php", ".cc", ".cxx", ".hpp", ".h", ".kts",
}


def _rebuild_code(watch_path: Path, *, follow_symlinks: bool = False) -> bool:
    """Re-run AST extraction + build + cluster + report for code files. No LLM needed.

    Returns True on success, False on error.
    """
    try:
        from graphify.scan import scan
        result = scan(watch_path, follow_symlinks=follow_symlinks)
        print(f"[graphify watch] Rebuilt: {result['node_count']} nodes, "
              f"{result['edge_count']} edges, {result['community_count']} communities")
        return True
    except RuntimeError as exc:
        print(f"[graphify watch] {exc}")
        return False
    except Exception as exc:
        print(f"[graphify watch] Rebuild failed: {exc}")
        return False


def _notify_only(watch_path: Path) -> None:
    """Write a flag file and print a notification (fallback for non-code-only corpora)."""
    flag = watch_path / "graphify-out" / "needs_update"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("1")
    print(f"\n[graphify watch] New or changed files detected in {watch_path}")
    print("[graphify watch] Non-code files changed - semantic re-extraction requires LLM.")
    print("[graphify watch] Run `/graphify --update` in Claude Code to update the graph.")
    print(f"[graphify watch] Flag written to {flag}")


def _has_non_code(changed_paths: list[Path]) -> bool:
    return any(p.suffix.lower() not in _CODE_EXTENSIONS for p in changed_paths)


def watch(watch_path: Path, debounce: float = 3.0) -> None:
    """
    Watch watch_path for new or modified files and auto-update the graph.

    For code-only changes: re-runs AST extraction + rebuild immediately (no LLM).
    For doc/paper/image changes: writes a needs_update flag and notifies the user
    to run /graphify --update (LLM extraction required).

    debounce: seconds to wait after the last change before triggering (avoids
    running on every keystroke when many files are saved at once).
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError as e:
        raise ImportError("watchdog not installed. Run: pip install watchdog") from e

    last_trigger: float = 0.0
    pending: bool = False
    changed: set[Path] = set()

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            nonlocal last_trigger, pending
            if event.is_directory:
                return
            path = Path(event.src_path)
            if path.suffix.lower() not in _WATCHED_EXTENSIONS:
                return
            if any(part.startswith(".") for part in path.parts):
                return
            if "graphify-out" in path.parts:
                return
            last_trigger = time.monotonic()
            pending = True
            changed.add(path)

    handler = Handler()
    observer = Observer()
    observer.schedule(handler, str(watch_path), recursive=True)
    observer.start()

    print(f"[graphify watch] Watching {watch_path.resolve()} - press Ctrl+C to stop")
    print(f"[graphify watch] Code changes rebuild graph automatically. "
          f"Doc/image changes require /graphify --update.")
    print(f"[graphify watch] Debounce: {debounce}s")

    try:
        while True:
            time.sleep(0.5)
            if pending and (time.monotonic() - last_trigger) >= debounce:
                pending = False
                batch = list(changed)
                changed.clear()
                print(f"\n[graphify watch] {len(batch)} file(s) changed")
                if _has_non_code(batch):
                    _notify_only(watch_path)
                else:
                    _rebuild_code(watch_path)
    except KeyboardInterrupt:
        print("\n[graphify watch] Stopped.")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Watch a folder and auto-update the graphify graph")
    parser.add_argument("path", nargs="?", default=".", help="Folder to watch (default: .)")
    parser.add_argument("--debounce", type=float, default=3.0,
                        help="Seconds to wait after last change before updating (default: 3)")
    args = parser.parse_args()
    watch(Path(args.path), debounce=args.debounce)
