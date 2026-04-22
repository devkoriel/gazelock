"""Stdlib HTTP dashboard tailing manifest.jsonl during a Blender render.

Usage:
    python -m gazelock_ml.synthesis.blender.progress_viz \
        --root weights/synthetic-eyes \
        --total 500000 \
        --port 8765

Serves a Precision Dark dashboard with live progress, ETA, rolling
average render time, and the last 20 rendered frames. Stdlib only —
safe to run 24/7 alongside a multi-day render.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

TEMPLATE_PATH = Path(__file__).parent / "viz_template.html"


@dataclass
class Stats:
    count: int
    total: int
    pct: float
    avg_seconds: float
    eta_seconds: float
    eta_human: str
    eta_done_at: str
    elapsed_seconds: float
    elapsed_human: str
    started_at: str
    latest_frames: list[dict[str, Any]]


class ManifestTail:
    """Incrementally reads manifest.jsonl; caches stats."""

    def __init__(self, root: Path, total: int) -> None:
        self.root = root
        self.total = total
        self._entries: list[dict[str, Any]] = []
        self._position = 0
        self._lock = threading.Lock()
        self._started_epoch: float | None = None

    def _manifest_path(self) -> Path:
        return self.root / "manifest.jsonl"

    def refresh(self) -> None:
        mp = self._manifest_path()
        if not mp.exists():
            return
        with self._lock:
            try:
                with mp.open() as fh:
                    fh.seek(self._position)
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        self._entries.append(entry)
                        if self._started_epoch is None and "rendered_at" in entry:
                            render_s = float(entry.get("render_seconds", 0))
                            self._started_epoch = entry["rendered_at"] - render_s
                    self._position = fh.tell()
            except OSError:
                pass

    def stats(self) -> Stats:
        with self._lock:
            count = len(self._entries)
            pct = count / max(self.total, 1)

            recent = self._entries[-200:]
            if recent:
                avg = sum(e.get("render_seconds", 0) for e in recent) / len(recent)
            else:
                avg = 0.0

            remaining_frames = max(0, self.total - count)
            eta_s = remaining_frames * avg

            if self._started_epoch and count:
                elapsed = time.time() - self._started_epoch
                started_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(self._started_epoch))
            else:
                elapsed = 0.0
                started_str = ""

            eta_done_at = ""
            if eta_s > 0:
                eta_done_at = time.strftime(
                    "%Y-%m-%d %H:%M",
                    time.localtime(time.time() + eta_s),
                )

            latest = [
                {
                    "frame_id": e["frame_id"],
                    "png_path": e["png_path"],
                }
                for e in self._entries[-20:]
            ]

        return Stats(
            count=count,
            total=self.total,
            pct=pct,
            avg_seconds=avg,
            eta_seconds=eta_s,
            eta_human=_fmt_duration(eta_s),
            eta_done_at=eta_done_at,
            elapsed_seconds=elapsed,
            elapsed_human=_fmt_duration(elapsed),
            started_at=started_str,
            latest_frames=latest,
        )


def _fmt_duration(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    s = int(seconds)
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    if d:
        return f"{d}d {h}h {m}m"
    if h:
        return f"{h}h {m}m"
    return f"{m}m"


def _make_handler(tail: ManifestTail):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                return self._send_file(TEMPLATE_PATH, "text/html")
            if self.path.startswith("/stats.json"):
                tail.refresh()
                payload = json.dumps(asdict(tail.stats())).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.path.startswith("/frame/"):
                fname = self.path[len("/frame/") :].split("?", 1)[0]
                path = tail.root / Path(fname).name
                if not path.exists():
                    self.send_error(404)
                    return
                mime, _ = mimetypes.guess_type(str(path))
                return self._send_file(path, mime or "application/octet-stream")
            self.send_error(404)

        def _send_file(self, path: Path, mime: str) -> None:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    tail = ManifestTail(root=args.root, total=args.total)

    server = ThreadingHTTPServer((args.host, args.port), _make_handler(tail))
    print(f"[viz] http://{args.host}:{args.port} (watching {args.root})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[viz] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
