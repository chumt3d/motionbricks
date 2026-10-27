# SPDX-License-Identifier: Apache-2.0
"""Serve the WebGPU benchmark with cross-origin isolation and collect its results.

    python server.py --port 8931 --results OUT_DIR --mount ort=PATH --mount models=PATH ...

Each --mount NAME=PATH serves PATH under /NAME/. POST /results/NAME.json writes the
request body to OUT_DIR/NAME.json. Everything is bound to 127.0.0.1.
"""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import posixpath
import re
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent


class Handler(SimpleHTTPRequestHandler):
    mounts: dict[str, Path] = {}
    results: Path = HERE

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def translate_path(self, path: str) -> str:
        parts = [part for part in posixpath.normpath(unquote(urlsplit(path).path)).split("/") if part and part != ".."]
        if parts and parts[0] in self.mounts:
            return str(self.mounts[parts[0]].joinpath(*parts[1:]))
        return str(HERE.joinpath(*parts))

    def do_POST(self) -> None:
        match = re.fullmatch(r"/results/([A-Za-z0-9._-]+\.(?:json|log))", urlsplit(self.path).path)
        if not match:
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        (self.results / match.group(1)).write_bytes(body)
        self.send_response(204)
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        if "POST" in (args[0] if args else ""):
            super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8931)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--mount", action="append", default=[], help="NAME=PATH")
    args = parser.parse_args()
    Handler.mounts = {name: Path(path).resolve() for name, path in (item.split("=", 1) for item in args.mount)}
    Handler.results = args.results.resolve()
    Handler.results.mkdir(parents=True, exist_ok=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler)).serve_forever()


if __name__ == "__main__":
    main()

