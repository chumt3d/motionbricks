# SPDX-License-Identifier: Apache-2.0
"""Write requests (and optional CPU reference outputs) as raw arrays for bench.html.

    python prepare_data.py requests.npz OUT_DIR [--reference NAME=outputs.npz ...]

outputs.npz holds "motion" [R, 64, 366] float32 and "tokens" [R] int, as written by
a CPU run of the original model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

INPUT_NAMES = (
    "global_root_values",
    "has_global_root_values",
    "local_root_values",
    "has_local_root_values",
    "local_poses",
    "has_local_poses",
    "allowed_pred_num_tokens",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("requests", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--reference", action="append", default=[], help="NAME=outputs.npz")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    requests = np.load(args.requests)
    manifest = {"count": int(requests[INPUT_NAMES[0]].shape[0]), "inputs": {}, "references": {}}
    for name in INPUT_NAMES:
        array = np.ascontiguousarray(requests[name])
        array.tofile(args.out / f"{name}.bin")
        manifest["inputs"][name] = {"dtype": str(array.dtype), "dims": list(array.shape[1:])}
    for item in args.reference:
        label, path = item.split("=", 1)
        reference = np.load(path)
        np.ascontiguousarray(reference["motion"], dtype=np.float32).tofile(args.out / f"ref_{label}_motion.bin")
        np.ascontiguousarray(reference["tokens"], dtype=np.int32).tofile(args.out / f"ref_{label}_tokens.bin")
        manifest["references"][label] = {"motion_dims": list(reference["motion"].shape[1:])}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

