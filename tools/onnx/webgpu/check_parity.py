# SPDX-License-Identifier: Apache-2.0
"""Compare a static WebGPU graph with the original SOMA ONNX on the CPU.

Each model runs alone, one request at a time, so only one session is in memory.
For every request it checks that pred_num_tokens matches and reports the largest
absolute difference of pred_global_motions over the valid frames.

    python tools/onnx/webgpu/check_parity.py ORIGINAL.onnx STATIC.onnx requests.npz --json out.json
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

INPUT_NAMES = (
    "global_root_values",
    "has_global_root_values",
    "local_root_values",
    "has_local_root_values",
    "local_poses",
    "has_local_poses",
    "allowed_pred_num_tokens",
)
FRAMES_PER_TOKEN = 4


def run_all(path: Path, requests: dict[str, np.ndarray], threads: int, optimize: bool) -> tuple[np.ndarray, np.ndarray, list[float]]:
    options = ort.SessionOptions()
    if not optimize:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    count = requests[INPUT_NAMES[0]].shape[0]
    motions, tokens, times = [], [], []
    for index in range(count):
        feed = {name: requests[name][index] for name in INPUT_NAMES}
        start = time.perf_counter()
        motion, token = session.run(["pred_global_motions", "pred_num_tokens"], feed)
        times.append((time.perf_counter() - start) * 1000.0)
        motions.append(motion[0])
        tokens.append(int(token[0]))
    del session
    gc.collect()
    return np.stack(motions), np.asarray(tokens), times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("original", type=Path)
    parser.add_argument("static", type=Path)
    parser.add_argument("requests", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable ONNX Runtime graph optimizations, so both graphs run the same kernels without fusions.",
    )
    args = parser.parse_args()

    requests = dict(np.load(args.requests))
    original_motion, original_tokens, original_ms = run_all(args.original, requests, args.threads, not args.no_optimize)
    static_motion, static_tokens, static_ms = run_all(args.static, requests, args.threads, not args.no_optimize)
    token_match = original_tokens == static_tokens
    diffs = []
    for index, tokens in enumerate(original_tokens):
        frames = FRAMES_PER_TOKEN * int(tokens)
        diffs.append(float(np.abs(original_motion[index, :frames] - static_motion[index, :frames]).max()))
    diffs = np.asarray(diffs)
    allowed = requests["allowed_pred_num_tokens"].reshape(len(static_tokens), -1)
    legal = np.asarray([0 <= tokens - 6 < allowed.shape[1] and allowed[i, tokens - 6] == 1 for i, tokens in enumerate(static_tokens)])
    finite = np.asarray([np.isfinite(static_motion[i, : FRAMES_PER_TOKEN * int(t)]).all() for i, t in enumerate(static_tokens)])
    summary = {
        "requests": int(len(diffs)),
        "token_count_matches": int(token_match.sum()),
        "token_counts": {str(k): int(v) for k, v in zip(*np.unique(original_tokens, return_counts=True))},
        "valid_frame_max_abs_diff": float(diffs[token_match].max()) if token_match.any() else None,
        "valid_frame_max_abs_diff_p99": float(np.percentile(diffs[token_match], 99)) if token_match.any() else None,
        "bit_identical_requests": int((diffs == 0).sum()),
        "mismatched_requests": [int(i) for i in np.flatnonzero(~token_match)],
        "static_finite_requests": int(finite.sum()),
        "static_legal_token_requests": int(legal.sum()),
        "cpu_ms_original_p50_p95": [float(np.percentile(original_ms, q)) for q in (50, 95)],
        "cpu_ms_static_p50_p95": [float(np.percentile(static_ms, q)) for q in (50, 95)],
        "onnxruntime": ort.__version__,
        "threads": args.threads,
        "graph_optimizations": not args.no_optimize,
    }
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

