# SPDX-License-Identifier: Apache-2.0
"""Build varied N=1 SOMA requests from the bundled sample clips.

The requests use the same encode kernels as MotionBricksSOMAEngine, run by Warp on
the CPU, so no TensorRT or CUDA device is needed. The output is an .npz file with
one stacked array per model input, for parity checks and browser benchmarks.

    python tools/onnx/webgpu/soma_requests.py --count 240 --out requests.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import warp as wp

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from motionbricks.io.bvh import load_bvh  # noqa: E402
from motionbricks.model.contract import MODEL_POSES_PER_TOKEN, MODEL_SAMPLE_RATE  # noqa: E402
from motionbricks.model.kernels import write_motionbricks_conditions_kernel  # noqa: E402
from motionbricks.model.soma import kernels  # noqa: E402
from motionbricks.model.soma.contract import SOMA_N1_CONTRACT, SOMA_N1_HEADING_JOINTS  # noqa: E402
from motionbricks.utils.pose_utils import compute_world_poses  # noqa: E402

INPUT_NAMES = (
    "global_root_values",
    "has_global_root_values",
    "local_root_values",
    "has_local_root_values",
    "local_poses",
    "has_local_poses",
    "allowed_pred_num_tokens",
)
MIN_TOKENS, MAX_TOKENS = 6, 16


def sample_poses(clip, times):
    poses = wp.empty((len(times), clip.skeleton.num_joints), dtype=wp.transform, device="cpu")
    for row, time in enumerate(times):
        clip.sample(float(time), poses[row])
    return poses.numpy()


def encode_request(skeleton, context, target, num_targets, token_counts, heading):
    """Mirror MotionBricksSOMAEngine._encode_guide_poses() and write_conditions()."""
    model_to_pose = wp.array(
        [skeleton.joint_index(name) for name in SOMA_N1_CONTRACT.joint_names], dtype=wp.int32, device="cpu"
    )
    right_leg, left_leg = (skeleton.joint_index(name) for name in SOMA_N1_HEADING_JOINTS)
    padded_target = np.concatenate(
        [target[:num_targets], np.repeat(target[num_targets - 1 : num_targets], MODEL_POSES_PER_TOKEN - num_targets, 0)]
    )
    local = wp.array(np.concatenate([context, padded_target]), dtype=wp.transform, device="cpu")
    world = wp.empty_like(local)
    compute_world_poses(local, skeleton.parent_indices, world)
    alignment = wp.empty(1, dtype=wp.transform, device="cpu")
    wp.launch(
        kernels.compute_guide_alignment_kernel,
        dim=1,
        inputs=[right_leg, left_leg, wp.vec2(*heading), world],
        outputs=[alignment],
        device="cpu",
    )
    tensors = {
        spec.name: wp.zeros(spec.shape, dtype=wp.float32 if np.dtype(spec.dtype) == np.float32 else wp.int32, device="cpu")
        for spec in SOMA_N1_CONTRACT.tensor_specs
        if spec.name in INPUT_NAMES
    }
    wp.launch(
        kernels.encode_motionbricks_inputs_kernel,
        dim=2 * MODEL_POSES_PER_TOKEN,
        inputs=[len(SOMA_N1_CONTRACT.joint_names), right_leg, left_leg, model_to_pose, world, alignment],
        outputs=[tensors["global_root_values"], tensors["local_root_values"], tensors["local_poses"]],
        device="cpu",
    )
    bits = 0
    for count in token_counts:
        bits |= 1 << (count - MIN_TOKENS)
    wp.launch(
        write_motionbricks_conditions_kernel,
        dim=MAX_TOKENS - MIN_TOKENS + 1,
        inputs=[num_targets, bits],
        outputs=[tensors[name] for name in ("has_global_root_values", "has_local_root_values", "has_local_poses", "allowed_pred_num_tokens")],
        device="cpu",
    )
    return {name: tensors[name].numpy() for name in INPUT_NAMES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    wp.config.kernel_cache_dir = os.environ.get("WARP_CACHE_PATH", wp.config.kernel_cache_dir)
    wp.config.quiet = True
    wp.init()
    rng = np.random.default_rng(args.seed)
    paths = sorted((REPO / "assets/motions/soma").rglob("*.bvh"))
    skeleton = load_bvh(paths[0]).skeleton
    clips = [load_bvh(path, skeleton) for path in paths]
    dt = 1.0 / MODEL_SAMPLE_RATE
    requests, meta = [], []
    for index in range(args.count):
        source = int(rng.integers(len(clips)))
        clip = clips[source]
        start = float(rng.uniform(3 * dt, max(3 * dt, clip.duration - 0.8)))
        context = sample_poses(clip, start + dt * np.arange(-3, 1))
        cross_clip = rng.random() < 0.3
        target_source = int(rng.integers(len(clips))) if cross_clip else source
        target_clip = clips[target_source]
        gap = float(rng.uniform(0.6, 1.9))
        target_start = float(rng.uniform(0.0, target_clip.duration)) if cross_clip else min(start + gap, target_clip.duration)
        target = sample_poses(target_clip, target_start + dt * np.arange(4))
        num_targets = int(rng.integers(1, 5))
        if rng.random() < 0.7:
            token_counts = tuple(range(8, 15))
        else:
            low = int(rng.integers(MIN_TOKENS, MAX_TOKENS + 1))
            token_counts = tuple(range(low, int(rng.integers(low, MAX_TOKENS + 1)) + 1))
        angle = 0.0 if rng.random() < 0.6 else float(rng.uniform(-np.pi, np.pi))
        heading = (float(np.cos(angle)), float(np.sin(angle)))
        requests.append(encode_request(skeleton, context, target, num_targets, token_counts, heading))
        meta.append((source, target_source, start, target_start, num_targets, min(token_counts), max(token_counts), angle))
    stacked = {name: np.stack([request[name] for request in requests]) for name in INPUT_NAMES}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **stacked, meta=np.asarray(meta, dtype=np.float64),
             clips=np.asarray([str(path.relative_to(REPO)) for path in paths]))
    print(f"wrote {args.count} requests from {len(paths)} clips to {args.out}")


if __name__ == "__main__":
    main()

