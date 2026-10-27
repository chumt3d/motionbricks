# Static N=1 SOMA graph for WebGPU

These scripts prepare the SOMA ONNX models for onnxruntime-web with the WebGPU
execution provider and check the result against the original models.

The exported models accept any batch size, so their shape arithmetic runs at run
time. In onnxruntime-web 1.30.0-dev (a6e96aa798) on an RTX 4070, the original FP32
graph places 185 nodes on the CPU and inserts 577 GPU<->host copies. The WebGPU EP
also lacks OneHot and some integer and bool kernels. The static graph runs every node
on the GPU with no copies, which is also what WebGPU graph capture requires.

| Script | Purpose |
| --- | --- |
| `make_static.py` | Fixes the batch at 1, folds constant subgraphs with the ONNX Runtime CPU kernels, replaces OneHot with Cast, Equal and Where, rewrites the integer and bool nodes the WebGPU EP cannot run, makes the scatters safe for graph capture, and writes the graph with external weights. |
| `soma_requests.py` | Builds varied N=1 requests from the bundled SOMA sample clips with the engine's own Warp encode kernels on the CPU. |
| `check_parity.py` | Runs the original and static graphs on the CPU and compares token counts and the valid frames of `pred_global_motions`, and checks that the static outputs are finite and their token counts allowed. |
| `bench/` | Browser benchmark for onnxruntime-web: a cross-origin-isolated static server, an inference worker, a frame-pacing worker and `bench.html`. It measures placement, copies, latency with and without graph capture, device parity, and co-residency with other ONNX sessions. |

```sh
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python numpy==2.4.3 warp-lang onnx onnxruntime
git lfs pull --include='motionbricks/assets/onnx/soma/*,assets/motions/soma/**'
.venv/bin/python tools/onnx/webgpu/make_static.py motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx soma_fp32_static.onnx
.venv/bin/python tools/onnx/webgpu/soma_requests.py --count 240 --out requests.npz
.venv/bin/python tools/onnx/webgpu/check_parity.py motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx soma_fp32_static.onnx requests.npz
```

## Results on 240 requests (ONNX Runtime 1.31.0, CPU)

| Graph | Nodes | Token counts equal | Max abs difference, valid frames |
| --- | --- | --- | --- |
| FP32 | 6,993 -> 1,978 | 240 of 240 | 1.4e-3 (median 5.9e-5) |
| Mixed FP16 | 7,169 -> 2,149 | 239 of 240 | see below |

The FP32 static graph matches the original. The remaining differences come from
kernel choices and accumulation order, not from token choices.

The FP16 comparison on the CPU is not a like-for-like reference. ONNX Runtime's CPU
EP has few FP16 kernels, so it casts FP16 chains to FP32 and keeps intermediate
results in FP32. The folded constants keep the FP16 types the graph declares; the
attention scale, for example, is 0.4204 in FP16 and 0.42045 in the CPU's FP32
chain. Mixed FP16 already picks different motion tokens than FP32 on many requests,
as the [SOMA model card](../../../docs/soma_model.md#inference) notes, so those
small constant changes move outputs by about as much as FP16 itself does:

| Compared with the original FP32 model | Token counts equal | Within 1e-2 | Median | 95th percentile |
| --- | --- | --- | --- | --- |
| Original mixed FP16 | 240 | 31 | 0.064 | 1.51 |
| Static mixed FP16 | 239 | 27 | 0.088 | 1.24 |

Use FP32 when outputs must match the reference runtime. Compare FP16 on the target
runtime, where FP16 kernels run in FP16.

## Graph capture

ONNX Runtime's WebGPU graph capture records compute dispatches and replays them. Its
GPU-to-GPU tensor copy (`BufferManager::MemCpy`) encodes a buffer copy that the
capture does not record. ScatterND and ScatterElements use that copy when their
output cannot reuse their data input, which is the case when the data input is a
constant. Replays then scatter into the previous run's output and the results drift.
`make_static.py` adds a zero computed from an input to each scatter's data
input, so ONNX Runtime runs every scatter in place and nothing needs the copy.

## Licence

The scripts are Apache-2.0 like the rest of the repository. The static graphs are
modified models under the NVIDIA Open Model Agreement in
[motionbricks/assets/onnx/LICENSE.md](../../../motionbricks/assets/onnx/LICENSE.md),
and are not committed. The sample clips the request builder reads are for
evaluation only under [assets/motions/LICENSE.txt](../../../assets/motions/LICENSE.txt).

