// SPDX-License-Identifier: Apache-2.0
// Inference worker for bench.html: MotionBricks SOMA and, optionally, ARDY Core40
// sessions on one WebGPU device, with onnxruntime-web's WebGPU EP.

const logLines = [];
const KEEP = /Memcpy|placed on|Node placements|not assigned|fallback|DEVICE LOST|rror|not supported|unsupported|VerifyEachNodeIsAssignedToAnEp/;
for (const level of ["log", "info", "warn", "error", "debug"]) {
  const original = console[level].bind(console);
  console[level] = (...args) => {
    const line = args.map(String).join(" ");
    if (logLines.length < 50000 && KEEP.test(line)) logLines.push(line.replace(/^\S+ \S+ \[([A-Z]):onnxruntime:[^\]]*\] /, "$1 "));
    original(...args);
  };
}

const workerParams = new URLSearchParams(self.location.search);
const ortRoot = workerParams.get("ort") || "/ort/dist";
const ort = await import(ortRoot + "/" + (workerParams.get("ortEntry") || "ort.webgpu.mjs"));
const wasmRoot = workerParams.get("wasm") || ortRoot;
ort.env.wasm.wasmPaths = {
  mjs: wasmRoot + "/ort-wasm-simd-threaded.asyncify.mjs",
  wasm: wasmRoot + "/ort-wasm-simd-threaded.asyncify.wasm",
};
ort.env.wasm.proxy = false;
ort.env.wasm.numThreads = Math.min(4, Math.max(1, navigator.hardwareConcurrency || 1));
ort.env.logLevel = "warning";

const INPUTS = [
  "global_root_values",
  "has_global_root_values",
  "local_root_values",
  "has_local_root_values",
  "local_poses",
  "has_local_poses",
  "allowed_pred_num_tokens",
];

let adapter;
let device;
let requests;
let reference;
let mb;
let core;

const now = () => performance.now();
const pct = (values, q) => {
  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length === 0) return null;
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil((q / 100) * sorted.length) - 1))];
};
const stats = (values) => ({
  n: values.length,
  mean: values.reduce((a, b) => a + b, 0) / Math.max(1, values.length),
  p50: pct(values, 50),
  p95: pct(values, 95),
  p99: pct(values, 99),
  max: Math.max(...values),
});

async function fetchBytes(url) {
  const start = now();
  const response = await fetch(url);
  if (!response.ok) throw new Error(url + " -> " + response.status);
  const bytes = new Uint8Array(await response.arrayBuffer());
  return { bytes, ms: now() - start };
}

async function initDevice({ powerPreference, wantF16 }) {
  adapter = await navigator.gpu.requestAdapter({ powerPreference });
  if (!adapter) throw new Error("no adapter for " + powerPreference);
  const features = [];
  if (wantF16 && adapter.features.has("shader-f16")) features.push("shader-f16");
  if (adapter.features.has("timestamp-query")) features.push("timestamp-query");
  const limitNames = [
    "maxBufferSize",
    "maxStorageBufferBindingSize",
    "maxStorageBuffersPerShaderStage",
    "maxComputeWorkgroupStorageSize",
    "maxComputeInvocationsPerWorkgroup",
    "maxComputeWorkgroupSizeX",
    "maxComputeWorkgroupSizeY",
    "maxComputeWorkgroupSizeZ",
    "maxComputeWorkgroupsPerDimension",
  ];
  const requiredLimits = Object.fromEntries(limitNames.map((name) => [name, adapter.limits[name]]));
  device = await adapter.requestDevice({ requiredFeatures: features, requiredLimits });
  device.lost.then((info) => console.error("DEVICE LOST " + info.reason + " " + info.message));
  ort.env.webgpu.adapter = adapter;
  const info = adapter.info || {};
  return {
    powerPreference,
    vendor: info.vendor,
    architecture: info.architecture,
    device: info.device,
    description: info.description,
    isFallbackAdapter: info.isFallbackAdapter,
    shaderF16: adapter.features.has("shader-f16"),
    deviceFeatures: [...device.features],
    limits: Object.fromEntries(limitNames.map((name) => [name, adapter.limits[name]])),
  };
}

function epOptions(extra = {}) {
  return {
    name: "webgpu",
    device,
    enableInt64: true,
    storageBufferCacheMode: "lazyRelease",
    uniformBufferCacheMode: "lazyRelease",
    queryResolveBufferCacheMode: "lazyRelease",
    defaultBufferCacheMode: "lazyRelease",
    ...extra,
  };
}

async function loadRequests() {
  if (requests) return;
  const manifest = await (await fetch("/data/manifest.json")).json();
  requests = { count: manifest.count, inputs: {} };
  for (const [name, meta] of Object.entries(manifest.inputs)) {
    const buffer = await (await fetch("/data/" + name + ".bin")).arrayBuffer();
    requests.inputs[name] = { meta, data: meta.dtype === "float32" ? new Float32Array(buffer) : new Int32Array(buffer) };
  }
  if (manifest.references.o32) {
    reference = {
      motion: new Float32Array(await (await fetch("/data/ref_o32_motion.bin")).arrayBuffer()),
      tokens: new Int32Array(await (await fetch("/data/ref_o32_tokens.bin")).arrayBuffer()),
    };
  }
}

function requestSlice(name, index) {
  const { meta, data } = requests.inputs[name];
  const size = meta.dims.reduce((a, b) => a * b, 1);
  return { dims: meta.dims, data: data.subarray(index * size, (index + 1) * size), type: meta.dtype === "float32" ? "float32" : "int32" };
}

function memcpySummary(lines) {
  const memcpy = lines.map((line) => line.match(/Add (MemcpyFromHost|MemcpyToHost) (after|before) (\S+) for (\S+)/)).filter(Boolean);
  const placements = {};
  let currentEp;
  const cpuNodes = [];
  const memcpyOps = {};
  for (const line of lines) {
    const header = line.match(/Node\(s\) placed on \[(\w+)\]\. Number of nodes: (\d+)/);
    if (header) { currentEp = header[1]; placements[currentEp] = Number(header[2]); continue; }
    const node = line.match(/^V\s{2,}(\S+) \((.*)\)\s*$/);
    if (node && currentEp === "CPUExecutionProvider") cpuNodes.push(node[1] + " " + node[2]);
    if (node && /^Memcpy(ToHost|FromHost)$/.test(node[1])) memcpyOps[node[1]] = (memcpyOps[node[1]] || 0) + 1;
  }
  const cpuOps = {};
  for (const entry of cpuNodes) { const op = entry.split(" ")[0]; cpuOps[op] = (cpuOps[op] || 0) + 1; }
  return {
    memcpyNodes: memcpy.length,
    memcpyFromHost: memcpy.filter((match) => match[1] === "MemcpyFromHost").map((match) => match[3]),
    memcpyToHost: memcpy.filter((match) => match[1] === "MemcpyToHost").map((match) => match[3]),
    memcpyOps,
    placements,
    cpuOps,
    cpuNodes: cpuNodes.slice(0, 80),
    unassignedWarnings: lines.filter((line) => /not assigned to the preferred execution providers/.test(line)).length,
  };
}

async function loadMB({ url, io, capture, freeDims, disableCpuFallback, skipFirstRun }) {
  if (mb) { await mb.session.release(); mb = undefined; }
  await loadRequests();
  const download = await fetchBytes(url);
  const external = url.includes("static") ? await fetchBytes(url + ".data") : null;
  const logStart = logLines.length;
  const options = {
    executionProviders: [epOptions(capture ? {} : {})],
    graphOptimizationLevel: "all",
    logSeverityLevel: 0,
    logVerbosityLevel: 0,
    enableGraphCapture: !!capture,
  };
  if (freeDims) options.freeDimensionOverrides = { batch_size: 1 };
  if (external) options.externalData = [{ data: external.bytes, path: url.split("/").pop() + ".data" }];
  if (disableCpuFallback) options.extra = { session: { disable_cpu_ep_fallback: "1" } };
  if (io === "gpu" || capture) options.preferredOutputLocation = { pred_global_motions: "gpu-buffer", pred_num_tokens: capture ? "gpu-buffer" : "cpu" };
  const createStart = now();
  const session = await ort.InferenceSession.create(download.bytes, options);
  const createMs = now() - createStart;
  const sessionLog = logLines.slice(logStart);
  mb = { session, io, capture: !!capture, url };
  if (io === "gpu" || capture) {
    mb.gpuInputs = {};
    for (const name of INPUTS) {
      const slice = requestSlice(name, 0);
      const byteLength = slice.data.byteLength;
      const buffer = device.createBuffer({ size: Math.ceil(byteLength / 16) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      mb.gpuInputs[name] = { buffer, tensor: ort.Tensor.fromGpuBuffer(buffer, { dataType: slice.type, dims: slice.dims }) };
    }
    if (capture) {
      mb.gpuOutputs = {};
      for (const [name, dims, type, bytes] of [["pred_global_motions", [1, 64, 366], "float32", 64 * 366 * 4], ["pred_num_tokens", [1], "int32", 4]]) {
        const buffer = device.createBuffer({ size: Math.ceil(bytes / 16) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        mb.gpuOutputs[name] = { buffer, tensor: ort.Tensor.fromGpuBuffer(buffer, { dataType: type, dims }) };
      }
      mb.readback = device.createBuffer({ size: 64 * 366 * 4 + 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
  }
  const first = skipFirstRun ? { ms: null, tokens: null } : await runMB(0, true);
  return {
    url,
    io,
    capture: !!capture,
    downloadBytes: download.bytes.byteLength + (external ? external.bytes.byteLength : 0),
    downloadMs: download.ms + (external ? external.ms : 0),
    createMs,
    firstRunMs: first.ms,
    firstRunTokens: first.tokens,
    ...memcpySummary(sessionLog),
    sessionLogSample: sessionLog.filter((line) => !/^V\s{2,}\S+ \(|Add Memcpy/.test(line)).slice(0, 30),
  };
}

async function runMB(index, withMotion) {
  const start = now();
  let tokens;
  let motion;
  if (mb.io === "gpu" || mb.capture) {
    for (const name of INPUTS) {
      const slice = requestSlice(name, index);
      device.queue.writeBuffer(mb.gpuInputs[name].buffer, 0, slice.data);
    }
    const feeds = Object.fromEntries(INPUTS.map((name) => [name, mb.gpuInputs[name].tensor]));
    if (mb.capture) {
      const fetches = Object.fromEntries(Object.entries(mb.gpuOutputs).map(([name, value]) => [name, value.tensor]));
      await mb.session.run(feeds, fetches);
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(mb.gpuOutputs.pred_num_tokens.buffer, 0, mb.readback, 0, 16);
      if (withMotion) encoder.copyBufferToBuffer(mb.gpuOutputs.pred_global_motions.buffer, 0, mb.readback, 16, 64 * 366 * 4);
      device.queue.submit([encoder.finish()]);
      await mb.readback.mapAsync(GPUMapMode.READ);
      const mapped = mb.readback.getMappedRange();
      tokens = new Int32Array(mapped, 0, 1)[0];
      if (withMotion) motion = new Float32Array(mapped.slice(16, 16 + 64 * 366 * 4));
      mb.readback.unmap();
    } else {
      const out = await mb.session.run(feeds);
      tokens = Number(out.pred_num_tokens.data[0]);
      if (withMotion) motion = await out.pred_global_motions.getData(true);
      else out.pred_global_motions.dispose();
    }
  } else {
    const feeds = {};
    for (const name of INPUTS) {
      const slice = requestSlice(name, index);
      feeds[name] = new ort.Tensor(slice.type, slice.data, slice.dims);
    }
    const out = await mb.session.run(feeds);
    tokens = Number(out.pred_num_tokens.data[0]);
    motion = out.pred_global_motions.data;
  }
  return { ms: now() - start, tokens, motion };
}

function compareToReference(index, tokens, motion) {
  if (!reference || !motion) return null;
  const refTokens = reference.tokens[index];
  const frames = 4 * Math.min(tokens, refTokens);
  const base = index * 64 * 366;
  let maxDiff = 0;
  for (let i = 0; i < frames * 366; i += 1) {
    const diff = Math.abs(motion[i] - reference.motion[base + i]);
    if (!(diff <= maxDiff)) maxDiff = Number.isNaN(diff) ? Infinity : diff;
  }
  return { tokenMatch: tokens === refTokens, maxDiff };
}

async function benchMB({ count, warmup, withMotion = true, compare = true }) {
  await loadRequests();
  const total = Math.min(count, requests.count);
  for (let i = 0; i < warmup; i += 1) await runMB(i % total, withMotion);
  const latencies = [];
  const frames = [];
  const rtf = [];
  let tokenMatches = 0;
  let legalTokens = 0;
  let finiteOutputs = 0;
  const diffs = [];
  for (let i = 0; i < total; i += 1) {
    const result = await runMB(i, withMotion);
    latencies.push(result.ms);
    const valid = Math.max(0, 4 * result.tokens - 6);
    frames.push(valid);
    rtf.push(valid / 30 / (result.ms / 1000));
    const allowed = requestSlice("allowed_pred_num_tokens", i).data;
    if (result.tokens >= 6 && result.tokens <= 16 && allowed[result.tokens - 6] === 1) legalTokens += 1;
    if (result.motion) {
      let finite = true;
      for (let k = 0; k < 4 * result.tokens * 366; k += 1) if (!Number.isFinite(result.motion[k])) { finite = false; break; }
      if (finite) finiteOutputs += 1;
    }
    if (compare) {
      const check = compareToReference(i, result.tokens, result.motion);
      if (check) {
        if (check.tokenMatch) { tokenMatches += 1; diffs.push(check.maxDiff); }
      }
    }
  }
  return {
    latencyMs: stats(latencies),
    framesPerCall: stats(frames),
    realtimeFactor: stats(rtf),
    legalTokens,
    finiteOutputs: withMotion ? finiteOutputs : null,
    parityVsCpuFp32: compare && reference ? { tokenMatches, of: total, maxDiffMatched: Math.max(...diffs), p99DiffMatched: pct(diffs, 99), medianDiffMatched: pct(diffs, 50) } : null,
  };
}

async function loadCore({ allowCpuFallback = false } = {}) {
  const manifest = await (await fetch("/core/manifest.json")).json();
  const graphs = ["history-tokenizer.onnx", "denoiser.onnx", "decoder.onnx", "prompt-encoder/text_encoder.onnx"];
  const result = { sessions: {} };
  core = { sessions: {} };
  for (const graph of graphs) {
    const entry = manifest[graph];
    const graphBytes = await fetchBytes("/core/" + graph);
    const dataPath = entry.externalData;
    const dataBytes = await fetchBytes("/core/" + dataPath);
    const logStart = logLines.length;
    const createStart = now();
    const session = await ort.InferenceSession.create(graphBytes.bytes, {
      executionProviders: [epOptions()],
      externalData: [{ data: dataBytes.bytes, path: dataPath.split("/").pop() }],
      extra: { session: { disable_cpu_ep_fallback: allowCpuFallback || graph.startsWith("prompt-encoder") ? "0" : "1" } },
      logSeverityLevel: 1,
      logVerbosityLevel: 0,
    });
    core.sessions[graph] = session;
    result.sessions[graph] = {
      downloadBytes: graphBytes.bytes.byteLength + dataBytes.bytes.byteLength,
      downloadMs: graphBytes.ms + dataBytes.ms,
      createMs: now() - createStart,
      ...memcpySummary(logLines.slice(logStart)),
    };
  }
  return result;
}

function filled(type, dims, value) {
  const size = dims.reduce((a, b) => a * b, 1);
  if (type === "int64") return new ort.Tensor("int64", new BigInt64Array(size).fill(BigInt(value)), dims);
  return new ort.Tensor("float32", new Float32Array(size).fill(value), dims);
}

async function runCoreWindow({ withText = false } = {}) {
  const t = {};
  const start = now();
  if (withText) {
    const s = now();
    const ids = new ort.Tensor("int64", BigInt64Array.from({ length: 16 }, (_, i) => BigInt(100 + i)), [1, 16]);
    await core.sessions["prompt-encoder/text_encoder.onnx"].run({ input_ids: ids, attention_mask: filled("int64", [1, 16], 1), token_type_ids: filled("int64", [1, 16], 0) });
    t.textMs = now() - s;
  }
  let s = now();
  const hist = await core.sessions["history-tokenizer.onnx"].run({ motion: filled("float32", [1, 200, 330], 0.01), motion_pad_mask: filled("float32", [1, 200], 1), motion_len: filled("int64", [1], 40) });
  t.historyMs = now() - s;
  let x = filled("float32", [1, 50, 148], 0.1);
  const mask = filled("float32", [1, 200], 1);
  const tokenMask = filled("float32", [1, 50], 1);
  const text = filled("float32", [1, 1, 4096], 0.01);
  const motionMask = filled("float32", [1, 200, 330], 0);
  const observed = filled("float32", [1, 200, 330], 0);
  s = now();
  for (let step = 9; step >= 0; step -= 1) {
    const out = await core.sessions["denoiser.onnx"].run({
      cfg_weight_text: filled("float32", [1], 2),
      cfg_weight_cstr: filled("float32", [1], 1),
      x,
      history_len: filled("int64", [1], 40),
      generation_len: filled("int64", [1], 40),
      history_mask: mask,
      generation_mask: mask,
      history_token_mask: tokenMask,
      generation_token_mask: tokenMask,
      future_token_mask: tokenMask,
      text_feat: text,
      timesteps: filled("int64", [1], step),
      first_heading_angle: filled("float32", [1], 0),
      motion_mask: motionMask,
      observed_motion: observed,
    });
    x = out.output;
  }
  t.denoiseMs = now() - s;
  s = now();
  await core.sessions["decoder.onnx"].run({ hybrid_motion: x, motion_pad_mask: mask, motion_len: filled("int64", [1], 80), global_translation: hist.global_translation });
  t.decodeMs = now() - s;
  t.totalMs = now() - start;
  return t;
}

async function interleave({ rounds }) {
  const mbMs = [];
  const coreMs = [];
  for (let i = 0; i < rounds; i += 1) {
    const r = await runMB(i % requests.count, true);
    mbMs.push(r.ms);
    const c = await runCoreWindow();
    coreMs.push(c.totalMs);
  }
  return { mbLatencyMs: stats(mbMs), coreWindowMs: stats(coreMs) };
}

async function coreAlone({ rounds, withText }) {
  const windows = [];
  for (let i = 0; i < rounds; i += 1) windows.push(await runCoreWindow({ withText: withText && i === 0 }));
  return { first: windows[0], windowMs: stats(windows.slice(1).map((w) => w.totalMs)), denoiseMs: stats(windows.slice(1).map((w) => w.denoiseMs)) };
}

async function mbLoop({ durationMs }) {
  const end = now() + durationMs;
  const latencies = [];
  let i = 0;
  while (now() < end) { latencies.push((await runMB(i % requests.count, true)).ms); i += 1; }
  return { latencyMs: stats(latencies) };
}

async function repeat({ indices, times }) {
  const rows = [];
  for (const index of indices) {
    for (let k = 0; k < times; k += 1) {
      const result = await runMB(index, true);
      const check = compareToReference(index, result.tokens, result.motion);
      rows.push({ index, k, tokens: result.tokens, ref: reference.tokens[index], maxDiff: check && check.maxDiff, ms: result.ms });
    }
  }
  return rows;
}

async function release({ what }) {
  if ((what === "mb" || what === "all") && mb) { await mb.session.release(); mb = undefined; }
  if ((what === "core" || what === "all") && core) { for (const s of Object.values(core.sessions)) await s.release(); core = undefined; }
  return { released: what };
}

const handlers = { initDevice, repeat, loadMB, benchMB, loadCore, coreAlone, interleave, mbLoop, release, logs: async ({ grep }) => logLines.filter((l) => !grep || new RegExp(grep).test(l)).slice(-300) };

self.onmessage = async (event) => {
  const { id, type, args } = event.data;
  try {
    const result = await handlers[type](args || {});
    self.postMessage({ id, ok: true, result });
  } catch (error) {
    self.postMessage({ id, ok: false, error: String(error && error.stack || error), logTail: logLines.slice(-40) });
  }
};
self.postMessage({ id: 0, ok: true, result: "ready" });

