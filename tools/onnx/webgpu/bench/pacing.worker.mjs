// SPDX-License-Identifier: Apache-2.0
// Frame-pacing probe: renders a 1920x1080 frame every 16.667 ms on its own WebGPU
// device and records when the GPU finishes each frame. A hidden tab does not run
// requestAnimationFrame, so this measures GPU completion, not display presentation.

let device;
let pipeline;
let view;
const now = () => performance.now();

const SHADER = `
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array(vec2f(-1.0, -3.0), vec2f(-1.0, 1.0), vec2f(3.0, 1.0));
  return vec4f(p[i], 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  var c = vec3f(pos.xy / vec2f(1920.0, 1080.0), 0.5);
  for (var k = 0; k < 24; k++) { c = fract(sin(c * 12.9898 + vec3f(f32(k))) * 43758.5453); }
  return vec4f(c, 1.0);
}`;

async function init({ powerPreference = "high-performance" }) {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference });
  device = await adapter.requestDevice();
  const module = device.createShaderModule({ code: SHADER });
  pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs" },
    fragment: { module, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
  });
  view = device.createTexture({ size: [1920, 1080], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT }).createView();
  return { adapter: adapter.info && adapter.info.description };
}

function frame() {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({ colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
  pass.setPipeline(pipeline);
  pass.draw(3);
  pass.end();
  device.queue.submit([encoder.finish()]);
  return device.queue.onSubmittedWorkDone();
}

const pct = (values, q) => { const s = [...values].sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.max(0, Math.ceil((q / 100) * s.length) - 1))] : null; };
const stats = (v) => ({ n: v.length, p50: pct(v, 50), p95: pct(v, 95), p99: pct(v, 99), max: v.length ? Math.max(...v) : null });

let stopRequested = false;
async function run({ durationMs }) {
  stopRequested = false;
  const period = 1000 / 60;
  const start = now() + 50;
  const records = [];
  const pending = [];
  let k = 0;
  await new Promise((resolve) => {
    const tick = () => {
      const scheduled = start + k * period;
      const fired = now();
      const record = { scheduled, fired, done: NaN };
      records.push(record);
      pending.push(frame().then(() => { record.done = now(); }));
      k += 1;
      if (stopRequested || now() - start > durationMs) { resolve(); return; }
      setTimeout(tick, Math.max(0, start + k * period - now()));
    };
    setTimeout(tick, Math.max(0, start - now()));
  });
  await Promise.all(pending);
  const latency = records.map((r) => r.done - r.scheduled);
  const jitter = records.map((r) => r.fired - r.scheduled);
  const intervals = [];
  for (let i = 1; i < records.length; i += 1) intervals.push(Math.max(period, records[i].done - records[i - 1].done));
  const missed = records.filter((r, i) => i + 1 < records.length && r.done > records[i + 1].scheduled).length;
  return {
    frames: records.length,
    frameTimeMs: stats(intervals),
    scheduledToGpuDoneMs: stats(latency),
    timerJitterMs: stats(jitter),
    missedDeadlines: missed,
    passP95: pct(intervals, 95) <= 16.7,
    passP99: pct(intervals, 99) <= 33.3,
  };
}

self.onmessage = async (event) => {
  const { id, type, args } = event.data;
  try {
    if (type === "stop") { stopRequested = true; self.postMessage({ id, ok: true, result: "stopping" }); return; }
    const result = await (type === "init" ? init(args || {}) : run(args || {}));
    self.postMessage({ id, ok: true, result });
  } catch (error) {
    self.postMessage({ id, ok: false, error: String(error && error.stack || error) });
  }
};

self.postMessage({ id: 0, ok: true, result: "ready" });
