// WebGPU device, counted allocations, pipeline cache and timed submits.

/**
 * Opens a WebGPU device with the largest buffer limits the adapter allows.
 * @returns {Promise<Gpu>}
 */
export async function openGpu({ powerPreference = 'high-performance' } = {}) {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference });
  if (!adapter) throw new Error('No WebGPU adapter.');
  const L = adapter.limits;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: L.maxBufferSize,
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: Math.min(L.maxStorageBuffersPerShaderStage, 10),
      maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX,
    },
  });
  return new Gpu(adapter, device);
}

export class Gpu {
  constructor(adapter, device) {
    this.adapter = adapter;
    this.device = device;
    /** Bytes currently allocated through this object, by tag. */
    this.bytes = new Map();
    this.total = 0;
    this.peak = 0;
    this.pipelines = new Map();
    /** Longest single submit seen, in ms. */
    this.longestSubmit = 0;
    this.lost = null;
    // After WebKit's watchdog takes the device, onSubmittedWorkDone and mapAsync never settle; every wait races this
    // so a lost GPU fails fast instead of hanging.
    this.whenLost = device.lost.then((info) => {
      this.lost = info;
      throw Object.assign(new Error(`GPU lost: ${info.message || info.reason}`), { gpuLost: info });
    });
    this.whenLost.catch(() => {});
    // Per-submit uniforms: every dispatch gets a 256-byte slot in one buffer, written once before the submit,
    // instead of a new GPUBuffer per dispatch (each one is a round trip to Safari's GPU process).
    this.SLOTS = 2048;
    this.frame = { buf: null, data: new ArrayBuffer(this.SLOTS * 256), slot: 0 };
    this.frame.buf = this.buffer(this.SLOTS * 256, 'uniforms', GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    /** Bind groups by "plan:slot", reused while the pipeline and buffers are the same (e.g. every decode step). */
    this.bindGroups = new Map();
    this.plan = null;
  }

  /** Names the sequence of dispatches that follows, so its bind groups can be reused next time (null: don't cache). */
  beginPlan(name) { this.plan = name; }

  get maxBinding() { return this.device.limits.maxStorageBufferBindingSize; }

  /** Allocates a storage buffer and counts it under `tag`. */
  buffer(size, tag = 'misc', usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
    const padded = Math.max(16, Math.ceil(size / 16) * 16);
    if (padded > this.device.limits.maxBufferSize) throw new Error(`${tag}: ${padded} bytes is over this GPU's buffer limit`);
    const b = this.device.createBuffer({ size: padded, usage });
    b.__tag = tag;
    this.bytes.set(tag, (this.bytes.get(tag) || 0) + padded);
    this.total += padded;
    this.peak = Math.max(this.peak, this.total);
    return b;
  }

  /** Frees a buffer made by `buffer()`. */
  free(b) {
    if (!b || b.__freed) return;
    b.__freed = true;
    this.bytes.set(b.__tag, this.bytes.get(b.__tag) - b.size);
    this.total -= b.size;
    b.destroy();
  }

  uniform(data) {
    const b = this.device.createBuffer({ size: Math.ceil(data.byteLength / 16) * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(b, 0, data);
    return b;
  }

  /** Compiles (once) a compute pipeline for WGSL source. */
  pipeline(code) {
    let p = this.pipelines.get(code);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      p = this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      this.pipelines.set(code, p);
    }
    return p;
  }

  /**
   * Records one dispatch. `bindings` are bound at 0..n in order: GPU buffers, or (first) a Uint32Array of
   * uniform values, which goes into this submit's uniform slots.
   */
  dispatch(enc, code, bindings, groups) {
    const p = this.pipeline(code);
    const res = bindings.map((b) => {
      if (!ArrayBuffer.isView(b)) return b;
      const f = this.frame;
      if (f.slot >= this.SLOTS) throw new Error('too many dispatches in one submit');
      new Uint32Array(f.data, f.slot * 256, b.length).set(new Uint32Array(b.buffer, b.byteOffset, b.length));
      return { buffer: f.buf, offset: 256 * f.slot++, size: 256 };
    });
    const key = this.plan && `${this.plan}:${this.frame.slot}`;
    let hit = key && this.bindGroups.get(key);
    if (!hit || hit.p !== p || hit.res.length !== res.length || hit.res.some((r, i) => (r.buffer || r) !== (res[i].buffer || res[i]))) {
      const bg = this.device.createBindGroup({
        layout: p.getBindGroupLayout(0),
        entries: res.map((r, i) => ({ binding: i, resource: r.buffer ? r : { buffer: r } })),
      });
      hit = { p, res, bg };
      if (key) {
        if (this.bindGroups.size > 4096) this.bindGroups.clear();
        this.bindGroups.set(key, hit);
      }
    }
    const bg = hit.bg;
    // All dispatches recorded into one encoder share one compute pass (WebGPU still orders their reads and
    // writes); a pass per dispatch costs real time in Safari's Metal backend.
    if (this.passEnc !== enc) { this.endPass(); this.passEnc = enc; this.pass = enc.beginComputePass(); }
    const pass = this.pass;
    pass.setPipeline(p);
    pass.setBindGroup(0, bg);
    const [x, y = 1, z = 1] = Array.isArray(groups) ? groups : [groups];
    pass.dispatchWorkgroups(x, y, z);
  }

  /** Ends the open compute pass; call before recording copies into the same encoder, or finishing it. */
  endPass() {
    if (this.pass) { this.pass.end(); this.pass = null; this.passEnc = null; }
  }

  /** Submits `record(encoder)` as one command buffer and waits for the GPU, timing it. */
  /** Writes this submit's uniform slots to the GPU (queued ahead of the submit) and starts over. */
  flush() {
    const f = this.frame;
    if (f.slot) this.device.queue.writeBuffer(f.buf, 0, f.data, 0, f.slot * 256);
    f.slot = 0;
  }

  /** Waits for p, or throws as soon as the device is lost. */
  #orLost(p) { return Promise.race([p, this.whenLost]); }

  /** Waits for all submitted work, or throws as soon as the device is lost. */
  done() { return this.#orLost(this.device.queue.onSubmittedWorkDone()); }

  async run(record, enc = this.device.createCommandEncoder()) {
    if (this.lost) throw new Error(`GPU lost: ${this.lost.message || this.lost.reason}`);
    record(enc);
    this.endPass();
    const t0 = performance.now();
    this.flush();
    this.device.queue.submit([enc.finish()]);
    await this.#orLost(this.device.queue.onSubmittedWorkDone());
    const ms = performance.now() - t0;
    this.longestSubmit = Math.max(this.longestSubmit, ms);
    if (this.lost) throw new Error(`GPU lost: ${this.lost.message || this.lost.reason}`);
    return ms;
  }

  /** Copies a buffer back to the CPU. */
  async read(buf, size = buf.size, offset = 0) {
    const staging = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, offset, staging, 0, size);
    this.device.queue.submit([enc.finish()]);
    await this.#orLost(staging.mapAsync(GPUMapMode.READ));
    const out = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return out;
  }

  /** Memory report in MB by tag. */
  report() {
    const mb = (n) => Math.round(n / 1048576);
    return { totalMB: mb(this.total), peakMB: mb(this.peak), byTag: Object.fromEntries([...this.bytes].map(([k, v]) => [k, mb(v)])), longestSubmitMs: Math.round(this.longestSubmit) };
  }
}
