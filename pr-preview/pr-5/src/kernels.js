// WGSL kernels. Activations are f32. Quantized weights are u32 words (see src/quant.js for layouts).
// No `shader-f16`: halves are unpacked with unpack2x16float.

/** Max tokens one matmul dispatch handles per weight pass; longer batches use more dispatches along z. */
export const MAXT = 8;

const Q6K = /* wgsl */ `
fn byteAt(base: u32, i: u32) -> u32 { return (W[base + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xffu; }
// One value of a padded Q6_K block (53 words) starting at word 'base'.
fn q6k(base: u32, idx: u32) -> f32 {
  let half = idx >> 7u; let l2 = idx & 127u; let quarter = l2 >> 5u; let l = l2 & 31u;
  let qlb = byteAt(base, half * 64u + l + (quarter & 1u) * 32u);
  let lo = select(qlb & 15u, qlb >> 4u, quarter >= 2u);
  let hi = (byteAt(base, 128u + half * 32u + l) >> (quarter * 2u)) & 3u;
  let q = f32(i32(lo | (hi << 4u)) - 32);
  let sb = byteAt(base, 192u + half * 8u + (l >> 4u) + quarter * 2u);
  let sc = f32(select(i32(sb), i32(sb) - 256, sb >= 128u));
  let d = unpack2x16float(W[base + 52u]).x;
  return d * sc * q;
}`;

/** Y[t, r] = W[r, :] · X[t, :] for a Q4_0 weight. One workgroup per row, `mt` tokens per pass (1 for decode). */
export const matmulQ4_0 = (mt = MAXT) => /* wgsl */ `
struct P { rows: u32, K: u32, T: u32, dOff: u32, yStride: u32, yOff: u32, xStride: u32, accumulate: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
const WG = 64u;
const MT = ${mt}u;
var<workgroup> red: array<f32, ${64 * mt}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.y * 32768u + wid.x;
  let t0 = wid.z * MT;
  if (row >= p.rows) { return; }
  let nT = min(MT, p.T - t0);
  let nb = p.K / 32u;
  var acc: array<f32, ${mt}>;
  for (var b = lid.x; b < nb; b += WG) {
    let blk = row * nb + b;
    let dw = W[p.dOff + (blk >> 1u)];
    let d = select(unpack2x16float(dw).x, unpack2x16float(dw).y, (blk & 1u) == 1u);
    for (var w = 0u; w < 4u; w++) {
      let q = W[blk * 4u + w];
      for (var j = 0u; j < 4u; j++) {
        let byte = (q >> (j * 8u)) & 0xffu;
        let lo = (f32(byte & 15u) - 8.0) * d;
        let hi = (f32(byte >> 4u) - 8.0) * d;
        let k = b * 32u + w * 4u + j;
        for (var t = 0u; t < nT; t++) {
          let xb = (t0 + t) * p.xStride + k;
          acc[t] += lo * X[xb] + hi * X[xb + 16u];
        }
      }
    }
  }
  for (var t = 0u; t < MT; t++) { red[t * WG + lid.x] = acc[t]; }
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s >>= 1u) {
    if (lid.x < s) { for (var t = 0u; t < MT; t++) { red[t * WG + lid.x] += red[t * WG + lid.x + s]; } }
    workgroupBarrier();
  }
  if (lid.x < nT) {
    let i = (t0 + lid.x) * p.yStride + p.yOff + row;
    let v = red[lid.x * WG];
    Y[i] = select(v, Y[i] + v, p.accumulate == 1u);
  }
}`;

/** Y[r] = W[r, :] · X for a padded Q6_K weight, one token (the output head). */
export const matmulQ6_K = /* wgsl */ `
struct P { rows: u32, K: u32, xOff: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
${Q6K}
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.y * 32768u + wid.x;
  if (row >= p.rows) { return; }
  let nb = p.K / 256u;
  var acc = 0.0;
  for (var k = lid.x; k < p.K; k += 64u) {
    let base = (row * nb + k / 256u) * 53u;
    acc += q6k(base, k & 255u) * X[p.xOff + k];
  }
  red[lid.x] = acc;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) { if (lid.x < s) { red[lid.x] += red[lid.x + s]; } workgroupBarrier(); }
  if (lid.x == 0u) { Y[row] = red[0]; }
}`;

/** X[t, :] = row tokens[t] of a padded Q6_K table. */
export const embedQ6_K = /* wgsl */ `
struct P { K: u32, T: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> tok: array<u32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
${Q6K}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  let nb = p.K / 256u;
  for (var k = lid.x; k < p.K; k += 256u) {
    X[t * p.K + k] = q6k((tok[t] * nb + k / 256u) * 53u, k & 255u);
  }
}`;

/** X[t, :] = row tok[t] of a Q4_0 embedding table (K wide), dequantized. Scales start at word dOff. */
export const embedQ4_0 = /* wgsl */ `
struct P { K: u32, dOff: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> tok: array<u32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  let nb = p.K / 32u;
  for (var k = lid.x; k < p.K; k += 256u) {
    let blk = tok[t] * nb + k / 32u;
    let dw = W[p.dOff + (blk >> 1u)];
    let d = select(unpack2x16float(dw).x, unpack2x16float(dw).y, (blk & 1u) == 1u);
    let j = k % 32u; let jj = j % 16u;
    let byte = (W[blk * 4u + jj / 4u] >> ((jj % 4u) * 8u)) & 0xffu;
    let q = select(byte & 15u, byte >> 4u, j >= 16u);
    X[t * p.K + k] = (f32(q) - 8.0) * d;
  }
}`;

/** RMSNorm over rows of length N. mode 0: Y = norm(X)·w. mode 1: Y += norm(X)·w (residual add). */
export const rmsnorm = /* wgsl */ `
struct P { N: u32, mode: u32, eps: f32, wMod: u32, xOff: u32, yOff: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let xb = p.xOff + wid.x * p.N;
  let yb = p.yOff + wid.x * p.N;
  var s = 0.0;
  for (var i = lid.x; i < p.N; i += 128u) { let v = X[xb + i]; s += v * v; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] += red[lid.x + k]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(p.N) + p.eps);
  for (var i = lid.x; i < p.N; i += 128u) {
    let v = X[xb + i] * inv * w[i % p.wMod];
    if (p.mode == 1u) { Y[yb + i] += v; } else { Y[yb + i] = v; }
  }
}`;

/** NeoX RoPE in place on [T, H, 128]; θ_i = pos · base^(-2i/128) / factor_i. */
export const rope = /* wgsl */ `
struct P { H: u32, pos0: u32, base: f32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> V: array<f32>;
@group(0) @binding(2) var<storage, read> factors: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x / p.H;
  let base = wid.x * 128u;
  let i = lid.x;
  let theta = f32(p.pos0 + t) * pow(p.base, -2.0 * f32(i) / 128.0) / factors[i];
  let c = cos(theta); let s = sin(theta);
  let a = V[base + i]; let b = V[base + i + 64u];
  V[base + i] = a * c - b * s;
  V[base + i + 64u] = a * s + b * c;
}`;

/** Writes K and V rows [T, KVH, 128] into ring caches of `slots` slots at position pos0 + t, as f16 pairs in u32
 * (llama.cpp's default KV type, and half the memory). */
export const kvWriteFor = (KVH) => /* wgsl */ `
struct P { pos0: u32, slots: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> Vv: array<f32>;
@group(0) @binding(3) var<storage, read_write> KC: array<u32>;
@group(0) @binding(4) var<storage, read_write> VC: array<u32>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  let slot = (p.pos0 + t) % p.slots;
  for (var i = lid.x; i < ${KVH * 64}u; i += 128u) {
    let s = t * ${KVH * 128}u + 2u * i;
    KC[slot * ${KVH * 64}u + i] = pack2x16float(vec2<f32>(K[s], K[s + 1u]));
    VC[slot * ${KVH * 64}u + i] = pack2x16float(vec2<f32>(Vv[s], Vv[s + 1u]));
  }
}`;
export const kvWrite = kvWriteFor(4);

// Position held by cache slot j once positions up to maxPos are written (ring of 'slots').
const SLOTPOS = /* wgsl */ `
fn slotPos(j: u32, maxPos: u32, slots: u32) -> i32 {
  let back = (maxPos + slots - (j % slots)) % slots;
  return i32(maxPos) - i32(back);
}`;

/** S[t·H+h, j] = q·k / √128 with causal and window masks (−inf when hidden); H query heads share KVH KV heads. */
export const attnScoresFor = (H, KVH) => /* wgsl */ `
struct P { pos0: u32, T: u32, slots: u32, L: u32, window: u32, sStride: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> KC: array<u32>;
@group(0) @binding(3) var<storage, read_write> S: array<f32>;
${SLOTPOS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  let th = gid.y;
  if (j >= p.L) { return; }
  let t = th / ${H}u; let h = th % ${H}u;
  let qp = i32(p.pos0 + t);
  let kp = slotPos(j, p.pos0 + p.T - 1u, p.slots);
  var s = -3.0e38;
  if (kp >= 0 && kp <= qp && u32(qp - kp) < p.window) {
    let qb = th * 128u; let kb = j * ${KVH * 128}u + (h / ${H / KVH}u) * 128u;
    var acc = 0.0;
    for (var d = 0u; d < 64u; d++) {
      let k = unpack2x16float(KC[kb / 2u + d]);
      acc += Q[qb + 2u * d] * k.x + Q[qb + 2u * d + 1u] * k.y;
    }
    s = acc * 0.08838834764831845;
  }
  S[th * p.sStride + j] = s;
}`;
export const attnScores = attnScoresFor(16, 4);

/** Softmax in place over rows of length L. */
export const softmax = /* wgsl */ `
struct P { L: u32, stride: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> S: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let b = wid.x * p.stride;
  var m = -3.0e38;
  for (var j = lid.x; j < p.L; j += 256u) { m = max(m, S[b + j]); }
  red[lid.x] = m; workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] = max(red[lid.x], red[lid.x + k]); } workgroupBarrier(); }
  m = red[0]; workgroupBarrier();
  var s = 0.0;
  for (var j = lid.x; j < p.L; j += 256u) { let e = select(exp(S[b + j] - m), 0.0, S[b + j] <= -1.0e38); S[b + j] = e; s += e; }
  red[lid.x] = s; workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] += red[lid.x + k]; } workgroupBarrier(); }
  let inv = 1.0 / red[0];
  for (var j = lid.x; j < p.L; j += 256u) { S[b + j] *= inv; }
}`;

/** O[t, h, :] = sigmoid(G[t, h]) · Σ_j P[t·H+h, j] · V[j, h/(H/KVH), :]; without `gated`, no G and no sigmoid. */
export const attnOutFor = (H, KVH, gated = true) => /* wgsl */ `
struct P { L: u32, sStride: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> VC: array<u32>;
${gated ? `@group(0) @binding(3) var<storage, read> G: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;` : '@group(0) @binding(3) var<storage, read_write> O: array<f32>;'}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let th = wid.x; let h = th % ${H}u; let d = lid.x;
  var acc = 0.0;
  for (var j = 0u; j < p.L; j++) {
    let w = S[th * p.sStride + j];
    if (w != 0.0) { let e = j * ${KVH * 128}u + (h / ${H / KVH}u) * 128u + d; let v = unpack2x16float(VC[e / 2u]); acc += w * select(v.x, v.y, (e & 1u) == 1u); }
  }
${gated ? `  let g = 1.0 / (1.0 + exp(-G[th]));
  O[th * 128u + d] = acc * g;` : '  O[th * 128u + d] = acc;'}
}`;
export const attnOut = attnOutFor(16, 4);

/**
 * Fused prompt attention (flash style): O[t, h, :] = softmax(q·K / √128, masked) · V, times sigmoid(G[t, h]) when `gated`.
 * One workgroup per KV head and 16 query rows (16/G tokens × the G = H/KVH heads sharing that KV head). K and V stream
 * through shared memory 16 keys at a time with a running max and sum, so the T×H×L score matrix is never written.
 * Same masks as attnScoresFor (causal, window, ring slots). Grid: [ceil(T / (16/G)), KVH].
 */
export const attnFlashFor = (H, KVH, gated = true) => {
  const G = H / KVH, BT = 16 / G;
  return /* wgsl */ `
struct P { pos0: u32, T: u32, slots: u32, L: u32, window: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> KC: array<u32>;
@group(0) @binding(3) var<storage, read> VC: array<u32>;
${gated ? `@group(0) @binding(4) var<storage, read> Gt: array<f32>;
@group(0) @binding(5) var<storage, read_write> O: array<f32>;` : '@group(0) @binding(4) var<storage, read_write> O: array<f32>;'}
${SLOTPOS}
var<workgroup> Qs: array<f32, 2048>;  // 16 rows × 128
var<workgroup> KV: array<u32, 1040>;  // 16 keys × 64 f16 pairs, row stride 65 (no bank conflicts); K, then V
var<workgroup> Ps: array<f32, 256>;   // 16 rows × 16 scores
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = lid.x; let kvh = wid.y; let t0 = wid.x * ${BT}u;
  for (var e = i; e < 2048u; e += 256u) {
    let r = e / 128u; let t = t0 + r / ${G}u; let h = kvh * ${G}u + r % ${G}u;
    Qs[e] = select(0.0, Q[min((t * ${H}u + h) * 128u + e % 128u, arrayLength(&Q) - 1u)], t < p.T);
  }
  let r = i / 16u; let c = i % 16u;  // scores: row r, key c; output: row r, dim pairs c + 16n
  let qp = i32(p.pos0 + t0 + r / ${G}u);
  let maxPos = p.pos0 + p.T - 1u;
  var acc = array<vec2<f32>, 4>(vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0));
  var m = -3.0e38; var l = 0.0;
  for (var j0 = 0u; j0 < p.L; j0 += 16u) {
    workgroupBarrier();
    for (var e = i; e < 1024u; e += 256u) {
      let j = j0 + e / 64u;
      KV[(e / 64u) * 65u + e % 64u] = select(0u, KC[min(j, p.L - 1u) * ${KVH * 64}u + kvh * 64u + e % 64u], j < p.L);
    }
    workgroupBarrier();
    let j = j0 + c;
    var s = -3.0e38;
    let kp = slotPos(j, maxPos, p.slots);
    if (j < p.L && kp >= 0 && kp <= qp && u32(qp - kp) < p.window) {
      var a = 0.0;
      for (var d = 0u; d < 64u; d++) {
        let k = unpack2x16float(KV[c * 65u + d]);
        a += Qs[r * 128u + 2u * d] * k.x + Qs[r * 128u + 2u * d + 1u] * k.y;
      }
      s = a * 0.08838834764831845;
    }
    Ps[i] = s;
    workgroupBarrier();
    for (var e = i; e < 1024u; e += 256u) {
      let jv = j0 + e / 64u;
      KV[(e / 64u) * 65u + e % 64u] = select(0u, VC[min(jv, p.L - 1u) * ${KVH * 64}u + kvh * 64u + e % 64u], jv < p.L);
    }
    workgroupBarrier();
    // Every thread of row r keeps the same running max and sum (same inputs, same order).
    var mn = m;
    for (var k = 0u; k < 16u; k++) { mn = max(mn, Ps[r * 16u + k]); }
    let corr = exp(m - mn);
    l *= corr;
    for (var n = 0u; n < 4u; n++) { acc[n] *= corr; }
    for (var k = 0u; k < 16u; k++) {
      let sv = Ps[r * 16u + k];
      if (sv > -1.0e38) {
        let w = exp(sv - mn);
        l += w;
        for (var n = 0u; n < 4u; n++) { acc[n] += w * unpack2x16float(KV[k * 65u + c + 16u * n]); }
      }
    }
    m = mn;
  }
  let t = t0 + r / ${G}u; let h = kvh * ${G}u + r % ${G}u;
  if (t < p.T) {
    let th = t * ${H}u + h;
    let g = ${gated ? '1.0 / (1.0 + exp(-Gt[th]))' : '1.0'} / l;
    for (var n = 0u; n < 4u; n++) {
      let d = 2u * (c + 16u * n);
      O[th * 128u + d] = acc[n].x * g; O[th * 128u + d + 1u] = acc[n].y * g;
    }
  }
}`;
};
export const attnFlash = attnFlashFor(16, 4);

/** LFM short conv over T tokens: Y = C ⊙ conv7(B ⊙ X), keeping the last 6 inputs per channel as state. */
export const shortconv = /* wgsl */ `
struct P { T: u32, N: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> BCX: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> state: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ch = gid.x;
  if (ch >= p.N) { return; }
  var win: array<f32, 7>;
  for (var k = 0u; k < 6u; k++) { win[k] = state[k * p.N + ch]; }
  for (var t = 0u; t < p.T; t++) {
    let row = t * 3u * p.N;
    win[6] = BCX[row + ch] * BCX[row + 2u * p.N + ch];
    var acc = 0.0;
    for (var k = 0u; k < 7u; k++) { acc += win[k] * w[ch * 7u + k]; }
    Y[t * p.N + ch] = BCX[row + p.N + ch] * acc;
    for (var k = 0u; k < 6u; k++) { win[k] = win[k + 1u]; }
  }
  for (var k = 0u; k < 6u; k++) { state[k * p.N + ch] = win[k]; }
}`;

/** G = silu(G) ⊙ U, elementwise over n values. */
export const swiglu = /* wgsl */ `
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> G: array<f32>;
@group(0) @binding(2) var<storage, read> U: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * 65535u * 256u + gid.x;
  if (i >= p.n) { return; }
  let g = G[i];
  G[i] = g / (1.0 + exp(-g)) * U[i];
}`;

// ───────────── Shared by the Omni senses (encoders run all positions at once, non-causal) ─────────────

/** Y[t, r] = W[r, :] · X[t, :] (+ b[r]) for a Q8_0 weight. One workgroup per row, `mt` tokens per pass. */
export const matmulQ8_0 = (mt = MAXT) => /* wgsl */ `
struct P { rows: u32, K: u32, T: u32, dOff: u32, bias: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> B: array<f32>;
const WG = 64u;
const MT = ${mt}u;
var<workgroup> red: array<f32, ${64 * mt}>;
fn s8(w: u32, j: u32) -> f32 { let b = (w >> (j * 8u)) & 0xffu; return f32(select(i32(b), i32(b) - 256, b >= 128u)); }
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.y * 32768u + wid.x;
  let t0 = wid.z * MT;
  if (row >= p.rows) { return; }
  let nT = min(MT, p.T - t0);
  let nb = p.K / 32u;
  var acc: array<f32, ${mt}>;
  for (var b = lid.x; b < nb; b += WG) {
    let blk = row * nb + b;
    let dw = W[p.dOff + (blk >> 1u)];
    let d = select(unpack2x16float(dw).x, unpack2x16float(dw).y, (blk & 1u) == 1u);
    for (var w = 0u; w < 8u; w++) {
      let q = W[blk * 8u + w];
      for (var j = 0u; j < 4u; j++) {
        let v = s8(q, j) * d;
        let k = b * 32u + w * 4u + j;
        for (var t = 0u; t < nT; t++) { acc[t] += v * X[(t0 + t) * p.K + k]; }
      }
    }
  }
  for (var t = 0u; t < MT; t++) { red[t * WG + lid.x] = acc[t]; }
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s >>= 1u) {
    if (lid.x < s) { for (var t = 0u; t < MT; t++) { red[t * WG + lid.x] += red[t * WG + lid.x + s]; } }
    workgroupBarrier();
  }
  if (lid.x < nT) {
    var v = red[lid.x * WG];
    if (p.bias == 1u) { v += B[row]; }
    Y[(t0 + lid.x) * p.rows + row] = v;
  }
}`;

/** Y[t, r] = W[r, :] · X[t, :] (+ b[r]) for an f32 weight. */
export const matmulF32 = (mt = MAXT) => /* wgsl */ `
struct P { rows: u32, K: u32, T: u32, bias: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> B: array<f32>;
const WG = 64u;
const MT = ${mt}u;
var<workgroup> red: array<f32, ${64 * mt}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.y * 32768u + wid.x;
  let t0 = wid.z * MT;
  if (row >= p.rows) { return; }
  let nT = min(MT, p.T - t0);
  var acc: array<f32, ${mt}>;
  for (var k = lid.x; k < p.K; k += WG) {
    let w = W[row * p.K + k];
    for (var t = 0u; t < nT; t++) { acc[t] += w * X[(t0 + t) * p.K + k]; }
  }
  for (var t = 0u; t < MT; t++) { red[t * WG + lid.x] = acc[t]; }
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s >>= 1u) {
    if (lid.x < s) { for (var t = 0u; t < MT; t++) { red[t * WG + lid.x] += red[t * WG + lid.x + s]; } }
    workgroupBarrier();
  }
  if (lid.x < nT) {
    var v = red[lid.x * WG];
    if (p.bias == 1u) { v += B[row]; }
    Y[(t0 + lid.x) * p.rows + row] = v;
  }
}`;

/** LayerNorm with bias over rows of length N. */
export const layernorm = /* wgsl */ `
struct P { N: u32, eps: f32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let base = wid.x * p.N;
  var s = 0.0;
  for (var i = lid.x; i < p.N; i += 128u) { s += X[base + i]; }
  red[lid.x] = s; workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] += red[lid.x + k]; } workgroupBarrier(); }
  let mean = red[0] / f32(p.N); workgroupBarrier();
  var v = 0.0;
  for (var i = lid.x; i < p.N; i += 128u) { let d = X[base + i] - mean; v += d * d; }
  red[lid.x] = v; workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] += red[lid.x + k]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(p.N) + p.eps);
  for (var i = lid.x; i < p.N; i += 128u) { Y[base + i] = (X[base + i] - mean) * inv * w[i] + b[i]; }
}`;

/** In place: X = gelu(X) with erf (HF "gelu"). */
export const geluErf = /* wgsl */ `
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
// erf with |error| < 1.5e-7 (Abramowitz & Stegun 7.1.26)
fn erf(x: f32) -> f32 {
  let s = sign(x); let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return s * y;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * 65535u * 256u + gid.x;
  if (i >= p.n) { return; }
  let x = X[i];
  X[i] = 0.5 * x * (1.0 + erf(x * 0.7071067811865476));
}`;

/** Y = X + R elementwise (residual add into Y). */
export const addInto = /* wgsl */ `
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * 65535u * 256u + gid.x;
  if (i < p.n) { Y[i] += X[i]; }
}`;

/** LFM3-VL 2D RoPE in place on [N, H, 64]: pairs (i, i+32); i < 16 rotate by row y, i ≥ 16 by column x. */
export const rope2d = /* wgsl */ `
struct P { H: u32, pw: u32, base: f32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> V: array<f32>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wid.x / p.H;
  let base = wid.x * 64u;
  let i = lid.x;
  let y = f32(n / p.pw); let x = f32(n % p.pw);
  let k = select(i, i - 16u, i >= 16u);
  let pos = select(y, x, i >= 16u);
  let theta = pos * pow(p.base, -2.0 * f32(k) / 32.0);
  let c = cos(theta); let s = sin(theta);
  let a = V[base + i]; let b = V[base + i + 32u];
  V[base + i] = a * c - b * s;
  V[base + i + 32u] = a * s + b * c;
}`;

/**
 * Scores for a block of queries: S[(q·H + h), j] = Q[q0+q, h] · K[j, h / (H/HK)] · scale.
 * window 0: every key (non-causal). window w > 0: causal, keys within the last w positions.
 */
export const encScores = /* wgsl */ `
struct P { N: u32, q0: u32, H: u32, HK: u32, D: u32, scale: f32, window: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read_write> S: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x; let qh = gid.y;
  if (j >= p.N) { return; }
  let q = qh / p.H; let h = qh % p.H;
  let qb = ((p.q0 + q) * p.H + h) * p.D;
  let kb = (j * p.HK + h / (p.H / p.HK)) * p.D;
  let qa = p.q0 + q;
  if (p.window > 0u && (j > qa || qa - j >= p.window)) { S[qh * p.N + j] = -3.0e38; return; }
  var acc = 0.0;
  for (var d = 0u; d < p.D; d++) { acc += Q[qb + d] * K[kb + d]; }
  S[qh * p.N + j] = acc * p.scale;
}`;

/** O[q0+q, h, :] = Σ_j S[(q·H + h), j] · V[j, h / (H/HK), :]. One workgroup per (q, h), D threads. */
export const encOut = /* wgsl */ `
struct P { N: u32, q0: u32, H: u32, HK: u32, D: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let qh = wid.x; let q = qh / p.H; let h = qh % p.H;
  for (var d = lid.x; d < p.D; d += 64u) {
    var acc = 0.0;
    for (var j = 0u; j < p.N; j++) { acc += S[qh * p.N + j] * V[(j * p.HK + h / (p.H / p.HK)) * p.D + d]; }
    O[((p.q0 + q) * p.H + h) * p.D + d] = acc;
  }
}`;

/** LFM3-VL conv block core: Y[n, c] = middle · dwconv3x3(first ⊙ third) over the pw × ph patch grid (zero pad). */
export const dwconv3 = /* wgsl */ `
struct P { pw: u32, ph: u32, C: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x; let n = gid.y;
  if (c >= p.C) { return; }
  let y = i32(n / p.pw); let x = i32(n % p.pw);
  var acc = 0.0;
  for (var ky = 0; ky < 3; ky++) {
    for (var kx = 0; kx < 3; kx++) {
      let yy = y + ky - 1; let xx = x + kx - 1;
      if (yy >= 0 && xx >= 0 && yy < i32(p.ph) && xx < i32(p.pw)) {
        let m = u32(yy) * p.pw + u32(xx);
        let g = X[m * 3u * p.C + c] * X[m * 3u * p.C + 2u * p.C + c];
        acc += g * w[c * 9u + u32(ky) * 3u + u32(kx)];
      }
    }
  }
  Y[n * p.C + c] = X[n * 3u * p.C + p.C + c] * acc;
}`;

/** 2×2 patch merge: out[t] = [patch(2gx,2gy), patch(2gx+1,2gy), patch(2gx,2gy+1), patch(2gx+1,2gy+1)]. */
export const merge2x2 = /* wgsl */ `
struct P { pw: u32, C: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  let gw = p.pw / 2u;
  let gx = t % gw; let gy = t / gw;
  for (var k = lid.x; k < 4u * p.C; k += 256u) {
    let sub = k / p.C; let c = k % p.C;
    let px = 2u * gx + (sub & 1u); let py = 2u * gy + (sub >> 1u);
    Y[t * 4u * p.C + k] = X[(py * p.pw + px) * p.C + c];
  }
}`;

/**
 * Tiled Y[t, r] = W[r, :] · X[t, :] for a Q4_0 weight and many tokens (prompt reading).
 * Each workgroup computes 64 rows × 32 tokens through shared memory, so each weight block is read and
 * dequantized once per 32 tokens; each thread accumulates 4 rows × 2 tokens.
 * Same parameters as matmulQ4_0.
 */
export const gemmQ4_0 = /* wgsl */ `
struct P { rows: u32, K: u32, T: u32, dOff: u32, yStride: u32, yOff: u32, xStride: u32, accumulate: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> Ws: array<f32, 2048>; // [32 k][64 rows]
var<workgroup> Xs: array<f32, 1024>; // [32 k][32 tokens]
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row0 = wid.x * 64u;
  let tok0 = wid.y * 32u;
  let nb = p.K / 32u;
  let tr = li / 16u; let tc = li % 16u;
  var acc: array<f32, 8>;
  // loader roles
  let lr = li / 4u; let lq = li % 4u;
  let lrow = row0 + lr;
  for (var b = 0u; b < nb; b++) {
    // 64 rows × 1 block: each thread dequantizes one word (8 values) of one row's block.
    if (lrow < p.rows) {
      let blk = lrow * nb + b;
      let dw = W[p.dOff + (blk >> 1u)];
      let d = select(unpack2x16float(dw).x, unpack2x16float(dw).y, (blk & 1u) == 1u);
      let q = W[blk * 4u + lq];
      for (var j = 0u; j < 4u; j++) {
        let byte = (q >> (j * 8u)) & 0xffu;
        let k = lq * 4u + j;
        Ws[k * 64u + lr] = (f32(byte & 15u) - 8.0) * d;
        Ws[(k + 16u) * 64u + lr] = (f32(byte >> 4u) - 8.0) * d;
      }
    } else {
      for (var j = 0u; j < 4u; j++) { Ws[(lq * 4u + j) * 64u + lr] = 0.0; Ws[(lq * 4u + j + 16u) * 64u + lr] = 0.0; }
    }
    // 32 tokens × 32 values of X.
    for (var m = 0u; m < 4u; m++) {
      let idx = li + 256u * m;
      let t = idx / 32u; let k = idx % 32u;
      let tok = tok0 + t;
      Xs[k * 32u + t] = select(0.0, X[tok * p.xStride + b * 32u + k], tok < p.T);
    }
    workgroupBarrier();
    for (var k = 0u; k < 32u; k++) {
      let x0 = Xs[k * 32u + tc]; let x1 = Xs[k * 32u + tc + 16u];
      for (var i = 0u; i < 4u; i++) {
        let w = Ws[k * 64u + tr + 16u * i];
        acc[i * 2u] += w * x0;
        acc[i * 2u + 1u] += w * x1;
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let row = row0 + tr + 16u * i;
    if (row < p.rows) {
      for (var j = 0u; j < 2u; j++) {
        let tok = tok0 + tc + 16u * j;
        if (tok < p.T) { Y[tok * p.yStride + p.yOff + row] = acc[i * 2u + j]; }
      }
    }
  }
}`;

/**
 * Y[t, row] = Σ_k W[row, k]·X[t, k] for Q4_0 W, tiled 64 rows × 64 tokens per workgroup with a 4×4 register block
 * per thread: each k step reads one vec4 of weights and one of inputs from workgroup memory for 16 multiply-adds
 * (gemmQ4_0 does 8 for 6 reads). Grid: [ceil(rows / 64), ceil(T / 64)]. Same uniforms as gemmQ4_0.
 */
export const gemmQ4_0x64 = /* wgsl */ `
struct P { rows: u32, K: u32, T: u32, dOff: u32, yStride: u32, yOff: u32, xStride: u32, accumulate: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
// Half a Q4_0 block (16 k) at a time, as scalars: 8 KB of workgroup memory. The first version kept a whole block in
// vec4s (16 KB, exactly the default limit) with each vec4 filled one component per thread, and an iPhone 16 (Safari
// 27.2) gave all zeros for it while SwiftShader matched (bench/kernel-check.html, Nov 21 2026).
var<workgroup> Ws: array<f32, 1024>; // [16 k][64 rows]
var<workgroup> Xs: array<f32, 1024>; // [16 k][64 tokens]
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row0 = wid.x * 64u;
  let tok0 = wid.y * 64u;
  let nb = p.K / 32u;
  let tr = li / 16u; let tc = li % 16u; // this thread's rows tr·4.., tokens tc·4..
  var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0); var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0); // rows tr·4+0..3
  // Loader roles: W: row lr, word lq of its block (4 low nibbles: k = lq·4..+3; 4 high: k + 16). X: token xt, 4 values.
  let lr = li / 4u; let lq = li % 4u;
  let lrow = row0 + lr;
  let xt = li / 4u; let xk = (li % 4u) * 4u;
  let xtok = tok0 + xt;
  for (var b = 0u; b < nb; b++) {
    var lo = vec4<f32>(0.0); var hi = vec4<f32>(0.0);
    if (lrow < p.rows) {
      let blk = lrow * nb + b;
      let dw = W[p.dOff + (blk >> 1u)];
      let d = select(unpack2x16float(dw).x, unpack2x16float(dw).y, (blk & 1u) == 1u);
      let q = W[blk * 4u + lq];
      let b0 = q & 0xffu; let b1 = (q >> 8u) & 0xffu; let b2 = (q >> 16u) & 0xffu; let b3 = q >> 24u;
      lo = (vec4<f32>(f32(b0 & 15u), f32(b1 & 15u), f32(b2 & 15u), f32(b3 & 15u)) - 8.0) * d;
      hi = (vec4<f32>(f32(b0 >> 4u), f32(b1 >> 4u), f32(b2 >> 4u), f32(b3 >> 4u)) - 8.0) * d;
    }
    for (var half = 0u; half < 2u; half++) {
      let w4 = select(lo, hi, half == 1u);
      Ws[(lq * 4u + 0u) * 64u + lr] = w4.x;
      Ws[(lq * 4u + 1u) * 64u + lr] = w4.y;
      Ws[(lq * 4u + 2u) * 64u + lr] = w4.z;
      Ws[(lq * 4u + 3u) * 64u + lr] = w4.w;
      let xb = xtok * p.xStride + b * 32u + half * 16u + xk;
      let ok = xtok < p.T;
      Xs[(xk + 0u) * 64u + xt] = select(0.0, X[min(xb + 0u, arrayLength(&X) - 1u)], ok);
      Xs[(xk + 1u) * 64u + xt] = select(0.0, X[min(xb + 1u, arrayLength(&X) - 1u)], ok);
      Xs[(xk + 2u) * 64u + xt] = select(0.0, X[min(xb + 2u, arrayLength(&X) - 1u)], ok);
      Xs[(xk + 3u) * 64u + xt] = select(0.0, X[min(xb + 3u, arrayLength(&X) - 1u)], ok);
      workgroupBarrier();
      for (var k = 0u; k < 16u; k++) {
        let wb = k * 64u + tr * 4u; let xo = k * 64u + tc * 4u;
        let x = vec4<f32>(Xs[xo], Xs[xo + 1u], Xs[xo + 2u], Xs[xo + 3u]);
        a0 += Ws[wb] * x; a1 += Ws[wb + 1u] * x; a2 += Ws[wb + 2u] * x; a3 += Ws[wb + 3u] * x;
      }
      workgroupBarrier();
    }
  }
  let r = row0 + tr * 4u; let t = tok0 + tc * 4u;
  put(r, t, a0); put(r + 1u, t, a1); put(r + 2u, t, a2); put(r + 3u, t, a3);
}
fn put(row: u32, t: u32, v: vec4<f32>) {
  if (row >= p.rows) { return; }
  if (t < p.T) { Y[t * p.yStride + p.yOff + row] = v.x; }
  if (t + 1u < p.T) { Y[(t + 1u) * p.yStride + p.yOff + row] = v.y; }
  if (t + 2u < p.T) { Y[(t + 2u) * p.yStride + p.yOff + row] = v.z; }
  if (t + 3u < p.T) { Y[(t + 3u) * p.yStride + p.yOff + row] = v.w; }
}`;

/**
 * Top 16 of each 256-value slice of the logits: C[(slice·16 + i)·2] = value bits, C[… + 1] = token id.
 * Reading back these candidates (8 KB per 1,000 slices) replaces reading back every logit; the true top-k is in
 * them unless more than 16 of it share one slice of 256 neighbouring token ids.
 */
export const topk16 = /* wgsl */ `
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> L: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<u32>;
var<workgroup> v: array<f32, 256>;
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = wid.x * 256u + lid.x;
  v[lid.x] = select(-3.0e38, L[i], i < p.n);
  workgroupBarrier();
  for (var r = 0u; r < 16u; r++) {
    rv[lid.x] = v[lid.x]; ri[lid.x] = lid.x;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
      if (lid.x < s && rv[lid.x + s] > rv[lid.x]) { rv[lid.x] = rv[lid.x + s]; ri[lid.x] = ri[lid.x + s]; }
      workgroupBarrier();
    }
    let best = ri[0];
    if (lid.x == 0u) {
      C[(wid.x * 16u + r) * 2u] = bitcast<u32>(rv[0]);
      C[(wid.x * 16u + r) * 2u + 1u] = wid.x * 256u + best;
    }
    if (lid.x == best) { v[lid.x] = -3.0e38; }
    workgroupBarrier();
  }
}`;

/**
 * Single-token y = W·x for Q4_0: 8 rows per 256-thread workgroup (32 threads per row, striding over blocks),
 * so far fewer, fuller workgroups than one per row. Same parameters as matmulQ4_0 (T must be 1).
 */
export const matvecQ4_0 = /* wgsl */ `
struct P { rows: u32, K: u32, T: u32, dOff: u32, yStride: u32, yOff: u32, xStride: u32, accumulate: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let g = li / 32u; let lane = li % 32u;
  let row = (wid.y * 4096u + wid.x) * 8u + g;
  let nb = p.K / 32u;
  var acc = 0.0;
  if (row < p.rows) {
    for (var b = lane; b < nb; b += 32u) {
      let blk = row * nb + b;
      let dw = W[p.dOff + (blk >> 1u)];
      let d = select(unpack2x16float(dw).x, unpack2x16float(dw).y, (blk & 1u) == 1u);
      let xb = b * 32u;
      var s = 0.0;
      for (var w = 0u; w < 4u; w++) {
        let q = W[blk * 4u + w];
        for (var j = 0u; j < 4u; j++) {
          let byte = (q >> (j * 8u)) & 0xffu;
          let k = xb + w * 4u + j;
          s += (f32(byte & 15u) - 8.0) * X[k] + (f32(byte >> 4u) - 8.0) * X[k + 16u];
        }
      }
      acc += s * d;
    }
  }
  red[li] = acc;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  if (lane == 0u && row < p.rows) { Y[p.yOff + row] = red[li]; }
}`;


// ───────────── LFM3 audio encoder ─────────────

/**
 * Stride-2 3×3 conv over [channel][freq][time] (time fastest): one zero column before the clip and one after it
 * in time, one zero row on each side in frequency. depthwise: channel c reads input channel c; otherwise all
 * output channels read the single input channel. Y = silu?(conv + bias).
 */
export const convDown = /* wgsl */ `
struct P { Tin: u32, Fin: u32, Tout: u32, Fout: u32, C: u32, depthwise: u32, silu: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let to = gid.x; let fo = gid.y % p.Fout; let c = gid.y / p.Fout;
  if (to >= p.Tout || c >= p.C) { return; }
  let ci = select(0u, c, p.depthwise == 1u);
  var acc = b[c];
  for (var kf = 0u; kf < 3u; kf++) {
    let fi = i32(2u * fo + kf) - 1;
    if (fi < 0 || fi >= i32(p.Fin)) { continue; }
    for (var kt = 0u; kt < 3u; kt++) {
      let ti = i32(2u * to + kt) - 1;
      if (ti < 0 || ti >= i32(p.Tin)) { continue; }
      acc += w[c * 9u + kf * 3u + kt] * X[(ci * p.Fin + u32(fi)) * p.Tin + u32(ti)];
    }
  }
  if (p.silu == 1u) { acc = acc / (1.0 + exp(-acc)); }
  Y[(c * p.Fout + fo) * p.Tout + to] = acc;
}`;

/** 1×1 conv over channels for [channel][pixels]: Y[co][q] = silu(Σ_ci w[co, ci] · X[ci][q] + b[co]). */
export const pointwise = /* wgsl */ `
struct P { C: u32, Q: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x; let co = gid.y;
  if (q >= p.Q) { return; }
  var acc = b[co];
  for (var ci = 0u; ci < p.C; ci++) { acc += w[co * p.C + ci] * X[ci * p.Q + q]; }
  Y[co * p.Q + q] = acc / (1.0 + exp(-acc));
}`;

/** [channel][freq][time] → rows per time step: Y[t, c·F + f] = X[c][f][t]. */
export const flattenCFT = /* wgsl */ `
struct P { C: u32, F: u32, T: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  for (var k = lid.x; k < p.C * p.F; k += 256u) { Y[t * p.C * p.F + k] = X[k * p.T + t]; }
}`;

/** Causal short conv of width W over T tokens from a zero state: Y = C ⊙ conv(B ⊙ X). Channels N, BCX rows 3N. */
export const shortconvFresh = (W) => /* wgsl */ `
struct P { T: u32, N: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> BCX: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ch = gid.x; let t = gid.y;
  if (ch >= p.N) { return; }
  var acc = 0.0;
  for (var k = 0u; k < ${W}u; k++) {
    let ti = i32(t) - i32(${W - 1}u) + i32(k);
    if (ti >= 0) {
      let row = u32(ti) * 3u * p.N;
      acc += BCX[row + ch] * BCX[row + 2u * p.N + ch] * w[ch * ${W}u + k];
    }
  }
  Y[t * p.N + ch] = BCX[t * 3u * p.N + p.N + ch] * acc;
}`;

/** NeoX RoPE in place on [T, H, D] at positions pos0 + t: pairs (i, i + D/2), θ_i = pos · base^(-2i/D). */
export const ropeN = /* wgsl */ `
struct P { H: u32, pos0: u32, base: f32, D: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> V: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x / p.H;
  let base = wid.x * p.D;
  let half = p.D / 2u;
  for (var i = lid.x; i < half; i += 64u) {
    let theta = f32(p.pos0 + t) * pow(p.base, -2.0 * f32(i) / f32(p.D));
    let c = cos(theta); let s = sin(theta);
    let a = V[base + i]; let b = V[base + i + half];
    V[base + i] = a * c - b * s;
    V[base + i + half] = a * s + b * c;
  }
}`;

// ───────────── LFM3 speech generation (depthformer + codec) ─────────────

/** Rows idx[i] of a Q8_0 matrix (K columns), dequantized: Y[i, :]. idx are u32 in `I`. */
export const gatherQ8_0 = /* wgsl */ `
struct P { K: u32, dOff: u32, n: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> I: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = wid.x;
  let row = I[i];
  let nb = p.K / 32u;
  for (var k = lid.x; k < p.K; k += 64u) {
    let blk = row * nb + k / 32u;
    let dw = W[p.dOff + (blk >> 1u)];
    let d = select(unpack2x16float(dw).x, unpack2x16float(dw).y, (blk & 1u) == 1u);
    let j = k % 32u;
    let byte = (W[blk * 8u + j / 4u] >> ((j % 4u) * 8u)) & 0xffu;
    Y[i * p.K + k] = f32(select(i32(byte), i32(byte) - 256, byte >= 128u)) * d;
  }
}`;

/** Depthformer input: X[i] = P[i] + (mask[i] ? T[i] : M) for n positions of width K (mask as f32 0/1). */
export const depthInput = /* wgsl */ `
struct P { K: u32, n: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Pj: array<f32>;
@group(0) @binding(2) var<storage, read> T: array<f32>;
@group(0) @binding(3) var<storage, read> M: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.K * p.n) { return; }
  let i = idx / p.K; let k = idx % p.K;
  X[idx] = Pj[idx] + mask[i] * T[idx] + (1.0 - mask[i]) * M[k];
}`;

/** Y[k] = Σ_i X[i, k] over n rows of width K. */
export const sumRows = /* wgsl */ `
struct P { K: u32, n: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= p.K) { return; }
  var s = 0.0;
  for (var i = 0u; i < p.n; i++) { s += X[i * p.K + k]; }
  Y[k] = s;
}`;

/** RMSNorm with a different weight row per input row: Y[r] = norm(X[r]) · w[r]. */
export const rmsnormRowW = /* wgsl */ `
struct P { N: u32, eps: f32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let b = wid.x * p.N;
  var s = 0.0;
  for (var i = lid.x; i < p.N; i += 128u) { let v = X[b + i]; s += v * v; }
  red[lid.x] = s; workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] += red[lid.x + k]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(p.N) + p.eps);
  for (var i = lid.x; i < p.N; i += 128u) { Y[b + i] = X[b + i] * inv * w[b + i]; }
}`;

/** Per-group matvec on a Q8_0 matrix: row r uses input row r / rowsPerGroup (codebook logits). */
export const matvecQ8Grouped = /* wgsl */ `
struct P { rows: u32, K: u32, dOff: u32, rowsPerGroup: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 256>;
fn s8(w: u32, j: u32) -> f32 { let b = (w >> (j * 8u)) & 0xffu; return f32(select(i32(b), i32(b) - 256, b >= 128u)); }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let g = li / 32u; let lane = li % 32u;
  let row = (wid.y * 4096u + wid.x) * 8u + g;
  let nb = p.K / 32u;
  var acc = 0.0;
  if (row < p.rows) {
    let xb = (row / p.rowsPerGroup) * p.K;
    for (var b = lane; b < nb; b += 32u) {
      let blk = row * nb + b;
      let dw = W[p.dOff + (blk >> 1u)];
      let d = select(unpack2x16float(dw).x, unpack2x16float(dw).y, (blk & 1u) == 1u);
      var s = 0.0;
      for (var w = 0u; w < 8u; w++) {
        let q = W[blk * 8u + w];
        for (var j = 0u; j < 4u; j++) { s += s8(q, j) * X[xb + b * 32u + w * 4u + j]; }
      }
      acc += s * d;
    }
  }
  red[li] = acc;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) { if (lane < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  if (lane == 0u && row < p.rows) { Y[row] = red[li]; }
}`;

/** In place: O[t, h, :] *= sigmoid(G[t·H + h]) for heads of size D. */
export const gateHeads = /* wgsl */ `
struct P { H: u32, D: u32, n: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> O: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  O[i] *= 1.0 / (1.0 + exp(-G[i / p.D]));
}`;

/** Y[t·R + r, :] = X[t, :]: every row repeated R times (nearest upsampling). */
export const repeatRows = /* wgsl */ `
struct P { K: u32, R: u32, n: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let row = i / p.K; let k = i % p.K;
  Y[i] = X[(row / p.R) * p.K + k];
}`;

/** Copy rows idx[i] of an f32 table (K columns): Y[i, :] = W[idx[i], :]. */
export const gatherF32 = /* wgsl */ `
struct P { K: u32, n: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> I: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n * p.K) { return; }
  Y[i] = W[I[i / p.K] * p.K + i % p.K];
}`;

/**
 * Causal sliding-window attention over a ring KV cache with general head sizes (codec decoder).
 * Scores: S[(t·H + h), j] for query positions pos0 + t, cache slot j holding the latest position ≡ j (mod slots).
 */
export const ringScores = /* wgsl */ `
struct P { pos0: u32, T: u32, slots: u32, L: u32, window: u32, H: u32, HK: u32, D: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> KC: array<f32>;
@group(0) @binding(3) var<storage, read_write> S: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x; let th = gid.y;
  if (j >= p.L) { return; }
  let t = th / p.H; let h = th % p.H;
  let qp = i32(p.pos0 + t);
  let maxPos = p.pos0 + p.T - 1u;
  let back = (maxPos + p.slots - (j % p.slots)) % p.slots;
  let kp = i32(maxPos) - i32(back);
  var s = -3.0e38;
  if (kp >= 0 && kp <= qp && u32(qp - kp) < p.window) {
    let qb = th * p.D; let kb = (j * p.HK + h / (p.H / p.HK)) * p.D;
    var acc = 0.0;
    for (var d = 0u; d < p.D; d++) { acc += Q[qb + d] * KC[kb + d]; }
    s = acc * inverseSqrt(f32(p.D));
  }
  S[th * p.L + j] = s;
}`;

/** O[t, h, :] = Σ_j S[(t·H + h), j] · VC[j, h / (H/HK), :]. */
export const ringOut = /* wgsl */ `
struct P { L: u32, H: u32, HK: u32, D: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> VC: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let th = wid.x; let h = th % p.H;
  for (var d = lid.x; d < p.D; d += 64u) {
    var acc = 0.0;
    for (var j = 0u; j < p.L; j++) { acc += S[th * p.L + j] * VC[(j * p.HK + h / (p.H / p.HK)) * p.D + d]; }
    O[th * p.D + d] = acc;
  }
}`;

/** Writes T rows of width W into a ring cache of `slots` rows at positions pos0 + t. */
export const ringWrite = /* wgsl */ `
struct P { pos0: u32, slots: u32, W: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  let slot = (p.pos0 + t) % p.slots;
  for (var i = lid.x; i < p.W; i += 64u) { C[slot * p.W + i] = X[t * p.W + i]; }
}`;

// ---- Kokoro (src/models/kokoro.js). Activations are channel-major [C][T] unless a kernel says otherwise. ----

/**
 * 1D convolution as a 64×64-tiled implicit GEMM: Y[o, t] = alpha · (Σ_{c,k} W[o, c, k] · act(X[c, t·s + k·d − pad]) + b[o]
 * + beta · Y[o, t]). W is [OC][IC·K] (torch layout), act is leaky ReLU with `slope` (1: none), out-of-range X is 0.
 * X and Y start at word offsets xOff / yOff, so a conv can read or write one block of a channel concat.
 */
export const conv1d = /* wgsl */ `
struct P { OC: u32, IC: u32, K: u32, T: u32, L: u32, s: u32, pad: u32, d: u32,
  xOff: u32, yOff: u32, bias: u32, p0: u32, alpha: f32, beta: f32, slope: f32, p1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tx = lid.x; let ty = lid.y; let tid = ty * 16u + tx;
  let n0 = wid.x * 64u; let m0 = wid.y * 64u;
  let KK = p.IC * p.K;
  var acc: array<f32, 16>;
  for (var kb = 0u; kb < KK; kb += 16u) {
    for (var r = 0u; r < 4u; r++) {
      let e = tid + r * 256u;
      let m = e / 16u; let ka = e % 16u;
      var a = 0.0;
      if (m0 + m < p.OC && kb + ka < KK) { a = W[(m0 + m) * KK + kb + ka]; }
      As[ka * 64u + m] = a;
      let n = e % 64u; let kk = e / 64u;
      var b = 0.0;
      if (n0 + n < p.L && kb + kk < KK) {
        let c = (kb + kk) / p.K; let k = (kb + kk) % p.K;
        let ti = i32((n0 + n) * p.s + k * p.d) - i32(p.pad);
        if (ti >= 0 && ti < i32(p.T)) { b = X[p.xOff + c * p.T + u32(ti)]; b = select(b * p.slope, b, b > 0.0); }
      }
      Bs[kk * 64u + n] = b;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      var av: array<f32, 4>; var bv: array<f32, 4>;
      for (var i = 0u; i < 4u; i++) { av[i] = As[kk * 64u + ty + 16u * i]; bv[i] = Bs[kk * 64u + tx + 16u * i]; }
      for (var i = 0u; i < 4u; i++) { for (var j = 0u; j < 4u; j++) { acc[i * 4u + j] += av[i] * bv[j]; } }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let m = m0 + ty + 16u * i;
    if (m >= p.OC) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let n = n0 + tx + 16u * j;
      if (n >= p.L) { continue; }
      var v = acc[i * 4u + j];
      if (p.bias == 1u) { v += B[m]; }
      let o = p.yOff + m * p.L + n;
      if (p.beta != 0.0) { v += p.beta * Y[o]; }
      Y[o] = p.alpha * v;
    }
  }
}`;

/**
 * Transposed conv (kernel 2·s, stride s), cropped by `trim` on both sides: Y[o, q] for q < L·s, plus bias, input
 * through leaky ReLU (`slope`). W is [IC][OC][K]. shift 1 reflect-pads one step on the left (Y has L·s + 1 steps).
 */
export const convT1d = /* wgsl */ `
struct P { IC: u32, OC: u32, K: u32, T: u32, s: u32, trim: u32, Lo: u32, shift: u32, slope: f32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x; let o = gid.y;
  if (q >= p.Lo) { return; }
  var qi = q;
  if (p.shift == 1u) { qi = select(q - 1u, 1u, q == 0u); }
  let tt = qi + p.trim;
  var acc = B[o];
  for (var k = tt % p.s; k < p.K; k += p.s) {
    if (k > tt) { break; }
    let i = (tt - k) / p.s;
    if (i >= p.T) { continue; }
    for (var c = 0u; c < p.IC; c++) {
      var x = X[c * p.T + i];
      x = select(x * p.slope, x, x > 0.0);
      acc += W[(c * p.OC + o) * p.K + k] * x;
    }
  }
  Y[o * p.Lo + q] = acc;
}`;

/**
 * AdaIN: per channel (one workgroup each), Y = act(norm_t(X) · (1 + G[c]) + G[C + c]), norm over T with eps.
 * mode 0: none, 1: leaky ReLU (slope), 2: snake x + sin²(a·x)/a with a = A[c]. In place is fine.
 */
export const adain = /* wgsl */ `
struct P { C: u32, T: u32, mode: u32, slope: f32, eps: f32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read> A: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let c = wid.x; let base = c * p.T;
  var s = 0.0;
  for (var t = lid.x; t < p.T; t += 256u) { s += X[base + t]; }
  red[lid.x] = s; workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] += red[lid.x + k]; } workgroupBarrier(); }
  let mean = red[0] / f32(p.T); workgroupBarrier();
  var v = 0.0;
  for (var t = lid.x; t < p.T; t += 256u) { let e = X[base + t] - mean; v += e * e; }
  red[lid.x] = v; workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] += red[lid.x + k]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(p.T) + p.eps);
  let g = 1.0 + G[c]; let b = G[p.C + c];
  for (var t = lid.x; t < p.T; t += 256u) {
    var y = (X[base + t] - mean) * inv * g + b;
    if (p.mode == 1u) { y = select(y * p.slope, y, y > 0.0); }
    if (p.mode == 2u) { let a = A[c]; let sn = sin(a * y); y = y + sn * sn / a; }
    Y[base + t] = y;
  }
}`;

/** Y[yOff + i] = a · Y[yOff + i] + b · X[xOff + i] (a = 0: scaled copy). */
export const axpby = /* wgsl */ `
struct P { n: u32, xOff: u32, yOff: u32, a: f32, b: f32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * 65535u * 256u + gid.x;
  if (i >= p.n) { return; }
  let o = p.yOff + i;
  var v = p.b * X[p.xOff + i];
  if (p.a != 0.0) { v += p.a * Y[o]; }
  Y[o] = v;
}`;

/**
 * Upsampling step of a Kokoro AdainResBlk1d on [C][T] → [C][2T]. mode 0: depthwise ConvTranspose1d (kernel 3,
 * stride 2, padding 1, output padding 1; W [3][C], tap k at k·C) plus bias. mode 1: nearest (each step twice).
 */
export const up2 = /* wgsl */ `
struct P { C: u32, T: u32, mode: u32, p0: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x; let c = gid.y;
  if (q >= 2u * p.T) { return; }
  let i = q >> 1u; let x = X[c * p.T + i];
  var y = x;
  if (p.mode == 0u) {
    if ((q & 1u) == 0u) { y = W[p.C + c] * x; }
    else { var nx = 0.0; if (i + 1u < p.T) { nx = X[c * p.T + i + 1u]; } y = W[2u * p.C + c] * x + W[c] * nx; }
    y += B[c];
  }
  Y[c * 2u * p.T + q] = y;
}`;

/** Y[c · rows + r] = X[r · cols + c] (rows × cols → cols × rows). */
export const transpose = /* wgsl */ `
struct P { rows: u32, cols: u32, p0: u32, p1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x; let r = gid.y;
  if (c < p.cols && r < p.rows) { Y[c * p.rows + r] = X[r * p.cols + c]; }
}`;

/** LayerNorm over the channels of [C][n] for each step (gamma, beta, eps 1e-5), then leaky ReLU 0.2, in place. */
export const lnChanLeaky = /* wgsl */ `
struct P { C: u32, n: u32, p0: u32, p1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= p.n) { return; }
  var m = 0.0;
  for (var c = 0u; c < p.C; c++) { m += X[c * p.n + t]; }
  m /= f32(p.C);
  var v = 0.0;
  for (var c = 0u; c < p.C; c++) { let e = X[c * p.n + t] - m; v += e * e; }
  let inv = inverseSqrt(v / f32(p.C) + 1e-5);
  for (var c = 0u; c < p.C; c++) {
    let y = (X[c * p.n + t] - m) * inv * g[c] + b[c];
    X[c * p.n + t] = select(0.2 * y, y, y > 0.0);
  }
}`;

/**
 * Rows of the prosody input [n][N + S]: mode 0 writes X[t, :N] = Y[t, :] and X[t, N:] = style. mode 1 is
 * AdaLayerNorm: X[t, :N] = (1 + G[j]) · LN(Y[t])[j] + G[N + j] (eps 1e-5); the style columns stay. One workgroup per row.
 */
export const prosodyRows = /* wgsl */ `
struct P { N: u32, S: u32, mode: u32, p0: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Y: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x; let src = t * p.N; let dst = t * (p.N + p.S);
  if (p.mode == 0u) {
    for (var j = lid.x; j < p.N; j += 128u) { X[dst + j] = Y[src + j]; }
    for (var j = lid.x; j < p.S; j += 128u) { X[dst + p.N + j] = G[j]; }
    return;
  }
  var s = 0.0;
  for (var j = lid.x; j < p.N; j += 128u) { s += Y[src + j]; }
  red[lid.x] = s; workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] += red[lid.x + k]; } workgroupBarrier(); }
  let mean = red[0] / f32(p.N); workgroupBarrier();
  var v = 0.0;
  for (var j = lid.x; j < p.N; j += 128u) { let e = Y[src + j] - mean; v += e * e; }
  red[lid.x] = v; workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] += red[lid.x + k]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(p.N) + 1e-5);
  for (var j = lid.x; j < p.N; j += 128u) { X[dst + j] = (1.0 + G[j]) * (Y[src + j] - mean) * inv + G[p.N + j]; }
}`;

/**
 * Bidirectional LSTM recurrence (torch gates i f g o, hidden 256): workgroup 0 runs forward, 1 backward.
 * G0/G1 hold the input projections plus biases [n][1024]; U0/U1 are W_hh transposed, [256][1024]. Y is [n][512].
 */
export const lstm = /* wgsl */ `
struct P { n: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> G0: array<f32>;
@group(0) @binding(2) var<storage, read> G1: array<f32>;
@group(0) @binding(3) var<storage, read> U0: array<f32>;
@group(0) @binding(4) var<storage, read> U1: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
var<workgroup> h: array<f32, 256>;
fn sig(x: f32) -> f32 { return 1.0 / (1.0 + exp(-clamp(x, -30.0, 30.0))); }
// tanh without the inf/inf of a fast-math exp form
fn th(x: f32) -> f32 { let u = clamp(x, -15.0, 15.0); return 1.0 - 2.0 / (exp(2.0 * u) + 1.0); }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let dir = wid.x; let j = lid.x;
  h[j] = 0.0;
  var c = 0.0;
  workgroupBarrier();
  for (var s = 0u; s < p.n; s++) {
    let t = select(s, p.n - 1u - s, dir == 1u);
    var a = vec4<f32>(0.0);
    if (dir == 0u) {
      a = vec4<f32>(G0[t * 1024u + j], G0[t * 1024u + 256u + j], G0[t * 1024u + 512u + j], G0[t * 1024u + 768u + j]);
      for (var i = 0u; i < 256u; i++) {
        let hv = h[i]; let r = i * 1024u + j;
        a += hv * vec4<f32>(U0[r], U0[r + 256u], U0[r + 512u], U0[r + 768u]);
      }
    } else {
      a = vec4<f32>(G1[t * 1024u + j], G1[t * 1024u + 256u + j], G1[t * 1024u + 512u + j], G1[t * 1024u + 768u + j]);
      for (var i = 0u; i < 256u; i++) {
        let hv = h[i]; let r = i * 1024u + j;
        a += hv * vec4<f32>(U1[r], U1[r + 256u], U1[r + 512u], U1[r + 768u]);
      }
    }
    c = sig(a.y) * c + sig(a.x) * th(a.z);
    let hn = sig(a.w) * th(c);
    workgroupBarrier();
    h[j] = hn;
    workgroupBarrier();
    Y[t * 512u + dir * 256u + j] = hn;
  }
}`;

/** ALBERT input: Y[t, i] = Tok[I[t], i] + Pos[t, i] for i < e (the token table rows are E wide, zero-padded). */
export const embedAdd = /* wgsl */ `
struct P { E: u32, e: u32, n: u32, p0: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Tok: array<f32>;
@group(0) @binding(2) var<storage, read> Pos: array<f32>;
@group(0) @binding(3) var<storage, read> I: array<u32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= p.n * p.e) { return; }
  let t = k / p.e; let i = k % p.e;
  Y[k] = Tok[I[t] * p.E + i] + Pos[t * p.e + i];
}`;

/** In place: X = gelu(X) with the tanh approximation (ggml_gelu). */
export const geluTanh = /* wgsl */ `
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * 65535u * 256u + gid.x;
  if (i >= p.n) { return; }
  let x = X[i];
  // tanh(u) = 1 - 2 / (exp(2u) + 1), clamped so exp stays finite
  let u = clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -15.0, 15.0);
  X[i] = 0.5 * x * (2.0 - 2.0 / (exp(2.0 * u) + 1.0));
}`;

/**
 * conv1d for stride 1 (same uniforms and bindings): each workgroup stages a window of 16 input channels once and
 * reuses it for every tap, so X is read about once instead of K times and the inner loop has no div/mod.
 * Window 64 + (K-1)·d must fit in 192 (K 11, d 5 is 114).
 */
export const conv1dWin = /* wgsl */ `
struct P { OC: u32, IC: u32, K: u32, T: u32, L: u32, s: u32, pad: u32, d: u32,
  xOff: u32, yOff: u32, bias: u32, p0: u32, alpha: f32, beta: f32, slope: f32, p1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> As: array<f32, 1024>;
var<workgroup> Xs: array<f32, 3072>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tx = lid.x; let ty = lid.y; let tid = ty * 16u + tx;
  let n0 = wid.x * 64u; let m0 = wid.y * 64u;
  let win = 64u + (p.K - 1u) * p.d;
  let KK = p.IC * p.K;
  var acc: array<f32, 16>;
  for (var cb = 0u; cb < p.IC; cb += 16u) {
    for (var e = tid; e < 16u * win; e += 256u) {
      let c = e / win; let x = e - c * win;
      let ti = i32(n0 + x) - i32(p.pad);
      var b = 0.0;
      if (cb + c < p.IC && ti >= 0 && ti < i32(p.T)) { b = X[p.xOff + (cb + c) * p.T + u32(ti)]; b = select(b * p.slope, b, b > 0.0); }
      Xs[c * 192u + x] = b;
    }
    for (var k = 0u; k < p.K; k++) {
      for (var r = 0u; r < 4u; r++) {
        let e = tid + r * 256u;
        let m = e >> 4u; let c = e & 15u;
        var a = 0.0;
        if (m0 + m < p.OC && cb + c < p.IC) { a = W[(m0 + m) * KK + (cb + c) * p.K + k]; }
        As[c * 64u + m] = a;
      }
      workgroupBarrier();
      let sh = k * p.d;
      for (var c = 0u; c < 16u; c++) {
        var av: array<f32, 4>; var bv: array<f32, 4>;
        for (var i = 0u; i < 4u; i++) { av[i] = As[c * 64u + ty + 16u * i]; bv[i] = Xs[c * 192u + tx + 16u * i + sh]; }
        for (var i = 0u; i < 4u; i++) { for (var j = 0u; j < 4u; j++) { acc[i * 4u + j] += av[i] * bv[j]; } }
      }
      workgroupBarrier();
    }
  }
  for (var i = 0u; i < 4u; i++) {
    let m = m0 + ty + 16u * i;
    if (m >= p.OC) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let n = n0 + tx + 16u * j;
      if (n >= p.L) { continue; }
      var v = acc[i * 4u + j];
      if (p.bias == 1u) { v += B[m]; }
      let o = p.yOff + m * p.L + n;
      if (p.beta != 0.0) { v += p.beta * Y[o]; }
      Y[o] = p.alpha * v;
    }
  }
}`;

/**
 * Stride-1 conv1d like conv1dWin, with the accumulators in vec4 registers (no dynamically indexed arrays, which
 * WebKit's Metal backend can leave in memory). Each thread owns 4 output channels × `cols` (4 or 8) contiguous
 * steps, so a workgroup covers 64 channels × 16·cols steps. Dispatch [ceil(L / (16·cols)), ceil(OC / 64)].
 */
export const conv1dRegSrc = (cols) => {
  const nv = cols / 4, tile = 16 * cols;
  const accs = [...Array(4 * nv)].map((_, i) => `var a${i} = vec4<f32>(0.0);`).join(' ');
  const bvs = [...Array(nv)].map((_, v) => `let b${v} = vec4<f32>(Xs[xb + ${4 * v}u], Xs[xb + ${4 * v + 1}u], Xs[xb + ${4 * v + 2}u], Xs[xb + ${4 * v + 3}u]);`).join('\n        ');
  const fmas = [...Array(4)].map((_, i) => [...Array(nv)].map((_, v) => `a${i * nv + v} += av.${'xyzw'[i]} * b${v};`).join(' ')).join('\n        ');
  const outs = [...Array(4)].map((_, i) => [...Array(nv)].map((_, v) => `out(m0 + ty * 4u + ${i}u, n0 + tx * ${cols}u + ${4 * v}u, a${i * nv + v});`).join(' ')).join('\n  ');
  return /* wgsl */ `
struct P { OC: u32, IC: u32, K: u32, T: u32, L: u32, s: u32, pad: u32, d: u32,
  xOff: u32, yOff: u32, bias: u32, p0: u32, alpha: f32, beta: f32, slope: f32, p1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
const XW = ${tile + 64}u;
var<workgroup> As: array<f32, 1024>;
var<workgroup> Xs: array<f32, ${16 * (tile + 64)}>;
fn out(m: u32, n: u32, v: vec4<f32>) {
  if (m >= p.OC) { return; }
  var b = 0.0;
  if (p.bias == 1u) { b = B[m]; }
  for (var j = 0u; j < 4u; j++) {
    if (n + j >= p.L) { return; }
    let o = p.yOff + m * p.L + n + j;
    var y = v[j] + b;
    if (p.beta != 0.0) { y += p.beta * Y[o]; }
    Y[o] = p.alpha * y;
  }
}
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tx = lid.x; let ty = lid.y; let tid = ty * 16u + tx;
  let n0 = wid.x * ${tile}u; let m0 = wid.y * 64u;
  let win = ${tile}u + (p.K - 1u) * p.d;
  let KK = p.IC * p.K;
  ${accs}
  for (var cb = 0u; cb < p.IC; cb += 16u) {
    for (var e = tid; e < 16u * win; e += 256u) {
      let c = e / win; let x = e - c * win;
      let ti = i32(n0 + x) - i32(p.pad);
      var b = 0.0;
      if (cb + c < p.IC && ti >= 0 && ti < i32(p.T)) { b = X[p.xOff + (cb + c) * p.T + u32(ti)]; b = select(b * p.slope, b, b > 0.0); }
      Xs[c * XW + x] = b;
    }
    for (var k = 0u; k < p.K; k++) {
      for (var r = 0u; r < 4u; r++) {
        let e = tid + r * 256u;
        let m = e >> 4u; let c = e & 15u;
        var a = 0.0;
        if (m0 + m < p.OC && cb + c < p.IC) { a = W[(m0 + m) * KK + (cb + c) * p.K + k]; }
        As[c * 64u + m] = a;
      }
      workgroupBarrier();
      let sh = k * p.d;
      for (var c = 0u; c < 16u; c++) {
        let ab = c * 64u + ty * 4u;
        let av = vec4<f32>(As[ab], As[ab + 1u], As[ab + 2u], As[ab + 3u]);
        let xb = c * XW + tx * ${cols}u + sh;
        ${bvs}
        ${fmas}
      }
      workgroupBarrier();
    }
  }
  ${outs}
}`;
};
