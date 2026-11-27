// CPU reference for Kokoro (src/models/kokoro.js): plain loops, a line-by-line port of llama.cpp's
// tools/mtmd/models/kokoro.cpp and src/models/kokoro.cpp. Slow; for checking the GPU path on small inputs.
// Activations are channel-major [C][T] like ggml's [T, C]; `w(name)` gives { dims (ggml order), data: Float32Array }.

import { harSource, istft, alignIndex, KOKORO } from '../models/kokoro.js';

const SQRT1_2 = Math.SQRT1_2;
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));

/** y[o] = W[o, :] · x (+ b), W row-major [rows][n]. */
function matvec(W, x, rows, n, b, xOff = 0) {
  const y = new Float32Array(rows);
  for (let o = 0; o < rows; o++) {
    let s = b ? b[o] : 0;
    for (let i = 0; i < n; i++) s += W[o * n + i] * x[xOff + i];
    y[o] = s;
  }
  return y;
}

/** ggml_conv_1d: w [K, IC, OC], x [IC][T] → [OC][L] (+ bias). */
export function conv1d(w, b, x, T, s = 1, p = 0, d = 1) {
  const [K, IC, OC] = w.dims, W = w.data;
  const L = Math.floor((T + 2 * p - d * (K - 1) - 1) / s) + 1;
  const y = new Float32Array(OC * L);
  for (let o = 0; o < OC; o++) for (let t = 0; t < L; t++) {
    let acc = b ? b.data[o] : 0;
    for (let c = 0; c < IC; c++) for (let k = 0; k < K; k++) {
      const ti = t * s + k * d - p;
      if (ti >= 0 && ti < T) acc += W[(o * IC + c) * K + k] * x[c * T + ti];
    }
    y[o * L + t] = acc;
  }
  return { y, L };
}

/** Per-channel norm over time (ggml_norm on [T, C]), eps 1e-5. */
function normTime(x, C, T, eps = 1e-5) {
  const y = new Float32Array(C * T);
  for (let c = 0; c < C; c++) {
    let m = 0; for (let t = 0; t < T; t++) m += x[c * T + t]; m /= T;
    let v = 0; for (let t = 0; t < T; t++) v += (x[c * T + t] - m) ** 2; v /= T;
    const inv = 1 / Math.sqrt(v + eps);
    for (let t = 0; t < T; t++) y[c * T + t] = (x[c * T + t] - m) * inv;
  }
  return y;
}

export class KokoroRef {
  /** @param {(name: string) => { dims: number[], data: Float32Array }} w */
  constructor(w, has) { this.w = w; this.has = has; }

  adain(name, x, C, T, style) {
    const fc = this.w(name + '.fc.weight'), h = matvec(fc.data, style, 2 * C, style.length, this.w(name + '.fc.bias').data);
    const y = normTime(x, C, T);
    for (let c = 0; c < C; c++) for (let t = 0; t < T; t++) y[c * T + t] = y[c * T + t] * (1 + h[c]) + h[C + c];
    return y;
  }

  snake(name, x, C, T) {
    const a = this.w(name).data, y = new Float32Array(x.length);
    for (let c = 0; c < C; c++) for (let t = 0; t < T; t++) { const v = x[c * T + t], s = Math.sin(a[c] * v); y[c * T + t] = v + (s * s) / a[c]; }
    return y;
  }

  conv(name, x, T, s = 1, p = 0, d = 1) {
    return conv1d(this.w(name + '.weight'), this.has(name + '.bias') ? this.w(name + '.bias') : null, x, T, s, p, d);
  }

  /** AdainResBlk1d; upsample doubles the length. Returns { y, C, T }. */
  resblk(name, x, C, T, style, upsample) {
    let r = this.adain(name + '.norm1', x, C, T, style).map((v) => (v > 0 ? v : 0.2 * v));
    let Tr = T;
    if (upsample) {
      const w = this.w(name + '.pool.weight').data, b = this.w(name + '.pool.bias').data, u = new Float32Array(C * 2 * T);
      for (let c = 0; c < C; c++) for (let i = 0; i < T; i++) {
        const next = i + 1 < T ? r[c * T + i + 1] : 0;
        u[c * 2 * T + 2 * i] = w[C + c] * r[c * T + i] + b[c];
        u[c * 2 * T + 2 * i + 1] = w[2 * C + c] * r[c * T + i] + w[c] * next + b[c];
      }
      r = u; Tr = 2 * T;
    }
    let o = this.conv(name + '.conv1', r, Tr, 1, 1);
    const C2 = o.y.length / Tr;
    r = this.adain(name + '.norm2', o.y, C2, Tr, style).map((v) => (v > 0 ? v : 0.2 * v));
    o = this.conv(name + '.conv2', r, Tr, 1, 1);
    let sc = x;
    if (upsample) { sc = new Float32Array(C * Tr); for (let c = 0; c < C; c++) for (let t = 0; t < Tr; t++) sc[c * Tr + t] = x[c * T + (t >> 1)]; }
    if (this.has(name + '.conv1x1.weight')) sc = this.conv(name + '.conv1x1', sc, Tr).y;
    const y = o.y.map((v, i) => (v + sc[i]) * SQRT1_2);
    return { y, C: C2, T: Tr };
  }

  genResblock(name, x, C, T, style) {
    const dil = [1, 3, 5];
    for (let j = 0; j < 3; j++) {
      const k = this.w(`${name}.convs1.${j}.weight`).dims[0];
      let xt = this.snake(`${name}.alpha1.${j}`, this.adain(`${name}.adain1.${j}`, x, C, T, style), C, T);
      xt = this.conv(`${name}.convs1.${j}`, xt, T, 1, (k * dil[j] - dil[j]) / 2, dil[j]).y;
      xt = this.snake(`${name}.alpha2.${j}`, this.adain(`${name}.adain2.${j}`, xt, C, T, style), C, T);
      xt = this.conv(`${name}.convs2.${j}`, xt, T, 1, (k - 1) / 2).y;
      x = xt.map((v, i) => v + x[i]);
    }
    return x;
  }

  /** Bidirectional LSTM: x [n][nIn] → y [n][2·256]. */
  lstm(name, x, n, nIn) {
    const H = KOKORO.HIDDEN / 2, y = new Float32Array(n * 2 * H);
    for (let dir = 0; dir < 2; dir++) {
      const sfx = dir ? '_reverse' : '';
      const Wi = this.w(`${name}.weight_ih_l0${sfx}`).data, Wh = this.w(`${name}.weight_hh_l0${sfx}`).data, B = this.w(`${name}.bias_l0${sfx}`).data;
      const h = new Float32Array(H), c = new Float32Array(H);
      for (let s = 0; s < n; s++) {
        const t = dir ? n - 1 - s : s;
        const gi = matvec(Wi, x, 4 * H, nIn, B, t * nIn), gh = matvec(Wh, h, 4 * H, H);
        for (let j = 0; j < H; j++) {
          const ig = sigmoid(gi[j] + gh[j]), fg = sigmoid(gi[H + j] + gh[H + j]);
          const gg = Math.tanh(gi[2 * H + j] + gh[2 * H + j]), og = sigmoid(gi[3 * H + j] + gh[3 * H + j]);
          c[j] = fg * c[j] + ig * gg; h[j] = og * Math.tanh(c[j]);
          y[t * 2 * H + dir * H + j] = h[j];
        }
      }
    }
    return y;
  }

  /** ALBERT (PL-BERT) on the padded tokens: [n][768]. */
  bert(tokens, nLayer, nHead, eps) {
    const n = tokens.length, E = this.w('token_embd.weight').dims[0], e = this.w('position_embd.weight').dims[0];
    const tok = this.w('token_embd.weight').data, pos = this.w('position_embd.weight').data;
    const ln = (x, rows, N, g, b, ep) => {
      const y = new Float32Array(rows * N);
      for (let r = 0; r < rows; r++) {
        let m = 0; for (let i = 0; i < N; i++) m += x[r * N + i]; m /= N;
        let v = 0; for (let i = 0; i < N; i++) v += (x[r * N + i] - m) ** 2; v /= N;
        const inv = 1 / Math.sqrt(v + ep);
        for (let i = 0; i < N; i++) y[r * N + i] = (x[r * N + i] - m) * inv * g[i] + b[i];
      }
      return y;
    };
    let x = new Float32Array(n * e);
    for (let t = 0; t < n; t++) for (let i = 0; i < e; i++) x[t * e + i] = tok[tokens[t] * E + i] + pos[t * e + i];
    x = ln(x, n, e, this.w('token_embd_norm.weight').data, this.w('token_embd_norm.bias').data, eps);
    const mm = (name, X, rows, N) => { const W = this.w(name + '.weight').data, B = this.w(name + '.bias').data, Y = new Float32Array(n * rows); for (let t = 0; t < n; t++) Y.set(matvec(W, X, rows, N, B, t * N), t * rows); return Y; };
    x = mm('token_embd_proj', x, E, e);
    const D = E / nHead, F = this.w('blk.0.ffn_up.weight').dims[1];
    for (let l = 0; l < nLayer; l++) {
      const q = mm('blk.0.attn_q', x, E, E), k = mm('blk.0.attn_k', x, E, E), v = mm('blk.0.attn_v', x, E, E), att = new Float32Array(n * E);
      for (let h = 0; h < nHead; h++) for (let i = 0; i < n; i++) {
        const s = new Float32Array(n);
        let m = -Infinity;
        for (let j = 0; j < n; j++) { let a = 0; for (let d = 0; d < D; d++) a += q[i * E + h * D + d] * k[j * E + h * D + d]; s[j] = a / Math.sqrt(D); m = Math.max(m, s[j]); }
        let z = 0; for (let j = 0; j < n; j++) { s[j] = Math.exp(s[j] - m); z += s[j]; }
        for (let d = 0; d < D; d++) { let a = 0; for (let j = 0; j < n; j++) a += s[j] * v[j * E + h * D + d]; att[i * E + h * D + d] = a / z; }
      }
      const o = mm('blk.0.attn_output', att, E, E);
      x = ln(o.map((val, i) => val + x[i]), n, E, this.w('blk.0.attn_output_norm.weight').data, this.w('blk.0.attn_output_norm.bias').data, eps);
      const u = mm('blk.0.ffn_up', x, F, E).map(gelu), dn = mm('blk.0.ffn_down', u, E, F);
      x = ln(dn.map((val, i) => val + x[i]), n, E, this.w('blk.0.layer_output_norm.weight').data, this.w('blk.0.layer_output_norm.bias').data, eps);
    }
    return x;
  }

  /** Text encoder CNN: tokens → [n][512]. */
  textCnn(tokens) {
    const emb = this.w('a.gen.text.embedding.weight'), C = emb.dims[0], n = tokens.length;
    let x = new Float32Array(C * n);
    for (let t = 0; t < n; t++) for (let c = 0; c < C; c++) x[c * n + t] = emb.data[tokens[t] * C + c];
    for (let i = 0; this.has(`a.gen.text.cnn.${i}.0.weight`); i++) {
      const q = `a.gen.text.cnn.${i}`, k = this.w(q + '.0.weight').dims[0];
      const y = this.conv(q + '.0', x, n, 1, (k - 1) / 2).y, g = this.w(q + '.1.gamma').data, b = this.w(q + '.1.beta').data;
      for (let t = 0; t < n; t++) {
        let m = 0; for (let c = 0; c < C; c++) m += y[c * n + t]; m /= C;
        let v = 0; for (let c = 0; c < C; c++) v += (y[c * n + t] - m) ** 2; v /= C;
        const inv = 1 / Math.sqrt(v + 1e-5);
        for (let c = 0; c < C; c++) { const z = (y[c * n + t] - m) * inv * g[c] + b[c]; x[c * n + t] = z > 0 ? z : 0.2 * z; }
      }
    }
    const out = new Float32Array(n * C);
    for (let t = 0; t < n; t++) for (let c = 0; c < C; c++) out[t * C + c] = x[c * n + t];
    return out;
  }

  /**
   * The whole utterance. tokens: phoneme ids padded with 0 at both ends; voiceRow: 256 values (decoder style, then
   * predictor style). Returns { audio, durations, f0 } and the intermediate tensors for comparisons.
   */
  speak(tokens, voiceRow, { speed = 1, rng, nLayer = 12, nHead = 12, eps = 1e-12 } = {}) {
    const n = tokens.length, Hd = KOKORO.HIDDEN, S = KOKORO.STYLE, nCat = Hd + S;
    const styleDec = voiceRow.subarray(0, S), stylePred = voiceRow.subarray(S, 2 * S);
    const bert = this.bert(tokens, nLayer, nHead, eps), E = bert.length / n;
    const text = this.textCnn(tokens);
    // Prosody predictor.
    const enc = this.w('a.gen.bert_enc.weight').data, encB = this.w('a.gen.bert_enc.bias').data;
    let x = new Float32Array(n * nCat);
    for (let t = 0; t < n; t++) { x.set(matvec(enc, bert, Hd, E, encB, t * E), t * nCat); x.set(stylePred, t * nCat + Hd); }
    for (let il = 0; this.has(`a.gen.pred.text_encoder.lstms.${2 * il}.weight_ih_l0`); il++) {
      const y = this.lstm(`a.gen.pred.text_encoder.lstms.${2 * il}`, x, n, nCat);
      const norm = `a.gen.pred.text_encoder.lstms.${2 * il + 1}.fc`;
      const gb = matvec(this.w(norm + '.weight').data, stylePred, 2 * Hd, S, this.w(norm + '.bias').data);
      for (let t = 0; t < n; t++) {
        let m = 0; for (let j = 0; j < Hd; j++) m += y[t * Hd + j]; m /= Hd;
        let v = 0; for (let j = 0; j < Hd; j++) v += (y[t * Hd + j] - m) ** 2; v /= Hd;
        const inv = 1 / Math.sqrt(v + 1e-5);
        for (let j = 0; j < Hd; j++) x[t * nCat + j] = (1 + gb[j]) * (y[t * Hd + j] - m) * inv + gb[Hd + j];
      }
    }
    const yd = this.lstm('a.gen.pred.lstm', x, n, nCat);
    const pw = this.w('a.gen.pred.duration_proj.linear_layer.weight'), nBins = pw.dims[1];
    const logits = new Float32Array(n * nBins);
    for (let t = 0; t < n; t++) logits.set(matvec(pw.data, yd, nBins, Hd, this.w('a.gen.pred.duration_proj.linear_layer.bias').data, t * Hd), t * nBins);
    const { durations, index } = alignIndex(logits, n, nBins, speed);
    const F = index.length;
    const gather = (src, w) => { const o = new Float32Array(F * w); for (let f = 0; f < F; f++) o.set(src.subarray(index[f] * w, index[f] * w + w), f * w); return o; };
    const shared = this.lstm('a.gen.pred.shared', gather(x, nCat), F, nCat);
    const asr = gather(this.lstm('a.gen.text.lstm', text, n, Hd), Hd);
    const tr = (a, rows, cols) => { const o = new Float32Array(a.length); for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) o[c * rows + r] = a[r * cols + c]; return o; };
    // F0 and energy curves.
    const curves = {};
    for (const curve of ['F0', 'N']) {
      let cur = { y: tr(shared, F, Hd), C: Hd, T: F };
      for (let i = 0; i < 3; i++) { const q = `a.gen.pred.${curve}.${i}`; cur = this.resblk(q, cur.y, cur.C, cur.T, stylePred, this.has(q + '.pool.weight')); }
      curves[curve] = this.conv(`a.gen.pred.${curve}_proj`, cur.y, cur.T).y;
    }
    const f0 = curves.F0, en = curves.N;
    // Decoder.
    const p = 'a.gen.dec.';
    const f0d = this.conv(p + 'F0_conv', f0, 2 * F, 2, 1).y, end = this.conv(p + 'N_conv', en, 2 * F, 2, 1).y;
    const asrT = tr(asr, F, Hd);
    let cur = this.resblk(p + 'encode', Float32Array.of(...asrT, ...f0d, ...end), Hd + 2, F, styleDec, false);
    const asrRes = this.conv(p + 'asr_res.0', asrT, F).y;
    for (let i = 0; this.has(`${p}decode.${i}.conv1.weight`); i++) {
      const q = `${p}decode.${i}`, cat = new Float32Array(cur.y.length + asrRes.length + 2 * F);
      cat.set(cur.y); cat.set(asrRes, cur.y.length); cat.set(f0d, cur.y.length + asrRes.length); cat.set(end, cur.y.length + asrRes.length + F);
      cur = this.resblk(q, cat, cat.length / F, F, styleDec, this.has(q + '.pool.weight'));
    }
    const { har, nHar } = harSource(f0, this.w(p + 'generator.m_source.l_linear.weight').data, this.w(p + 'generator.m_source.l_linear.bias').data, rng);
    const harT = tr(har, nHar, 2 * KOKORO.BINS);
    const g = p + 'generator.';
    let X = cur.y, C = cur.C, T = cur.T;
    for (let i = 0; this.has(`${g}ups.${i}.weight`); i++) {
      X = X.map((v) => (v > 0 ? v : 0.1 * v));
      const sw = this.w(`${g}noise_convs.${i}.weight`), sk = sw.dims[0], ss = sk > 1 ? sk / 2 : 1;
      const src = this.conv(`${g}noise_convs.${i}`, harT, nHar, ss, sk > 1 ? (ss + 1) >> 1 : 0);
      const srcR = this.genResblock(`${g}noise_res.${i}`, src.y, src.y.length / src.L, src.L, styleDec);
      const uw = this.w(`${g}ups.${i}.weight`), [K, OC] = uw.dims, stride = K / 2, trim = stride / 2, ub = this.w(`${g}ups.${i}.bias`).data;
      const L = T * stride, last = !this.has(`${g}ups.${i + 1}.weight`), Lo = L + (last ? 1 : 0), y = new Float32Array(OC * Lo);
      for (let o = 0; o < OC; o++) for (let t = 0; t < L; t++) {
        let a = ub[o];
        const tt = t + trim;
        for (let c = 0; c < C; c++) for (let k = tt % stride; k < K; k += stride) { const ii = (tt - k) / stride; if (ii >= 0 && ii < T) a += uw.data[(c * OC + o) * K + k] * X[c * T + ii]; }
        y[o * Lo + t + (last ? 1 : 0)] = a;
      }
      if (last) for (let o = 0; o < OC; o++) y[o * Lo] = y[o * Lo + 2];
      for (let j = 0; j < y.length; j++) y[j] += srcR[j];
      let acc = null;
      for (let j = 0; j < 3; j++) { const r = this.genResblock(`${g}resblocks.${i * 3 + j}`, y, OC, Lo, styleDec); acc = acc ? acc.map((v, q) => v + r[q]) : r; }
      X = acc.map((v) => v / 3); C = OC; T = Lo;
    }
    const spec = this.conv(g + 'conv_post', X.map((v) => (v > 0 ? v : 0.01 * v)), T, 1, 3).y;
    const rows = tr(spec, 2 * KOKORO.BINS, T);
    return { audio: istft(rows, T), durations, f0, en, spec: rows, bert, text, har };
  }
}
