// Tilde public API.
//
//   const llm = await Tilde.load('https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/main/LFM3-1.7B-QAD-Q4_0.gguf');
//   const text = await llm.chat([{ role: 'user', content: 'hi' }], { onToken: (t) => print(t) });

import { openGpu } from './gpu.js';
import { httpSource } from './gguf.js';
import { Tokenizer } from './tokenizer.js';
import { cachedSource, clearCache, cacheFile, cachedFileInfo, removeCachedFile } from './cache.js';
import { readGGUF } from './gguf.js';
import { patchify } from './image.js';
import { readVideo, groups as videoGroups, frameTimes } from './video.js';
import { logMel, decodeAudio, parseWav } from './audio.js';
import { Istft, sampleCode, wav, SPEECH_RATE } from './speech.js';

// Model code loads on demand: only the file for the GGUF's architecture, and each sense's encoder when it is
// first used, so adding architectures costs nothing to pages that don't run them.
const ARCHS = {
  lfm3: () => import('./models/lfm3.js').then((m) => m.LFM3),
  qwen3: () => import('./models/qwen3.js').then((m) => m.Qwen3),
};
const vision = () => import('./models/lfm3vl.js').then((m) => m.LFM3Vision);
const audio = () => import('./models/lfm3a.js').then((m) => m.LFM3Audio);
const speechGen = () => import('./models/lfm3gen.js').then((m) => m.LFM3SpeechGen);

/** Architectures Tilde can run (general.architecture in the GGUF). */
export const architectures = Object.keys(ARCHS);

export { openGpu, httpSource, Tokenizer, cachedSource, clearCache, cacheFile, cachedFileInfo, removeCachedFile, patchify, logMel, decodeAudio, parseWav, wav, SPEECH_RATE, readVideo, frameTimes };

const roundEven = (x) => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 ? r - 1 : r; };

export class Tilde {
  /**
   * Loads a GGUF model onto the GPU, streaming it in.
   * @param {string | import('./gguf.js').ByteSource} source URL or byte source
   * @param {{ ctx?: number, chunk?: number, maxChunk?: number, submitBudgetMs?: number, cache?: boolean,
   *   gpu?: import('./gpu.js').Gpu, onProgress?: Function, mmproj?: string, imageMaxPixels?: number,
   *   keepEncoders?: boolean }} [opts]
   *   `chunk` is the prompt tokens per GPU pass to start with; it adapts down when a pass would take longer than
   *   `submitBudgetMs` (default 1000; WebKit kills the GPU at ~3 s) and back up to `maxChunk` (default `chunk`)
   *   when there's room. A larger `maxChunk` reads long prompts faster but sizes activations for it (more memory).
   *   `cache` (default true for URLs) keeps files in the browser's Cache Storage so they download once.
   *   `mmproj` (Omni/VL models) is the encoder file; encoders load only when an image arrives and are freed after
   *   unless `keepEncoders`. `imageMaxPixels` caps image size (default 512·512, at most 256 image tokens).
   */
  static async load(source, { ctx = 4096, chunk = 64, maxChunk = chunk, submitBudgetMs = 1000, cache = true, gpu, onProgress, mmproj, imageMaxPixels = 512 * 512, keepEncoders = false } = {}) {
    gpu ||= await openGpu();
    const open = async (s) => (typeof s === 'string' && cache ? cachedSource(httpSource(s), { key: s }) : typeof s === 'string' ? httpSource(s) : s);
    const src = await open(source);
    const gguf = await readGGUF(src);
    const arch = gguf.kv['general.architecture'];
    if (!ARCHS[arch]) throw new Error(`Tilde doesn't run "${arch}" models yet (it runs: ${architectures.join(', ')})`);
    const model = await (await ARCHS[arch]()).load(gpu, src, { ctx, chunk, maxChunk, submitBudgetMs, onProgress, gguf });
    const k = new Tilde(gpu, model);
    if (mmproj) {
      const src = await open(mmproj);
      k.encoders = { src, gguf: await readGGUF(src) };
    }
    k.imageMaxPixels = imageMaxPixels;
    k.keepEncoders = keepEncoders;
    return k;
  }

  constructor(gpu, model) {
    this.gpu = gpu;
    this.model = model;
    this.tokenizer = Tokenizer.fromGGUF(model.gguf.kv);
    const s = model.gguf.kv;
    this.defaults = {
      temperature: s['general.sampling.temp'] ?? 0.1,
      topK: s['general.sampling.top_k'] ?? 50,
      repeatPenalty: s['general.sampling.penalty_repeat'] ?? 1.1,
    };
    /** Tokens the model has already read, in order. */
    this.history = [];
  }

  /**
   * Formats messages with the model's chat format (system/user/assistant; tools are left to the caller).
   * Content is a string, or an array of parts: strings, { type: 'text', text }, { type: 'image', image },
   * { type: 'audio', audio } (16 kHz mono Float32Array; see decodeAudio), { type: 'video', video } (from readVideo).
   * Returns a string, or (with media) an array of strings and { image } / { audio } / { video } parts.
   * @param {{ role: string, content: string | any[] }[]} messages
   */
  formatChat(messages, { thinking = true } = {}) {
    const parts = [this.tokenizer.bos === undefined ? '' : this.tokenizer.tokens[this.tokenizer.bos]];
    const push = (x) => { if (typeof x === 'string' && typeof parts.at(-1) === 'string') parts[parts.length - 1] += x; else parts.push(x); };
    const content = (c) => {
      if (!Array.isArray(c)) return push(String(c ?? ''));
      for (const p of c) {
        if (typeof p === 'string') push(p);
        else if (p.type === 'image') { push('<|vision_start|>'); push({ image: p.image }); push('<|vision_end|>'); }
        else if (p.type === 'audio') { push('<|vision_start|>'); push({ audio: p.audio }); push('<|vision_end|>'); }
        else if (p.type === 'video') { push('<|vision_start|>'); push({ video: p.video }); push('<|vision_end|>'); }
        else push(String(p.text ?? ''));
      }
    };
    let lastUser = -1;
    messages.forEach((m, i) => { if (m.role === 'user') lastUser = i; });
    messages.forEach((m, i) => {
      push(`<|im_start|>${m.role}\n`);
      if (m.role === 'assistant' && i < lastUser) push('<think></think>' + String(m.content).split('</think>').pop().trim());
      else content(m.content);
      push('<|im_end|>\n');
    });
    push('<|im_start|>assistant\n' + (thinking ? '<think>' : '<think></think>'));
    return parts.length === 1 ? parts[0] : parts;
  }

  /** Chat completion: formats, generates, returns the reply text. */
  chat(messages, opts = {}) {
    return this.generate(this.formatChat(messages, opts), opts);
  }

  /**
   * Generates text after `prompt`: a string, token ids, or an array of strings and { image } parts.
   * If the prompt starts with everything the model has already read (same tokens, same media objects), only the
   * new part is read. `snapshotAt` (a token count into the prompt) remembers the model's state at that point:
   * a later prompt that starts with those same tokens goes back there and reads only the rest, even after a reply
   * whose words were tokenized differently or a different continuation (e.g. a chat template that rewrites the end).
   * Otherwise the model starts over.
   * @param {string | number[] | (string | { image: any })[]} prompt
   * @param {{ maxTokens?: number, temperature?: number, topK?: number, topP?: number, repeatPenalty?: number,
   *   stop?: string[], snapshotAt?: number, onToken?: (text: string, id: number) => void, onStatus?: (s: string) => void,
   *   onRead?: (r: { reused: number, read: number }) => void, signal?: AbortSignal }} [opts]
   */
  async generate(prompt, opts = {}) {
    const o = { maxTokens: 512, topP: 1, stop: [], ...this.defaults, ...opts };
    const items = this.items(prompt);
    const m = this.model;
    const h = this.history;
    const prefix = (a) => a && a.length < items.length && a.every((t, i) => t === items[i]);
    if (prefix(h) && (h.length || m.pos === 0)) {
      // carry on from what was read
    } else if (this.snap && prefix(this.snap)) {
      await m.restore();
      this.history = this.snap.slice();
    } else {
      m.reset(); this.history = []; this.snap = null;
    }
    const start = this.history.length;
    const fresh = items.slice(start);
    this.history = []; // until the whole prompt is read: a read cut short leaves the model somewhere in between
    const mark = o.snapshotAt > start && o.snapshotAt < items.length ? o.snapshotAt - start : -1;
    for (let i = 0; i < fresh.length;) {
      if (o.signal?.aborted) throw new DOMException('Aborted', 'AbortError');
      if (i === mark) { await m.snapshot(); this.snap = items.slice(0, start + i); }
      if (typeof fresh[i] !== 'number') {
        if (fresh[i].audio) await this.#readAudio(fresh[i].audio, o);
        else if (fresh[i].video) await this.#readVideo(fresh[i].video, o);
        else await this.#readImage(fresh[i].image, o);
        i++;
        continue;
      }
      let j = i;
      const end = mark > i ? Math.min(mark, fresh.length) : fresh.length;
      while (j < end && j - i < m.chunk && typeof fresh[j] === 'number') j++;
      await m.forward(fresh.slice(i, j));
      i = j;
    }
    o.onRead?.({ reused: start, read: fresh.length });
    this.history = [...items];

    const next = this.tokenizer.decoder();
    let text = '';
    for (let n = 0; n < o.maxTokens; n++) {
      if (o.signal?.aborted) break;
      const id = sampleCandidates(await m.candidates(), this.history.slice(-64).filter((t) => typeof t === 'number'), o);
      if (id === this.tokenizer.eos || this.tokenizer.tokens[id] === '<|im_end|>') break;
      const piece = next(id);
      text += piece;
      o.onToken?.(piece, id);
      if (o.stop.some((s) => text.endsWith(s))) break;
      if (m.pos >= m.ctx) break;
      await m.forward([id]);
      this.history.push(id);
    }
    return text;
  }

  /** A prompt as items: token ids, and media parts (one item each, compared by identity). */
  items(prompt) {
    return typeof prompt === 'string' ? this.tokenizer.encode(prompt)
      : prompt.flatMap((p) => (typeof p === 'number' ? [p] : typeof p === 'string' ? this.tokenizer.encode(p) : [p]));
  }

  /** Encodes 16 kHz mono audio (loading the audio encoder if needed) and feeds its embeddings to the model. */
  async #readAudio(pcm, o) {
    if (!this.encoders) throw new Error('This model was loaded without an mmproj, so it cannot hear audio.');
    if (!this.audio) {
      o.onStatus?.('loading audio encoder');
      this.audio = await (await audio()).load(this.gpu, this.encoders.src, this.encoders.gguf);
    }
    o.onStatus?.('listening');
    const { buf, tokens } = await this.audio.encode(logMel(pcm));
    if (!this.keepEncoders) { this.audio.free(); this.audio = null; }
    try {
      for (let i = 0; i < tokens; i += this.model.chunk) await this.model.forwardEmbeds(buf, i, Math.min(this.model.chunk, tokens - i));
    } finally { this.gpu.free(buf); }
    o.onStatus?.(`heard audio (${tokens} tokens, ${(pcm.length / 16000).toFixed(1)} s)`);
  }

  /**
   * Reads a video from readVideo: for each pair of frames its timestamp, the pair (one temporal patch) and, for Omni,
   * the pair's slice of the soundtrack. The soundtrack is encoded once as one clip and each pair takes its rows
   * (llama.cpp MTMD_VIDEO_TEMPLATE_LFM3_OMNI). The vision encoder stays loaded for the whole clip.
   */
  async #readVideo(video, o) {
    if (!this.encoders) throw new Error('This model was loaded without an mmproj, so it cannot watch video.');
    const m = this.model;
    const hasAudioEnc = this.encoders.gguf.tensors.some((t) => t.name.startsWith('a.blk.'));
    const pcm = hasAudioEnc ? video.audio : null;
    const gs = videoGroups(video.times, pcm?.length || 0);
    let aud = null;
    try {
      if (pcm) {
        if (!this.audio) { o.onStatus?.('loading audio encoder'); this.audio = await (await audio()).load(this.gpu, this.encoders.src, this.encoders.gguf); }
        o.onStatus?.('listening to the soundtrack');
        aud = await this.audio.encode(logMel(pcm));
        if (!this.keepEncoders) { this.audio.free(); this.audio = null; }
      }
      if (!this.vision) { o.onStatus?.('loading vision encoder'); this.vision = await (await vision()).load(this.gpu, this.encoders.src, this.encoders.gguf); }
      const f0 = video.frames[0];
      const cfg = { ...this.vision.cfg, minPixels: 0, maxPixels: f0.width * f0.height };
      let samples = 0, row = 0, tokens = 0;
      for (let g = 0; g < gs.length; g++) {
        if (o.signal?.aborted) throw new DOMException('Aborted', 'AbortError');
        const grp = gs[g];
        o.onStatus?.(`watching ${g + 1} of ${gs.length}`);
        // <t seconds><|vision_start|> pair <|vision_end|>, then the pair's sound (llama.cpp wraps each pair like an image).
        const ids = this.tokenizer.encode(grp.text + '<|vision_start|>'), close = this.tokenizer.encode('<|vision_end|>');
        for (let i = 0; i < ids.length; i += m.chunk) await m.forward(ids.slice(i, i + m.chunk));
        const a = patchify(video.frames[grp.frames[0]], cfg), b = patchify(video.frames[grp.frames[1]], cfg);
        const { buf, tokens: n } = await this.vision.encode({ ...a, patches2: b.patches });
        try {
          for (let i = 0; i < n; i += m.chunk) await m.forwardEmbeds(buf, i, Math.min(m.chunk, n - i));
        } finally { this.gpu.free(buf); }
        await m.forward(close);
        tokens += ids.length + n + close.length;
        if (aud) {
          samples += grp.audio[1] - grp.audio[0];
          const end = g === gs.length - 1 ? aud.tokens : Math.min(aud.tokens, Math.max(row, roundEven(samples / 1280)));
          for (let i = row; i < end; i += m.chunk) await m.forwardEmbeds(aud.buf, i, Math.min(m.chunk, end - i));
          tokens += end - row;
          row = end;
        }
      }
      o.onStatus?.(`watched video (${video.frames.length} frames${aud ? ', with sound' : ''}, ${tokens} tokens)`);
    } finally {
      if (aud) this.gpu.free(aud.buf);
      if (!this.keepEncoders && this.vision) { this.vision.free(); this.vision = null; }
    }
  }

  /** Encodes an image (loading the vision encoder if needed) and feeds its embeddings to the model. */
  async #readImage(image, o) {
    if (!this.encoders) throw new Error('This model was loaded without an mmproj, so it cannot read images.');
    if (!this.vision) {
      o.onStatus?.('loading vision encoder');
      this.vision = await (await vision()).load(this.gpu, this.encoders.src, this.encoders.gguf);
    }
    o.onStatus?.('reading image');
    const pt = patchify(image, { ...this.vision.cfg, maxPixels: Math.min(this.vision.cfg.maxPixels, this.imageMaxPixels) });
    const { buf, tokens } = await this.vision.encode(pt);
    if (!this.keepEncoders) { this.vision.free(); this.vision = null; }
    try {
      for (let i = 0; i < tokens; i += this.model.chunk) await this.model.forwardEmbeds(buf, i, Math.min(this.model.chunk, tokens - i));
    } finally { this.gpu.free(buf); }
    o.onStatus?.(`read image (${tokens} tokens)`);
  }

  /**
   * Text to speech with the model's own voice (Omni mmproj). Streams PCM chunks (24 kHz mono Float32Array) to
   * `onAudio` as frames are made, and resolves to the whole clip.
   * @param {string} text
   * @param {{ voice?: 'female' | 'male', language?: string, maxSeconds?: number, temperature?: number,
   *   onAudio?: (pcm: Float32Array) => void, onStatus?: (s: string) => void, signal?: AbortSignal }} [opts]
   */
  async speak(text, { voice = 'female', language = 'American English', maxSeconds = 30, temperature, onAudio, onStatus, signal } = {}) {
    if (!this.encoders) throw new Error('This model was loaded without an mmproj, so it cannot speak.');
    if (!this.speechGen) {
      onStatus?.('loading voice');
      this.speechGen = await (await speechGen()).load(this.gpu, this.encoders.src, this.encoders.gguf);
    }
    const g = this.speechGen, m = this.model, V = g.nVocab, n = g.nCb;
    const cfg = { ...g.sampling, temp: temperature ?? g.sampling.temp };
    const prompt = this.tokenizer.tokens[this.tokenizer.bos] +
      `<|im_start|>system\nPerform TTS in ${language} using a ${voice} voice.<|im_end|>\n` +
      `<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n<think></think><|audio_start|>`;
    const ids = this.tokenizer.encode(prompt);
    // Speech runs in a sequence of its own, so the conversation's cache is still there afterwards.
    const main = m.useSequence(this.#side());
    m.reset();
    const chunks = [];
    try {
    for (let i = 0; i < ids.length; i += m.chunk) await m.forward(ids.slice(i, i + m.chunk));
    g.resetCodec();
    const ist = new Istft();
    const emit = (pcm) => { if (pcm.length) { chunks.push(pcm); onAudio?.(pcm); } };
    const recent = Array.from({ length: n }, () => []);
    onStatus?.('speaking');
    const maxFrames = Math.ceil(maxSeconds * 12.5);
    for (let f = 0; f < maxFrames && m.pos < m.ctx; f++) {
      if (signal?.aborted) break;
      // MaskGIT: reveal the 12 codebooks in a random order, 4 per depthformer pass.
      const order = [...Array(n).keys()].sort(() => Math.random() - 0.5);
      const codes = new Array(n).fill(-1);
      for (let pass = 0, k = 0; pass < 3; pass++) {
        const l = await g.codeLogits(m.a.headIn, codes);
        for (let j = 0; j < 4; j++, k++) {
          const i = order[k];
          codes[i] = sampleCode(l.subarray(i * V, (i + 1) * V), recent[i].slice(-cfg.lastN), cfg);
        }
      }
      if (codes.includes(V - 1)) break; // end of audio
      codes.forEach((c, i) => recent[i].push(c));
      await m.forwardEmbeds(await g.feedback(codes), 0, 1);
      const st = await g.decodeFrame(codes);
      for (let s = 0; s < 6; s++) emit(ist.frame(st, s * 1282));
    }
    emit(ist.flush());
    } finally { this.sideSeq.pos = m.pos; m.useSequence(main); }
    if (!this.keepEncoders) { g.free(); this.speechGen = null; }
    const out = new Float32Array(chunks.reduce((s, c) => s + c.length, 0));
    let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; }
    onStatus?.(`spoke ${(out.length / SPEECH_RATE).toFixed(1)} s`);
    return out;
  }

  /** The side sequence for speech and transcription (2,048 positions, made on first use). */
  #side() {
    this.sideSeq ||= this.model.newSequence(Math.min(2048, this.model.ctx));
    return this.sideSeq;
  }

  /**
   * Speech to text with the model's own hearing (Omni): what is said in 16 kHz mono `pcm`, as text. Runs in the
   * side sequence, so the conversation's cache is kept.
   * @param {Float32Array} pcm
   * @param {{ maxTokens?: number, onToken?: (text: string) => void, onStatus?: (s: string) => void, signal?: AbortSignal }} [opts]
   */
  async transcribe(pcm, { maxTokens = 256, onToken, onStatus, signal } = {}) {
    const m = this.model;
    const main = m.useSequence(this.#side());
    const keep = { history: this.history, snap: this.snap };
    this.history = []; this.snap = null;
    m.reset();
    try {
      const prompt = this.formatChat([{ role: 'user', content: [{ type: 'audio', audio: pcm }, 'Transcribe the audio exactly, word for word. Write only the words that are said.'] }], { thinking: false });
      const text = await this.generate(prompt, { temperature: 0, repeatPenalty: 1, maxTokens, onToken, onStatus, signal });
      return text.trim();
    } finally {
      this.sideSeq.pos = m.pos;
      m.useSequence(main);
      this.history = keep.history; this.snap = keep.snap;
    }
  }

  /** Memory and timing report. */
  stats() { return this.gpu.report(); }
}

/**
 * Samples from GPU top candidates ({ values, ids }): repeat penalty, temperature, top-k, top-p; temperature 0 = greedy.
 * The penalty only lowers scores, so the penalized top-k is still among the candidates.
 */
export function sampleCandidates({ values, ids }, recent, o) {
  const rec = new Set(recent);
  const pen = o.repeatPenalty && o.repeatPenalty !== 1 ? o.repeatPenalty : 0;
  const order = [...values.keys()].filter((i) => values[i] > -1e38);
  const score = (i) => (pen && rec.has(ids[i]) ? (values[i] > 0 ? values[i] / pen : values[i] * pen) : values[i]);
  const sc = new Float32Array(values.length);
  for (const i of order) sc[i] = score(i);
  order.sort((a, b) => sc[b] - sc[a]);
  if (!o.temperature) return ids[order[0]];
  const k = Math.min(o.topK || order.length, order.length);
  const top = order.slice(0, k);
  const mx = sc[top[0]];
  let p = top.map((i) => Math.exp((sc[i] - mx) / o.temperature));
  const sum = p.reduce((a, b) => a + b, 0);
  p = p.map((x) => x / sum);
  let cut = top.length, acc = 0;
  if (o.topP < 1) for (let i = 0; i < p.length; i++) { acc += p[i]; if (acc >= o.topP) { cut = i + 1; break; } }
  let r = Math.random() * p.slice(0, cut).reduce((a, b) => a + b, 0);
  for (let i = 0; i < cut; i++) { r -= p[i]; if (r <= 0) return ids[top[i]]; }
  return ids[top[cut - 1]];
}

/** Repeat penalty, temperature, top-k, top-p over full logits. temperature 0 = greedy. */
export function sample(logits, recent, { temperature, topK, topP, repeatPenalty }) {
  const l = logits;
  if (repeatPenalty && repeatPenalty !== 1) {
    for (const t of new Set(recent)) l[t] = l[t] > 0 ? l[t] / repeatPenalty : l[t] * repeatPenalty;
  }
  if (!temperature) { let b = 0; for (let i = 1; i < l.length; i++) if (l[i] > l[b]) b = i; return b; }
  const k = Math.min(topK || l.length, l.length);
  // Partial selection of the top k.
  const idx = [];
  for (let i = 0; i < l.length; i++) {
    if (idx.length < k) { idx.push(i); if (idx.length === k) idx.sort((a, b) => l[b] - l[a]); continue; }
    if (l[i] > l[idx[k - 1]]) { let j = k - 1; while (j > 0 && l[idx[j - 1]] < l[i]) { idx[j] = idx[j - 1]; j--; } idx[j] = i; }
  }
  if (idx.length < k) idx.sort((a, b) => l[b] - l[a]);
  const mx = l[idx[0]];
  let p = idx.map((i) => Math.exp((l[i] - mx) / temperature));
  const sum = p.reduce((a, b) => a + b, 0);
  p = p.map((x) => x / sum);
  let cut = idx.length, acc = 0;
  if (topP < 1) for (let i = 0; i < p.length; i++) { acc += p[i]; if (acc >= topP) { cut = i + 1; break; } }
  let r = Math.random() * p.slice(0, cut).reduce((a, b) => a + b, 0);
  for (let i = 0; i < cut; i++) { r -= p[i]; if (r <= 0) return idx[i]; }
  return idx[cut - 1];
}
