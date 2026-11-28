// Byte-level BPE (GPT-2 style) read from GGUF metadata. Pre-tokenizers, as llama.cpp maps them:
// "lfm2" (and "llama-bpe"): the Llama 3 split with ignore_merges (a word that is already a token is kept whole);
// "qwen2": the same split but digits one at a time, and every word goes through the merges.

const SPLIT = /(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;
const SPLIT_QWEN2 = /(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

// GPT-2's reversible byte → printable character map.
const BYTE2CHAR = (() => {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
  const m = new Array(256);
  bs.forEach((b, i) => { m[b] = String.fromCodePoint(cs[i]); });
  return m;
})();
const CHAR2BYTE = new Map(BYTE2CHAR.map((c, b) => [c, b]));
const ENC = new TextEncoder();

export class Tokenizer {
  /** @param {{ tokens: string[], merges: string[], types?: number[], bos?: number, eos?: number, pre?: string }} meta */
  constructor({ tokens, merges, types, bos, eos, pre }) {
    this.tokens = tokens;
    this.split = pre === 'qwen2' ? SPLIT_QWEN2 : SPLIT;
    this.ignoreMerges = pre !== 'qwen2';
    this.bos = bos;
    this.eos = eos;
    this.ids = new Map(tokens.map((t, i) => [t, i]));
    this.ranks = new Map(merges.map((m, i) => [m, i]));
    // Control (3) and user-defined (4) tokens are matched literally in text, longest first.
    this.special = types
      ? tokens.map((t, i) => [t, i]).filter(([t, i]) => (types[i] === 3 || types[i] === 4) && t.length).sort((a, b) => b[0].length - a[0].length)
      : [];
    this.specialIds = new Set(this.special.map(([, i]) => i));
    this.cache = new Map();
  }

  static fromGGUF(kv) {
    return new Tokenizer({
      tokens: kv['tokenizer.ggml.tokens'], merges: kv['tokenizer.ggml.merges'], types: kv['tokenizer.ggml.token_type'],
      bos: kv['tokenizer.ggml.bos_token_id'], eos: kv['tokenizer.ggml.eos_token_id'], pre: kv['tokenizer.ggml.pre'],
    });
  }

  /** Text → token ids. With `special` (default), special-token strings in the text become their tokens. */
  encode(text, { special = true } = {}) {
    const out = [];
    const parts = special ? this.#splitSpecial(text) : [text];
    for (const part of parts) {
      if (typeof part === 'number') { out.push(part); continue; }
      for (const m of part.matchAll(this.split)) this.#word(m[0], out);
    }
    return out;
  }

  /** Token ids → text. */
  decode(ids, { special = true } = {}) {
    return new TextDecoder().decode(this.#bytes(ids, special));
  }

  /**
   * For streaming: returns a function that takes one token id at a time and gives back the text it adds.
   * A character whose bytes are split across tokens (an emoji often is) comes out whole with the token that
   * finishes it, instead of as U+FFFD and a stray half.
   */
  decoder({ special = true } = {}) {
    const dec = new TextDecoder();
    return (id) => dec.decode(this.#bytes([id], special), { stream: true });
  }

  #bytes(ids, special) {
    const bytes = [];
    for (const id of ids) {
      const t = this.tokens[id];
      if (t === undefined) continue;
      if (this.specialIds.has(id)) { if (special) bytes.push(...ENC.encode(t)); continue; }
      for (const ch of t) { const b = CHAR2BYTE.get(ch); if (b !== undefined) bytes.push(b); else bytes.push(...ENC.encode(ch)); }
    }
    return new Uint8Array(bytes);
  }

  #splitSpecial(text) {
    if (!this.special.length) return [text];
    const out = [];
    let i = 0, start = 0;
    while (i < text.length) {
      if (text[i] === '<' || text[i] === '[') {
        const hit = this.special.find(([t]) => text.startsWith(t, i));
        if (hit) { if (i > start) out.push(text.slice(start, i)); out.push(hit[1]); i += hit[0].length; start = i; continue; }
      }
      i++;
    }
    if (start < text.length) out.push(text.slice(start));
    return out;
  }

  #word(word, out) {
    let ids = this.cache.get(word);
    if (!ids) {
      const chars = [...ENC.encode(word)].map((b) => BYTE2CHAR[b]);
      const whole = this.ignoreMerges ? this.ids.get(chars.join('')) : undefined;
      ids = whole !== undefined ? [whole] : this.#bpe(chars);
      if (this.cache.size < 50000) this.cache.set(word, ids);
    }
    out.push(...ids);
  }

  #bpe(parts) {
    while (parts.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + ' ' + parts[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      parts.splice(best, 2, parts[best] + parts[best + 1]);
    }
    return parts.map((p) => {
      const id = this.ids.get(p);
      if (id === undefined) throw new Error(`no token for "${p}"`);
      return id;
    });
  }
}
