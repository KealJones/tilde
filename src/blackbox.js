// Crash black box: learn from deaths instead of probing for them.
//
// iOS Safari gives no warning before it kills a tab for memory: committed GPU memory past the tab's ~1.5–2 GB kills
// it in well under a second, with no device.lost and no error event, and probing for the limit either shows a
// fake number (allocations are lazy until written) or kills the tab. The limit also moves with what else is running.
// So Tilde writes a tiny record of what it is doing to localStorage as it goes (a synchronous write survives an iOS
// out-of-memory kill), and on the next load reads how the last session ended and backs off if it died there.
// Findings this relies on: nearform/crashbox docs/research 01–03 and 06 (iPhone 15 Pro, iOS 18.7).
//
//   - a clean exit is only `pagehide` with persisted false (app switches and BFCache parks are not exits)
//   - `document.wasDiscarded` is set by an iOS discard, never by a crash
//   - a crash fires nothing; Safari reloads within a second, so heartbeat age tells nothing either
//
//   - a tab hidden (app switch) or just shown again can be killed for other apps' memory; that's not Tilde's doing
//
// How a session ended decides the next one's settings:
//   killed   (no clean exit, not discarded, visible): memory. If it died after everything was allocated, the next
//            session is sized to fit under 85% of the MB it died at, from the memory split (weights, KV cache,
//            activations) recorded when the model was ready: the larger of the KV cache (halve ctx) and the
//            activations (halve maxChunk) shrinks first. If it died earlier (while loading), each level halves
//            maxChunk (twice at most), then ctx.
//   gpu-lost (device lost while visible): the GPU watchdog. Each level takes 30% off the submit budget.
//   background (died or lost the GPU while hidden, or within 5 s of being shown again): no change.
// Three good sessions in a row give one step back (the GPU level first, then the memory cap +10%, then a level).

const PREFIX = 'tilde:blackbox:';
const MAX_LEVEL = 3;
const CLEAN_TO_RELAX = 3;
const RESUME_MS = 5000;
const CAP = 0.85;
const MIN_CTX = 1024, MIN_CHUNK = 16;
const ALLOCATED = new Set(['ready', 'gpu', 'idle']);

/**
 * How the previous session ended.
 * @param {object | undefined} prev the previous session record
 * @param {boolean} wasDiscarded document.wasDiscarded on this load
 * @returns {'none' | 'clean' | 'discarded' | 'background' | 'gpu-lost' | 'killed'}
 */
export function classify(prev, wasDiscarded = false) {
  if (!prev) return 'none';
  const background = prev.hidden || (prev.shownAt != null && (prev.seen ?? 0) - prev.shownAt < RESUME_MS);
  if (prev.lost) return background ? 'background' : 'gpu-lost';
  if (prev.closed) return 'clean';
  if (wasDiscarded) return 'discarded';
  return background ? 'background' : 'killed';
}

/**
 * Next levels from the stored ones and how the last session ended.
 * @param {{ mem?: number, gpu?: number, good?: number, capMB?: number }} levels
 * @param {ReturnType<typeof classify>} verdict
 * @param {object | undefined} prev
 */
export function nextLevels({ mem = 0, gpu = 0, good = 0, capMB } = {}, verdict, prev) {
  const out = (o) => (o.capMB == null ? { mem: o.mem, gpu: o.gpu, good: o.good } : o);
  if (verdict === 'killed') {
    // Died with everything allocated: its MB is the whole footprint, so fit under it instead of guessing.
    if (ALLOCATED.has(prev?.stage) && prev.mb) return out({ mem, gpu, good: 0, capMB: Math.round(Math.min(capMB ?? Infinity, prev.mb * CAP)) });
    return out({ mem: Math.min(MAX_LEVEL, mem + 1), gpu, good: 0, capMB });
  }
  if (verdict === 'gpu-lost') return out({ mem, gpu: Math.min(MAX_LEVEL, gpu + 1), good: 0, capMB });
  // A session counts as good only if the model got as far as running.
  if ((verdict === 'clean' || verdict === 'discarded') && prev?.ran) {
    good++;
    if (good >= CLEAN_TO_RELAX && (mem || gpu || capMB != null)) {
      if (gpu) return out({ mem, gpu: gpu - 1, good: 0, capMB });
      if (capMB != null) return out({ mem, gpu, good: 0, capMB: Math.round(capMB * 1.1) });
      return out({ mem: mem - 1, gpu, good: 0 });
    }
  }
  return out({ mem, gpu, good, capMB });
}

/**
 * Halves ctx or maxChunk, whichever part of memory is larger, until the estimate fits under capMB.
 * @param {{ ctx: number, maxChunk: number }} s
 * @param {{ fixedMB: number, kvMB: number, actMB: number, ctx: number, maxChunk: number }} split memory when ready
 * @param {number} capMB
 */
export function fitMemory(s, split, capMB) {
  let { ctx, maxChunk } = s;
  const est = (c, m) => split.fixedMB + split.kvMB * c / split.ctx + split.actMB * m / split.maxChunk;
  while (est(ctx, maxChunk) > capMB) {
    const kv = split.kvMB * ctx / split.ctx, act = split.actMB * maxChunk / split.maxChunk;
    const canCtx = ctx / 2 >= MIN_CTX, canChunk = maxChunk / 2 >= MIN_CHUNK;
    if (canCtx && (kv >= act || !canChunk)) ctx /= 2;
    else if (canChunk) maxChunk /= 2;
    else break;
  }
  return { ctx, maxChunk };
}

/**
 * Settings after backing off by the given levels.
 * @param {{ ctx: number, chunk: number, maxChunk: number, submitBudgetMs: number }} want
 * @param {{ mem: number, gpu: number }} levels
 */
export function backOff(want, { mem = 0, gpu = 0, capMB } = {}, split) {
  let maxChunk = Math.max(Math.min(MIN_CHUNK, want.maxChunk), want.maxChunk >> Math.min(mem, 2));
  let ctx = Math.max(Math.min(MIN_CTX, want.ctx), want.ctx >> Math.max(0, mem - 1));
  if (capMB != null && split) ({ ctx, maxChunk } = fitMemory({ ctx, maxChunk }, split, capMB));
  const submitBudgetMs = Math.max(Math.min(300, want.submitBudgetMs), Math.round(want.submitBudgetMs * 0.7 ** gpu));
  return { ctx, chunk: Math.min(want.chunk, maxChunk), maxChunk, submitBudgetMs };
}

export class BlackBox {
  /**
   * Reads how the last session for `name` ended, works out the settings to use, and starts a new record.
   * @param {string} name model identity (one record per model)
   * @param {{ ctx: number, chunk: number, maxChunk: number, submitBudgetMs: number }} want requested settings
   * @param {{ storage?: Storage, apply?: boolean, wasDiscarded?: boolean, target?: EventTarget }} [o]
   *   `apply: false` records and reports but keeps the requested settings (benchmarks).
   */
  constructor(name, want, { storage = globalThis.localStorage, apply = true, wasDiscarded = globalThis.document?.wasDiscarded ?? false, target = globalThis } = {}) {
    this.storage = storage;
    this.key = PREFIX + name;
    let stored = {};
    try { stored = JSON.parse(storage?.getItem(this.key) || '{}') || {}; } catch { stored = {}; }
    const prev = stored.session;
    /** How the previous session ended. */
    this.verdict = classify(prev, wasDiscarded);
    /** The previous session's last record (stage, MB, settings), if any. */
    this.previous = prev;
    this.levels = nextLevels(stored.levels, this.verdict, prev);
    /** Memory split recorded when the model was last ready (see split()); kept across sessions. */
    this.memSplit = stored.split;
    /** Settings to use. */
    this.settings = apply ? backOff(want, this.levels, this.memSplit) : { ...want };
    this.session = { t: Date.now(), stage: 'loading', ...this.settings };
    this.#write();
    this.last = 0;
    if (target?.addEventListener) {
      // Only a non-persisted pagehide is an exit; a BFCache park comes back with pageshow and runs on.
      target.addEventListener('pagehide', (e) => { if (!e.persisted) this.close(); });
      target.addEventListener('pageshow', (e) => { if (e.persisted && this.session.closed) { delete this.session.closed; this.#write(); } });
      const doc = target.document;
      doc?.addEventListener?.('visibilitychange', () => {
        if (doc.visibilityState === 'hidden') this.note({ hidden: true }, true);
        else this.note({ hidden: false, shownAt: Date.now() }, true);
      });
    }
  }

  /** Merges fields into the record. Writes at once on a stage change or `force`, else at most every 250 ms. */
  note(fields, force = false) {
    const changed = fields.stage !== undefined && fields.stage !== this.session.stage;
    Object.assign(this.session, fields);
    const now = Date.now();
    if (changed || force || now - this.last >= 250) { this.last = now; this.#write(); }
  }

  /**
   * Records how memory splits between weights, KV cache and activations (MB), with the ctx and maxChunk it was
   * measured at, so a later kill can be fitted under (see fitMemory).
   * @param {{ fixedMB: number, kvMB: number, actMB: number, ctx: number, maxChunk: number }} split
   */
  split(split) { this.memSplit = split; this.#write(); }

  /** Marks a clean end (pagehide, or the app is done with the model). */
  close() { this.session.closed = true; this.#write(); }

  /** A plain summary for logs and benches. */
  report() {
    return { verdict: this.verdict, previous: this.previous, levels: { mem: this.levels.mem, gpu: this.levels.gpu, capMB: this.levels.capMB }, settings: this.settings };
  }

  #write() {
    this.session.seen = Date.now();
    try { this.storage?.setItem(this.key, JSON.stringify({ levels: this.levels, split: this.memSplit, session: this.session })); } catch { /* private mode, quota: memory is best-effort */ }
  }
}
