// Image preprocessing for LFM3-VL (Qwen3-VL style smart resize, mean/std normalize, 16×16 patches).

const roundEven = (x) => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 ? r - 1 : r; };

/** Target size: multiples of `factor`, pixel count within [minPixels, maxPixels] (llama.cpp's smart_resize). */
export function smartResize(width, height, factor, minPixels, maxPixels) {
  let h = roundEven(height / factor) * factor;
  let w = roundEven(width / factor) * factor;
  if (h * w > maxPixels) {
    const beta = Math.sqrt((height * width) / maxPixels);
    h = Math.floor(height / beta / factor) * factor;
    w = Math.floor(width / beta / factor) * factor;
  } else if (h * w < minPixels) {
    const beta = Math.sqrt(minPixels / (height * width));
    h = Math.ceil((height * beta) / factor) * factor;
    w = Math.ceil((width * beta) / factor) * factor;
  }
  return { width: Math.max(factor, w), height: Math.max(factor, h) };
}

/**
 * Turns an image into normalized 16×16 patches in raster order: Float32Array [nPatches, 3·16·16],
 * each patch laid out channel, row, column (the order of the patch-embedding kernel).
 * @param {ImageBitmap | HTMLImageElement | HTMLCanvasElement | ImageData | {width:number,height:number,data:Uint8ClampedArray}} image
 * @param {{ mean: number[], std: number[], minPixels: number, maxPixels: number, patch?: number, merge?: number }} cfg
 */
export function patchify(image, { mean, std, minPixels, maxPixels, patch = 16, merge = 2 }) {
  const size = smartResize(image.width, image.height, patch * merge, minPixels, maxPixels);
  const rgba = pixels(image, size);
  const pw = size.width / patch, ph = size.height / patch, P = patch * patch;
  const out = new Float32Array(pw * ph * 3 * P);
  for (let py = 0; py < ph; py++) {
    for (let px = 0; px < pw; px++) {
      const base = (py * pw + px) * 3 * P;
      for (let ky = 0; ky < patch; ky++) {
        const row = ((py * patch + ky) * size.width + px * patch) * 4;
        for (let kx = 0; kx < patch; kx++) {
          const i = row + kx * 4;
          for (let c = 0; c < 3; c++) out[base + c * P + ky * patch + kx] = (rgba[i + c] / 255 - mean[c]) / std[c];
        }
      }
    }
  }
  return { patches: out, pw, ph, width: size.width, height: size.height };
}

// RGBA bytes at the target size. Pixel arrays already at that size are used as-is (exact, for testing).
function pixels(image, { width, height }) {
  if (image.data && image.width === width && image.height === height) return image.data;
  if (typeof OffscreenCanvas === 'undefined') throw new Error('resizing an image needs OffscreenCanvas');
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  if (image.data) {
    const src = new OffscreenCanvas(image.width, image.height);
    src.getContext('2d').putImageData(image instanceof ImageData ? image : new ImageData(image.data, image.width, image.height), 0, 0);
    ctx.drawImage(src, 0, 0, width, height);
  } else {
    ctx.drawImage(image, 0, 0, width, height);
  }
  return ctx.getImageData(0, 0, width, height).data;
}
