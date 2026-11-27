// Runs a bench page in headless Chrome on SwiftShader WebGPU and prints its log.
// Usage: node bench/gpu-test.mjs [page.html?query] (default lfm3-check.html)
import puppeteer from 'puppeteer-core';
import { serve } from './server.mjs';

const page = process.argv[2] || 'lfm3-check.html';
const server = serve();
const chrome = process.env.CHROME || '/usr/bin/google-chrome';
const browser = await puppeteer.launch({
  executablePath: chrome, headless: 'new', protocolTimeout: 0,
  args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader',
    '--use-webgpu-adapter=swiftshader', '--disable-vulkan-surface', '--use-angle=swiftshader'],
});
const tab = await browser.newPage();
tab.on('console', (m) => console.log(m.text()));
tab.on('pageerror', (e) => console.log('pageerror', e.message));
await tab.goto(`http://localhost:${server.address().port}/bench/${page}`);
await tab.waitForFunction('window.__result', { timeout: 0, polling: 1000 });
const r = await tab.evaluate('window.__result');
await browser.close();
server.close();
process.exit(r.ok ? 0 : 1);
