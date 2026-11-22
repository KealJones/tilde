// Runs bench/kernel-check.html on a real iPhone in BrowserStack's device cloud and fails if any kernel does.
// The emulated WebGPU in CI (SwiftShader) passed a kernel that gave all zeros on iPhones (Nov 21, 2026); a real
// Apple GPU is the only thing that catches that kind of bug.
//
//   BROWSERSTACK_USERNAME=… BROWSERSTACK_ACCESS_KEY=… node tools/iphone-kernel-check.mjs <kernel-check URL>
//   [--device "iPhone 16 Pro Max"] [--os 27]
//
// Plain WebDriver over fetch, so it needs no packages. Each run takes about a minute of BrowserStack time.

const args = process.argv.slice(2);
const opt = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const url = args.find((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--')));
const device = opt('--device', 'iPhone 16 Pro Max');
const osVersion = opt('--os', '27');
const user = process.env.BROWSERSTACK_USERNAME;
const key = process.env.BROWSERSTACK_ACCESS_KEY;
if (!url || !user || !key) {
  console.error('usage: BROWSERSTACK_USERNAME=… BROWSERSTACK_ACCESS_KEY=… node tools/iphone-kernel-check.mjs <url>');
  process.exit(2);
}

const hub = 'https://hub-cloud.browserstack.com/wd/hub';
const auth = 'Basic ' + Buffer.from(`${user}:${key}`).toString('base64');
async function wd(method, path, body) {
  const res = await fetch(hub + path, { method, headers: { authorization: auth, 'content-type': 'application/json' }, body: body && JSON.stringify(body) });
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(`${method} ${path}: HTTP ${res.status} ${JSON.stringify(json).slice(0, 400)}`);
  return json.value;
}

const started = Date.now();
const session = await wd('POST', '/session', {
  capabilities: { alwaysMatch: { browserName: 'safari', 'bstack:options': {
    deviceName: device, osVersion, realMobile: 'true', buildName: 'tilde kernel-check', sessionName: url,
  } } },
});
const id = session.sessionId;
console.log(`${device}, iOS ${osVersion}: session ${id}`);

let text = '';
let result = null;
try {
  await wd('POST', `/session/${id}/url`, { url });
  for (let i = 0; i < 60 && !result; i++) {
    const [log, done] = await wd('POST', `/session/${id}/execute/sync`, {
      script: 'const l = document.getElementById("log"); return [l ? l.textContent : "", window.__result || null];', args: [],
    });
    text = log; result = done;
    if (!result) await new Promise((r) => setTimeout(r, 3000));
  }
} finally {
  await wd('DELETE', `/session/${id}`).catch(() => {});
}

console.log(text.trimEnd());
const failed = text.split('\n').filter((l) => /FAIL|ERROR|GPU error/.test(l));
console.log(`\n${((Date.now() - started) / 1000).toFixed(0)} s on the device`);
if (!result) { console.error('kernel-check never finished'); process.exit(1); }
if (!result.ok || failed.length) { console.error(`${failed.length} failing line(s):\n${failed.join('\n')}`); process.exit(1); }
console.log('all kernels ok on a real iPhone');
