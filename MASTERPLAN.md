# Masterplan — Launch OSTIAR.AI on Cloudflare (verbatim build spec)

## Context

Brandon wants a launchable OSTIAR surface on Cloudflare at **ostiar-ai**, a private GitHub, and a
working MVP + UI/UX to send to Jono. Confirmed end-state: **build → private repo → Artifact preview;
he runs the live Cloudflare deploy himself.**

**Scope call:** the orchestration *engine* (Rust/Python/**C++·HIP·K8s on AMD GPUs**) can't run on
Cloudflare, is **attribution-gated** (`03B-ostiar/CLAUDE.md` bars deriving from Kozak's platform), and
is ~10–20% built. So this ships the correct first launch: **public company front + design-partner
funnel**, mirroring the proven CarNimbus Cloudflare pattern (Worker + static `site/` + D1, no npm,
hardened headers). Tooling confirmed: `gh` authed `braylo777`; `node/npm/git` present; `wrangler` via
`npx`. Repo root: `03-port/03B-ostiar/01-prod/01A-app/ostiar-ai-site/`.

Everything below is the **verbatim content of each file**. Executing = transcribing these + running
the commands in §Build and §Verify.

---

## FILE 1 — `wrangler.jsonc`
```jsonc
{
  // OSTIAR — static launch site (Assets) + design-partner funnel (D1), security-hardened.
  "name": "ostiar-ai",
  "compatibility_date": "2025-01-01",
  "main": "worker.js",
  "assets": {
    "directory": "./site",
    "binding": "ASSETS",
    "not_found_handling": "404-page",
    "run_worker_first": true
  },
  "d1_databases": [
    { "binding": "DB", "database_name": "ostiar-design-partners", "database_id": "REPLACE_AFTER_D1_CREATE" }
  ],
  "vars": { "DEV_MODE": "0" }
  // ,"routes": [ { "pattern": "ostiar.ai", "custom_domain": true } ]   // uncomment when domain is wired
}
```

## FILE 2 — `worker.js` (complete)
```js
// OSTIAR — static site (Assets) + design-partner API (D1), security-hardened.
const SEC = {
  "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
  "X-XSS-Protection": "0",
  "Referrer-Policy": "strict-origin-when-cross-origin",
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Resource-Policy": "same-origin",
  "Permissions-Policy": "geolocation=(), microphone=(), camera=(), interest-cohort=()",
  "Content-Security-Policy": [
    "default-src 'self'", "img-src 'self' data:", "style-src 'self' 'unsafe-inline'",
    "font-src 'self'", "script-src 'self'", "connect-src 'self'",
    "manifest-src 'self'", "base-uri 'none'", "form-action 'self'",
    "frame-ancestors 'none'", "object-src 'none'", "upgrade-insecure-requests",
  ].join("; "),
};
const j = (o, s = 200) => new Response(JSON.stringify(o), { status: s, headers: { "content-type": "application/json" } });
const sec = (r) => { const h = new Headers(r.headers); for (const k in SEC) h.set(k, SEC[k]); return new Response(r.body, { status: r.status, headers: h }); };
const FLEET = new Set(["", "<64", "64-512", "512+"]);
const ACCEL = new Set(["", "amd-instinct", "nvidia", "mixed", "other"]);
const EMAIL = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;
function sameOrigin(req){ const o=req.headers.get("Origin")||""; if(!o) return true; try{ return new URL(o).hostname.replace(/^www\./,"")==="ostiar.ai"; }catch(_){ return false; } }

export default {
  async fetch(request, env, ctx) {
    try { return await route(request, env); }
    catch (e) { console.error("route_error:", (e && e.stack) || e); return sec(j({ ok:false, error:"server_error" }, 500)); }
  },
};

async function route(request, env) {
  const url = new URL(request.url);
  if (url.pathname === "/healthz") return sec(j({ ok:true }));
  if (url.pathname === "/api/design-partner") {
    if (request.method !== "POST") return sec(j({ ok:false, error:"method" }, 405));
    if (!sameOrigin(request)) return sec(j({ ok:false, error:"origin" }, 403));
    let b; try { b = await request.json(); } catch(_) { return sec(j({ ok:false, error:"bad_json" }, 400)); }
    const email = String(b.email||"").trim().toLowerCase();
    const fleet = String(b.fleet_size||"").trim();
    const accel = String(b.accelerator||"").trim();
    const note  = String(b.note||"").trim().slice(0, 500);
    if (!EMAIL.test(email) || email.length > 254) return sec(j({ ok:false, error:"email" }, 400));
    if (!FLEET.has(fleet) || !ACCEL.has(accel))    return sec(j({ ok:false, error:"field" }, 400));
    const ip = request.headers.get("CF-Connecting-IP") || "";
    const ua = (request.headers.get("User-Agent") || "").slice(0, 300);
    // naive per-IP rate limit: <=5 inserts / 60s
    try {
      const r = await env.DB.prepare("SELECT COUNT(*) c FROM design_partners WHERE ip=?1 AND created_at > datetime('now','-60 seconds')").bind(ip).first();
      if (r && r.c >= 5) return sec(j({ ok:false, error:"rate" }, 429));
    } catch(_) {}
    try {
      await env.DB.prepare("INSERT INTO design_partners (email,fleet_size,accelerator,note,ip,user_agent,created_at) VALUES (?1,?2,?3,?4,?5,?6,datetime('now')) ON CONFLICT(email) DO NOTHING")
        .bind(email, fleet, accel, note, ip, ua).run();
    } catch(e) { console.error("insert_err:", e); return sec(j({ ok:false, error:"store" }, 500)); }
    return sec(j({ ok:true }));   // uniform response, no enumeration
  }
  // static assets (run_worker_first routes them here); wrap with security headers
  return sec(await env.ASSETS.fetch(request));
}
```

## FILE 3 — `migrations/0001_design_partners.sql`
```sql
CREATE TABLE IF NOT EXISTS design_partners (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT UNIQUE NOT NULL,
  fleet_size TEXT, accelerator TEXT, note TEXT,
  ip TEXT, user_agent TEXT, created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_dp_created ON design_partners(created_at);
```

## FILE 4 — `site/index.html` (complete; all copy public-safe)
```html
<!doctype html><html lang="en" data-theme="light"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OSTIAR — Open Agentics for GPU Inference</title>
<meta name="description" content="OSTIAR is the open, vendor-neutral orchestration layer between your models and your silicon — rack-scale GPU inference, open-core.">
<meta property="og:title" content="OSTIAR — Open Agentics for GPU Inference">
<meta property="og:description" content="The orchestration layer between your models and your silicon. Vendor-neutral, rack-scale, open-core.">
<meta property="og:type" content="website"><meta name="robots" content="index,follow">
<link rel="icon" href="/assets/favicon.svg"><link rel="manifest" href="/site.webmanifest">
<link rel="stylesheet" href="/assets/ostiar.css">
</head><body>
<header class="nav"><a class="brand" href="/"><img src="/assets/mark.svg" width="26" height="26" alt=""><b>OSTIAR</b></a>
  <nav><a href="#how">How it works</a><a href="#partners">Design partners</a>
  <button id="theme" class="ghost" aria-label="Toggle theme">◐</button></nav></header>

<section class="hero"><div class="wrap">
  <p class="eyebrow">OPEN · VENDOR-NEUTRAL · RACK-SCALE</p>
  <h1>Open Agentics for <span class="ind">GPU Inference</span></h1>
  <p class="lede">The orchestration layer between your models and your silicon. One rack, one endpoint — open-core, priced per GPU.</p>
  <div class="cta"><a class="btn" href="#partners">Request design-partner access</a><a class="lnk" href="#how">Read the thesis →</a></div>
  <canvas id="flow" height="220" aria-hidden="true"></canvas>
  <p class="flowcap">MODEL → OSTIAR → COMPUTE · live admission flow</p>
</div></section>

<section class="band"><div class="wrap">
  <p class="eyebrow">01 · THE PROBLEM</p><h2>GPU optimization is the new bottleneck.</h2>
  <div class="cards">
    <div class="card"><b class="stat">5%</b><p>avg GPU utilization across ~23k enterprise clusters.</p><span class="src">Cast AI, 2026</span></div>
    <div class="card"><b class="stat">10×</b><p>gap between best-run and fleet-average GPU clusters.</p><span class="src">Cast AI, 2026</span></div>
    <div class="card"><b class="stat">90%</b><p>of a recurring ML bill is now real-time inference.</p><span class="src">AWS, 2026</span></div>
  </div></div></section>

<section class="band alt"><div class="wrap">
  <p class="eyebrow">02 · WHY NOW</p><h2>The future of orchestration is <span class="ind">open agentics</span>.</h2>
  <div class="cards">
    <div class="card"><b class="stat">150k+</b><p>AI agents a single F-500 could run by 2028.</p><span class="src">Gartner, 2026</span></div>
    <div class="card"><b class="stat">62%</b><p>of organizations are already experimenting with AI agents.</p><span class="src">McKinsey, 2025</span></div>
    <div class="card"><b class="stat">1.70%</b><p>open models now trail closed — down from 8.04% in under a year.</p><span class="src">Stanford, 2025</span></div>
  </div>
  <p class="note">11 vendors now co-develop llm-d under the CNCF. The orchestration layer is going vendor-neutral — and the open, rack-scale lane is still unclaimed.</p></div></section>

<section id="how" class="band"><div class="wrap">
  <p class="eyebrow">03 · THE SOLUTION</p><h2>One rack. One endpoint.</h2>
  <div class="flowmap">
    <div class="col"><span class="lab">MODEL · REQUESTS</span><div class="chip">OpenAI-compatible API</div><div class="chip">Enterprise workloads</div><div class="chip">Agentic workflows</div></div>
    <div class="core"><img src="/assets/mark.svg" width="54" height="54" alt="OSTIAR"><span>OSTIAR control plane</span></div>
    <div class="col"><span class="lab">COMPUTE · GPU POOLS</span><div class="chip">prefill · 01–08</div><div class="chip">decode · 09–24</div><div class="chip">kv-cache · G1–G4</div></div>
  </div>
  <div class="mods"><span>bench · Py</span><span>router · Rust</span><span>scheduler · Py</span><span>kv · C++/HIP</span><span>deploy · Go</span><b class="apache">Apache-2.0 core</b></div>
  <p class="note">Request routing · scheduling · KV-cache management · load balancing · GPU resource management — vendor-neutral across NVIDIA, AMD, and future accelerators. <em>This is what we're building.</em></p>
</div></section>

<section class="band alt"><div class="wrap">
  <p class="eyebrow">04 · IN MOTION</p><h2>Utilization, recovered.</h2>
  <div class="dash">
    <div class="tile"><span class="lab">GOODPUT @ SLO</span><canvas class="spark" data-metric="goodput" height="60"></canvas><b class="metric" data-out="goodput">—</b></div>
    <div class="tile"><span class="lab">KV-CACHE HIT</span><canvas class="spark" data-metric="kv" height="60"></canvas><b class="metric" data-out="kv">—</b></div>
    <div class="tile"><span class="lab">GPU UTILIZATION</span><canvas class="spark" data-metric="util" height="60"></canvas><b class="metric" data-out="util">—</b></div>
  </div>
  <p class="illus">◔ Illustrative simulation — not live telemetry.</p></div></section>

<section id="partners" class="band cta-band"><div class="wrap narrow">
  <p class="eyebrow">05 · DESIGN PARTNERS</p><h2>Building in the open. Come build with us.</h2>
  <p class="lede">We're onboarding a small group of design partners running non-hyperscaler GPU fleets. Open-core, Apache-2.0, priced per GPU. Request access below.</p>
  <form id="dp" novalidate>
    <input type="email" name="email" required placeholder="you@company.com" autocomplete="email">
    <div class="row">
      <select name="fleet_size"><option value="">GPU fleet size</option><option value="&lt;64">Under 64</option><option value="64-512">64–512</option><option value="512+">512+</option></select>
      <select name="accelerator"><option value="">Accelerator</option><option value="amd-instinct">AMD Instinct</option><option value="nvidia">NVIDIA</option><option value="mixed">Mixed</option><option value="other">Other</option></select>
    </div>
    <input type="text" name="note" maxlength="500" placeholder="Anything we should know? (optional)">
    <button class="btn" type="submit">Request access</button>
    <p class="formmsg" id="msg" role="status"></p>
  </form></div></section>

<footer class="foot"><div class="wrap">
  <a class="brand" href="/"><img src="/assets/mark.svg" width="22" height="22" alt=""><b>OSTIAR</b></a>
  <span>Open Agentics for GPU Inference · Apache-2.0 · ostiar.ai</span>
  <a href="/privacy.html">Privacy</a></div></footer>

<script src="/assets/admission-flow.js" defer></script>
<script src="/assets/telemetry.js" defer></script>
<script src="/assets/waitlist.js" defer></script>
<script>document.getElementById('theme').onclick=()=>{const r=document.documentElement;r.dataset.theme=r.dataset.theme==='dark'?'light':'dark';};
if(matchMedia('(prefers-color-scheme:dark)').matches)document.documentElement.dataset.theme='dark';</script>
</body></html>
```

## FILE 5 — `site/assets/ostiar.css` (complete)
```css
:root{--indigo:#5B5BF6;--indigo-600:#4A44E0;--ink:#0E0E13;--paper:#F6F7FB;--card:#fff;--line:#E6E7F0;--muted:#6B6D80;
 --sans:system-ui,-apple-system,"Segoe UI",sans-serif;--mono:ui-monospace,"SF Mono",Menlo,monospace;--w:1080px}
:root[data-theme=dark]{--ink:#EDEEF5;--paper:#0B0B12;--card:#14141E;--line:#23232F;--muted:#9A9BB0}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:var(--w);margin:0 auto;padding:0 24px}.narrow{max-width:640px}
.eyebrow{font-family:var(--mono);font-size:.72rem;letter-spacing:.18em;color:var(--muted);margin:0 0 .6rem}
h1{font-size:clamp(2.2rem,6vw,4rem);line-height:1.02;letter-spacing:-.02em;margin:.2rem 0;text-wrap:balance}
h2{font-size:clamp(1.5rem,3.5vw,2.3rem);letter-spacing:-.015em;margin:.2rem 0 1.2rem;text-wrap:balance}
.ind{color:var(--indigo)}.lede{font-size:1.15rem;color:var(--muted);max-width:46ch}
.nav{display:flex;justify-content:space-between;align-items:center;padding:16px 24px;max-width:var(--w);margin:0 auto}
.brand{display:flex;gap:.5rem;align-items:center;text-decoration:none;color:inherit}.brand b{letter-spacing:.14em;font-weight:800}
.nav nav{display:flex;gap:1.2rem;align-items:center}.nav a{color:var(--muted);text-decoration:none;font-size:.92rem}
.ghost{background:none;border:1px solid var(--line);border-radius:8px;color:var(--ink);cursor:pointer;padding:.3rem .5rem}
.hero{padding:48px 0 24px;text-align:center}.hero .cta{display:flex;gap:1rem;justify-content:center;align-items:center;margin:1.4rem 0}
.hero .lede{margin:1rem auto}
.btn{background:var(--indigo);color:#fff;border:0;border-radius:10px;padding:.8rem 1.3rem;font-weight:600;text-decoration:none;cursor:pointer;font-size:1rem}
.btn:hover{background:var(--indigo-600)}.lnk{color:var(--indigo);text-decoration:none;font-weight:600}
#flow{width:100%;max-width:900px;margin:1rem auto 0;display:block}.flowcap{font-family:var(--mono);font-size:.7rem;letter-spacing:.16em;color:var(--muted)}
.band{padding:64px 0;border-top:1px solid var(--line)}.band.alt{background:color-mix(in srgb,var(--card) 55%,var(--paper))}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:24px;min-height:150px}
.stat{font-size:2.6rem;color:var(--indigo);display:block;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.card p{margin:.4rem 0}.src{font-family:var(--mono);font-size:.68rem;color:var(--muted)}
.note{color:var(--muted);margin-top:1.4rem;max-width:60ch}.note em{color:var(--ink);font-style:normal;font-weight:600}
.flowmap{display:grid;grid-template-columns:1fr auto 1fr;gap:22px;align-items:center;margin:1rem 0}
.flowmap .col{display:flex;flex-direction:column;gap:10px}.lab{font-family:var(--mono);font-size:.68rem;letter-spacing:.14em;color:var(--muted)}
.chip{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.6rem .8rem;font-size:.9rem}
.core{display:flex;flex-direction:column;align-items:center;gap:.4rem;color:var(--indigo);font-family:var(--mono);font-size:.72rem;text-align:center}
.mods{display:flex;flex-wrap:wrap;gap:10px;margin-top:16px;align-items:center}
.mods span{font-family:var(--mono);font-size:.74rem;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.35rem .6rem}
.apache{color:var(--indigo);font-family:var(--mono);font-size:.74rem}
.dash{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px}
.spark{width:100%;display:block;margin:.4rem 0}.metric{font-size:1.8rem;color:var(--indigo);font-variant-numeric:tabular-nums}
.illus{font-family:var(--mono);font-size:.72rem;color:var(--muted);margin-top:1rem}
.cta-band{text-align:center}form{display:flex;flex-direction:column;gap:12px;margin-top:1.4rem;text-align:left}
input,select{font:inherit;padding:.8rem;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--ink);width:100%}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}.formmsg{font-size:.9rem;min-height:1.2em;margin:.2rem 0 0}
.formmsg.ok{color:#12855a}.formmsg.err{color:#c0392b}
.foot{border-top:1px solid var(--line);padding:28px 0}.foot .wrap{display:flex;gap:1rem;justify-content:space-between;align-items:center;flex-wrap:wrap}
.foot span{color:var(--muted);font-size:.85rem}.foot a{color:var(--muted);text-decoration:none}
@media(max-width:760px){.cards,.dash{grid-template-columns:1fr}.flowmap{grid-template-columns:1fr}.row{grid-template-columns:1fr}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
```

## FILE 6 — `site/assets/admission-flow.js` (complete)
```js
(function(){const c=document.getElementById('flow');if(!c)return;const x=c.getContext('2d');
let W,H,dpr=Math.min(devicePixelRatio||1,2);const reduce=matchMedia('(prefers-reduced-motion:reduce)').matches;
const ind='#5B5BF6';function size(){W=c.clientWidth;H=c.height;c.width=W*dpr;c.height=H*dpr;x.setTransform(dpr,0,0,dpr,0,0);}
addEventListener('resize',size);size();
const cx=()=>W*0.5,cy=()=>H*0.5;const inLanes=[.28,.5,.72],outLanes=[.28,.5,.72];
let dots=[];function spawn(){const l=inLanes[Math.floor(Math.random()*3)];dots.push({p:0,y:l,o:outLanes[Math.floor(Math.random()*3)]});}
function frame(t){x.clearRect(0,0,W,H);
 // paths
 x.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--line');x.lineWidth=1;
 inLanes.forEach(l=>{x.setLineDash([4,5]);x.beginPath();x.moveTo(0,H*l);x.lineTo(cx(),cy());x.stroke();});
 outLanes.forEach(l=>{x.beginPath();x.moveTo(cx(),cy());x.lineTo(W,H*l);x.stroke();});x.setLineDash([]);
 // aperture
 const pulse=6+3*Math.sin(t/400);x.fillStyle=ind;x.globalAlpha=.15;x.beginPath();x.arc(cx(),cy(),22+pulse,0,7);x.fill();
 x.globalAlpha=1;x.strokeStyle=ind;x.lineWidth=2.5;x.strokeRect(cx()-10,cy()-16,20,32);
 // dots
 dots.forEach(d=>{d.p+=0.012;let px,py;if(d.p<.5){const k=d.p/.5;px=k*cx();py=H*d.y+(cy()-H*d.y)*k;}else{const k=(d.p-.5)/.5;px=cx()+k*(W-cx());py=cy()+(H*d.o-cy())*k;}
   x.fillStyle=ind;x.beginPath();x.arc(px,py,3,0,7);x.fill();});
 dots=dots.filter(d=>d.p<1);requestAnimationFrame(frame);}
if(reduce){spawn();spawn();spawn();frame(0);}else{setInterval(spawn,520);requestAnimationFrame(frame);}
})();
```

## FILE 7 — `site/assets/telemetry.js` (complete)
```js
(function(){const seeds={goodput:{v:34,t:88,u:'%'},kv:{v:41,t:94,u:'%'},util:{v:11,t:63,u:'%'}};
const reduce=matchMedia('(prefers-reduced-motion:reduce)').matches;const ind='#5B5BF6';
document.querySelectorAll('.spark').forEach(cv=>{const m=cv.dataset.metric,s=seeds[m];s.hist=Array.from({length:40},(_,i)=>s.v+(s.t-s.v)*(i/40)+ (Math.sin(i)*2));
 const x=cv.getContext('2d');function draw(){const W=cv.clientWidth,H=cv.height,dpr=Math.min(devicePixelRatio||1,2);cv.width=W*dpr;cv.height=H*dpr;x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,W,H);
   const mn=Math.min(...s.hist),mx=Math.max(...s.hist)||1;x.strokeStyle=ind;x.lineWidth=2;x.beginPath();
   s.hist.forEach((v,i)=>{const px=i/(s.hist.length-1)*W,py=H-((v-mn)/(mx-mn||1))*(H-8)-4;i?x.lineTo(px,py):x.moveTo(px,py);});x.stroke();
   x.globalAlpha=.12;x.lineTo(W,H);x.lineTo(0,H);x.closePath();x.fillStyle=ind;x.fill();x.globalAlpha=1;}
 cv.__draw=draw;draw();const out=document.querySelector('[data-out='+m+']');function tick(){s.v+=(s.t-s.v)*0.04+(Math.random()-.5);s.hist.push(s.v);s.hist.shift();out.textContent=Math.round(s.v)+s.u;draw();}
 out.textContent=Math.round(s.v)+s.u;if(!reduce){let iv=setInterval(()=>{if(!document.hidden)tick();},900);} });
})();
```

## FILE 8 — `site/assets/waitlist.js` (complete)
```js
(function(){const f=document.getElementById('dp'),msg=document.getElementById('msg');if(!f)return;
f.addEventListener('submit',async e=>{e.preventDefault();msg.className='formmsg';msg.textContent='Sending…';
 const b={email:f.email.value.trim(),fleet_size:f.fleet_size.value,accelerator:f.accelerator.value,note:f.note.value.trim()};
 if(!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(b.email)){msg.className='formmsg err';msg.textContent='Please enter a valid email.';return;}
 try{const r=await fetch('/api/design-partner',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(b)});
   const d=await r.json();if(d.ok){f.reset();msg.className='formmsg ok';msg.textContent="You're on the list. We'll be in touch.";}
   else{msg.className='formmsg err';msg.textContent='Something went wrong — try again.';}}
 catch(_){msg.className='formmsg err';msg.textContent='Network error — try again.';}});
})();
```

## FILES 9–13 — small statics
- **`site/assets/mark.svg`** & **`site/assets/favicon.svg`** — the IO mark:
  ```svg
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><rect x="5" y="5" width="38" height="38" rx="11" fill="none" stroke="#0E0E13" stroke-width="4"/><rect x="20" y="14" width="8" height="20" rx="4" fill="#5B5BF6"/></svg>
  ```
- **`site/privacy.html`** — minimal page: what's collected (email + optional fleet/accelerator/note),
  why (design-partner contact only), no third-party sharing, contact `hello@ostiar.ai`, links home.
- **`site/404.html`** — branded "Lost the signal." + link home.
- **`site/robots.txt`** — `User-agent: *\nAllow: /\nSitemap: https://ostiar.ai/sitemap.xml` (sitemap
  optional; drop the line if not generating one).
- **`site/site.webmanifest`** — name/short_name OSTIAR, theme `#5B5BF6`, icon `/assets/favicon.svg`.

## FILES 14–17 — repo meta
- **`LICENSE`** — Apache-2.0 full text (matches the deck's open-core positioning).
- **`.gitignore`** — `.wrangler/` `node_modules/` `.dev.vars` `*.log` `.DS_Store`
- **`README.md`** — what it is (launch site + design-partner funnel), local dev
  (`npx wrangler dev`), deploy pointer to `DEPLOY-RUNBOOK.md`, "engine is a separate gated repo."
- **`DEPLOY-RUNBOOK.md`**:
  ```
  npx wrangler login
  npx wrangler d1 create ostiar-design-partners      # paste id → wrangler.jsonc
  npx wrangler d1 migrations apply ostiar-design-partners --remote
  npx wrangler deploy
  # Dashboard → Workers → ostiar-ai → Custom domain → ostiar.ai; uncomment routes[] and redeploy
  ```

---

## Build order (execution)

1. `Edit 03B-ostiar/CLAUDE.md`: allowed-line → add `; source code under 01-prod/01A-app/ostiar-ai-site/`.
2. Create the tree above; write FILES 1–17 exactly.
3. `cd …/ostiar-ai-site && git init -b main && git add -A && git commit -m "feat: OSTIAR launch site + design-partner funnel"`
4. Verify locally (below).
5. **Artifact preview:** copy `site/index.html` → a self-contained single file (inline `ostiar.css`
   into `<style>`, inline the three JS into `<script>`, inline `mark.svg` as a data-URI, strip the
   `/api` fetch to a friendly "preview — form disabled" message), write to scratchpad, publish via
   Artifact (favicon 🔲, title "OSTIAR — Open Agentics for GPU Inference"). Hand Brandon the link.
6. `gh repo create braylo777/ostiar-ai --private --source=. --push` (authorized by "make the call").
7. File `MASTERPLAN.md` (this) into the repo AND a copy into
   `03B-ostiar/08-exec/08E-decis/2026-07-24-ostiar-ai-launch-build/`; append `03B-ostiar/_autofile-log.md`; vault-backup.

## Verify (run before handoff; expected results noted)

- **Static render:** `cd site && python3 -m http.server 8787` → load `:8787` → hero canvas animates;
  five sections render; theme toggle flips light/dark legibly; at 375px width no horizontal scroll;
  with OS reduced-motion the hero shows a static frame and telemetry stops.
- **Worker API:** `npx wrangler d1 create` (local) + `wrangler d1 migrations apply … --local` +
  `npx wrangler dev`; then:
  - valid: `curl -s -X POST localhost:8787/api/design-partner -H 'content-type:application/json' -d '{"email":"a@b.co","fleet_size":"64-512","accelerator":"amd-instinct"}'` → `{"ok":true}`; row present.
  - bad email → `{"ok":false,"error":"email"}` (400); bad enum (`"accelerator":"tpu"`) → `error:"field"`; 6th rapid post from same IP → `429`.
  - `curl -sI localhost:8787/` shows `content-security-policy`, `strict-transport-security`, `x-frame-options: DENY`.
- **Confidential-leak gate:** `grep -Rai -E '1\.75M|15M cap|SAFE|\$885|TensorWave|Hot Aisle|Vultr|CoreWeave|Crusoe|Nebius|ACV|LTV|CAC|EBITDA' site/` → **0 hits** (printed).
- **Attribution gate:** `grep -Rai -E 'kozak|carrieros|carriera' .` → **0 hits**.
- **Secrets gate:** `git ls-files | grep -cE 'dev\.vars|\.wrangler|\.env'` → **0**.
- **Repo:** `gh repo view braylo777/ostiar-ai --json visibility -q .visibility` → `private`.
- **Artifact:** link loads, self-contained (no network), theme toggles, form shows preview-disabled note.

## Deliverable to Brandon

Private `braylo777/ostiar-ai` repo + an Artifact link to forward to Jono today; a brand-matched,
hardened, honest launch site with a working design-partner funnel; `DEPLOY-RUNBOOK.md` to go live on
ostiar.ai from his own Cloudflare; and a plain note of what was left out and why — the engine
(Cloudflare-incompatible + attribution-gated) and every confidential deck term.
