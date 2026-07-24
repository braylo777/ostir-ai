// OSTIR — static site (Assets) + design-partner API (D1), security-hardened.
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
const NODES  = new Set(["", "<16", "16-128", "128+"]);
const DEPLOY = new Set(["", "on-prem", "air-gapped", "edge", "cloud"]);
const EMAIL = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;
function sameOrigin(req){ const o=req.headers.get("Origin")||""; if(!o) return true; try{ return new URL(o).hostname.replace(/^www\./,"")==="ostir.ai"; }catch(_){ return false; } }

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
    const nodes = String(b.node_count||"").trim();
    const dep   = String(b.deployment||"").trim();
    const note  = String(b.note||"").trim().slice(0, 500);
    if (!EMAIL.test(email) || email.length > 254) return sec(j({ ok:false, error:"email" }, 400));
    if (!NODES.has(nodes) || !DEPLOY.has(dep))     return sec(j({ ok:false, error:"field" }, 400));
    const ip = request.headers.get("CF-Connecting-IP") || "";
    const ua = (request.headers.get("User-Agent") || "").slice(0, 300);
    // naive per-IP rate limit: <=5 inserts / 60s
    try {
      const r = await env.DB.prepare("SELECT COUNT(*) c FROM design_partners WHERE ip=?1 AND created_at > datetime('now','-60 seconds')").bind(ip).first();
      if (r && r.c >= 5) return sec(j({ ok:false, error:"rate" }, 429));
    } catch(_) {}
    try {
      await env.DB.prepare("INSERT INTO design_partners (email,node_count,deployment,note,ip,user_agent,created_at) VALUES (?1,?2,?3,?4,?5,?6,datetime('now')) ON CONFLICT(email) DO NOTHING")
        .bind(email, nodes, dep, note, ip, ua).run();
    } catch(e) { console.error("insert_err:", e); return sec(j({ ok:false, error:"store" }, 500)); }
    return sec(j({ ok:true }));   // uniform response, no enumeration
  }
  // static assets (run_worker_first routes them here); wrap with security headers
  return sec(await env.ASSETS.fetch(request));
}
