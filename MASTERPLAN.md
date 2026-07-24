# Masterplan — Launch OSTIR.AI on Cloudflare (build record)

## What this repo is
The public launch surface for **OSTIR — Quantized AI-Caching for CPU Inference** at **ostir.ai**:
a static marketing site + a hardened, D1-backed design-partner funnel, served by one Cloudflare
Worker. No npm at runtime, no build step — vanilla HTML/CSS/JS. Repo root:
`03-port/03B-ostiar/01-prod/01A-app/ostir-ai-site/`.

This repo is the **company front + funnel only.** The serving **engine** (bench / router /
scheduler / kv / deploy — CPU-native quantized caching: Q4 weights, 3-bit KV cache, RAM→NVMe
session tiering) is a **separate, gated build** and does not run on Cloudflare. It is
**attribution-gated** (`03B-ostiar/CLAUDE.md` bars deriving from the filed contact's platform) and
unbuilt here by design.

## Guardrails (binding)
- **The fundraising deck is CONFIDENTIAL — DO NOT DISTRIBUTE.** No confidential term reaches this
  public site: the raise, cap, instrument, unit economics, projections, use-of-funds, pilot pricing,
  or named prospect companies. The site sells the thesis, never the deal.
- **Why-now / problem statistics are attributed to their named sources as claims**, never presented
  as OSTIR's own measured results. Own-capability copy is forward-looking/illustrative only; the
  in-motion dashboard is hard-labelled "Illustrative simulation — not live telemetry."
- **No team bios on the public site** (living people, sensitive credentials).
- **Engine stays gated.** No agent builds or derives the CPU/SIMD/quantized-caching engine, and no
  outreach to the filed contact, AMD, or any named party.

## Files
- `wrangler.jsonc` — Worker `ostir-ai`; assets `./site` with `run_worker_first: true`; D1
  `ostir-nodes`; `ostir.ai` custom-domain route commented until the domain is wired.
- `worker.js` — security headers (HSTS, `X-Frame-Options: DENY`, strict CSP `default-src 'self'`),
  `/healthz`, and `POST /api/design-partner` (same-origin check → `ostir.ai`, email + enum
  validation, per-IP rate limit ≤5/60s, `INSERT … ON CONFLICT(email) DO NOTHING`, uniform `{ok:true}`).
- `site/index.html` — the launch page (hero, problem, why-now, solution, in-motion, design-partner
  funnel), linking `/assets/ostir.css`.
- `site/assets/` — `ostir.css` (design tokens, light/dark), `telemetry.js` (illustrative sparklines),
  `admission-flow.js` (hero canvas), `waitlist.js` (form → API), `mark.svg`, `favicon.svg`.
- `site/privacy.html`, `site/404.html`, `site/site.webmanifest`, `site/robots.txt` — meta/support.
- `migrations/0001_design_partners.sql` — `design_partners(email UNIQUE, node_count, deployment,
  note, ip, user_agent, created_at)` + `idx_dp_created`.

## Design-partner funnel
Form fields: email · node count (`Under 16 / 16–128 / 128+`) · deployment (`On-prem / Air-gapped /
Edge / Cloud`) · optional note. **No named partners** anywhere on the page. `worker.js` validates
`node_count` / `deployment` against fixed enums; unknown values → `error:"field"`.

## Deploy
See `DEPLOY-RUNBOOK.md`. Summary: `npx wrangler login` → `d1 create ostir-nodes` (paste id into
`wrangler.jsonc`) → `d1 migrations apply ostir-nodes --remote` → `deploy` → add custom domain
`ostir.ai`, uncomment the route, redeploy. Secrets never live in the repo.

## Ship path (house rule)
Build → private GitHub repo `braylo777/ostir-ai` → default-private Artifact preview to forward.
Brandon runs the live Cloudflare deploy from his own account.

## Deliverable
A private, brand-matched `braylo777/ostir-ai` repo on the CPU quantized-caching thesis, a per-node
design-partner funnel, and an Artifact preview link — with a plain note that the engine stays gated
pending the open attribution question.
