# ostiar-ai

**OSTIAR — Open Agentics for GPU Inference.** The public launch site + design-partner funnel for
ostiar.ai. Static site served by a Cloudflare Worker, with a hardened D1-backed design-partner
signup. No npm at runtime, no build step — vanilla HTML/CSS/JS.

## Structure
- `site/` — the static launch page and assets (`index.html`, `assets/`, `privacy.html`, `404.html`).
- `worker.js` — serves the site with security headers + the `POST /api/design-partner` endpoint.
- `migrations/` — D1 schema for the design-partner list.
- `wrangler.jsonc` — Cloudflare config.

## Local dev
```
npx wrangler d1 migrations apply ostiar-design-partners --local
npx wrangler dev
```
Static-only preview (no API): `cd site && python3 -m http.server 8787`.

## Deploy
See `DEPLOY-RUNBOOK.md`.

## Note on scope
This repo is the **company front + funnel**. The orchestration **engine** (bench / router /
scheduler / kv / deploy) is a separate, gated build — it targets AMD Instinct GPUs / ROCm / K8s and
does not run on Cloudflare.
