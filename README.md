# ostir-ai

**OSTIR — Quantized AI-Caching for CPU Inference.**

Serve today's models on the CPU fleets you already own. OSTIR is an open-core serving layer that
makes commodity CPUs a first-class inference target through aggressive, accuracy-aware quantization
and cache tiering — **Q4 weights, 3-bit KV cache, RAM→NVMe session tiering** — behind one
OpenAI-compatible endpoint. Priced per node, deployable on-prem, air-gapped, or at the edge.

This repository is the **public launch site + design-partner funnel** for [ostir.ai](https://ostir.ai).
Static site served by a single Cloudflare Worker, with a hardened, D1-backed design-partner signup.
No npm at runtime, no build step — vanilla HTML/CSS/JS.

## Why this exists

Inference — not training — is where AI spend actually lives, and that bill scales with success:

- ~**90%** of an AI system's lifetime cost is inference, not training *(FinOps Foundation, 2026)*.
- **73%** of organizations report AI costs blew past their original budget *(State of FinOps, 2026)*.
- Meanwhile the serving layer is moving **on-prem, CPU-native, and open** — **55%** of enterprise
  inference now runs on-prem or edge, up from 12% in 2023 *(industry survey, 2026)*.

Two shifts make CPU serving viable *now*: quantization research is crushing KV-cache memory to ~3
bits (~6× smaller — *Google Research, TurboQuant, 2026*), and AMX-class CPU instructions roughly
double quantized-LLM throughput (*OpenMetal, 2025*). The sovereign, quantized, CPU-native lane is
still unclaimed. OSTIR is built to claim it.

*(Figures above are attributed to their named sources as third-party claims, not OSTIR's own
measured results.)*

## What OSTIR does

- **CPU-native serving** — precision-adaptive **Q4 weights** and fused CPU kernels turn commodity
  fleets into inference capacity, no GPU required.
- **KV-cache-aware** — **3-bit KV cache** plus **RAM→NVMe session tiering** keep long-context and
  many-session workloads inside the memory you have.
- **One endpoint** — an OpenAI-compatible API in front of the fleet; drop-in for existing enterprise
  and agentic workloads.
- **Sovereign by default** — air-gap-ready, vendor-neutral, open-core (Apache-2.0 core), so
  regulated and sovereignty-constrained operators can run it on hardware they already control.

**Engine modules** (separate, gated build — see *Scope* below): `bench · Py`, `router · Rust`,
`scheduler · Py`, `kv · C++/SIMD`, `deploy · Go`.

## Go-to-market

Open-source wedge first — ship the serving layer in the open to earn adoption among teams already
running CPU inference — then land **regulated, sovereignty-constrained fleets** where air-gap and
vendor-neutrality are hard requirements. Design partners are onboarded through the funnel in this
repo. *(Commercial terms, pricing, projections, and named prospects are confidential and are not
recorded in this repository.)*

## Structure

- `site/` — the static launch page and assets (`index.html`, `assets/`, `privacy.html`, `404.html`).
- `worker.js` — serves the site with security headers + the `POST /api/design-partner` endpoint.
- `migrations/` — D1 schema for the design-partner list.
- `wrangler.jsonc` — Cloudflare config.

## Local dev

```
npx wrangler d1 migrations apply ostir-nodes --local
npx wrangler dev
```

Static-only preview (no API): `cd site && python3 -m http.server 8787`.

## Deploy

See `DEPLOY-RUNBOOK.md`.

## Scope of this repo

This repo is the **company front + funnel only.** The serving **engine** (bench / router /
scheduler / kv / deploy) is a **separate, gated build** — CPU-native quantized caching — and does
**not** run on Cloudflare. It is not present in this repository by design.
