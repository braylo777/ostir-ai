# DEPLOY RUNBOOK — ostir.ai

Prereqs: a Cloudflare account, and the `ostir.ai` domain added to it. `npx` pulls wrangler on demand.

```bash
git clone git@github.com:braylo777/ostir-ai && cd ostir-ai

# 1. Auth
npx wrangler login

# 2. Create the D1 database, then paste the printed database_id into wrangler.jsonc
npx wrangler d1 create ostir-nodes

# 3. Apply the schema to the remote DB
npx wrangler d1 migrations apply ostir-nodes --remote

# 4. Deploy the Worker + site
npx wrangler deploy

# 5. Wire the domain:
#    Cloudflare dashboard → Workers & Pages → ostir-ai → Settings → Domains & Routes
#    → Add custom domain → ostir.ai
#    then uncomment the "routes" line in wrangler.jsonc and re-run `npx wrangler deploy`.
```

## Check it
```bash
curl -sI https://ostir.ai | grep -i content-security-policy      # headers present
curl -s -X POST https://ostir.ai/api/design-partner \
  -H 'content-type: application/json' \
  -d '{"email":"you@example.com","node_count":"16-128","deployment":"air-gapped"}'   # {"ok":true}
```

## Read the design-partner list
```bash
npx wrangler d1 execute ostir-nodes --remote \
  --command "SELECT email,node_count,deployment,created_at FROM design_partners ORDER BY created_at DESC"
```

No secrets live in this repo. If you later add Turnstile or email, put keys in
`npx wrangler secret put <NAME>` — never commit them.
