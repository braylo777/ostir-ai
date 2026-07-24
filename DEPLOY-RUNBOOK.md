# DEPLOY RUNBOOK — ostiar.ai

Prereqs: a Cloudflare account, and the `ostiar.ai` domain added to it. `npx` pulls wrangler on demand.

```bash
git clone git@github.com:braylo777/ostiar-ai && cd ostiar-ai

# 1. Auth
npx wrangler login

# 2. Create the D1 database, then paste the printed database_id into wrangler.jsonc
npx wrangler d1 create ostiar-design-partners

# 3. Apply the schema to the remote DB
npx wrangler d1 migrations apply ostiar-design-partners --remote

# 4. Deploy the Worker + site
npx wrangler deploy

# 5. Wire the domain:
#    Cloudflare dashboard → Workers & Pages → ostiar-ai → Settings → Domains & Routes
#    → Add custom domain → ostiar.ai
#    then uncomment the "routes" line in wrangler.jsonc and re-run `npx wrangler deploy`.
```

## Check it
```bash
curl -sI https://ostiar.ai | grep -i content-security-policy      # headers present
curl -s -X POST https://ostiar.ai/api/design-partner \
  -H 'content-type: application/json' \
  -d '{"email":"you@example.com","fleet_size":"64-512","accelerator":"amd-instinct"}'   # {"ok":true}
```

## Read the design-partner list
```bash
npx wrangler d1 execute ostiar-design-partners --remote \
  --command "SELECT email,fleet_size,accelerator,created_at FROM design_partners ORDER BY created_at DESC"
```

No secrets live in this repo. If you later add Turnstile or email, put keys in
`npx wrangler secret put <NAME>` — never commit them.
