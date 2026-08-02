# X Bot Detector — Claude Code Context

## What this project is
A Chrome extension that scores X (Twitter) profiles and reply threads for bot likelihood. Core model is XGBoost, trained on 37,438 labelled accounts (F1 = 0.8076), validated against the MGTAB benchmark. Backend is FastAPI, deployed on Hugging Face Spaces. The extension is complete and live on the Chrome Web Store.

Originally built as a General Assembly Data Science Immersive capstone project (BIBF, in partnership with Tamkeen), then productionised beyond the coursework version.

## Key URLs
- GitHub (canonical, personal, MIT licensed): https://github.com/mobucheeri-datsci/x-bot-detector
- GA Enterprise repo (capstone origin, not canonical): git.generalassemb.ly/mobucheeri/twitter-bot-detector
- Hugging Face Space (backend, canonical): https://mobucheeri-twitter-bot-detector.hf.space
- Hugging Face account: mobucheeri

## Stack
- Python, XGBoost, scikit-learn
- FastAPI backend
- Hugging Face Spaces (deployment), Docker
- Chrome Extension APIs, JavaScript (extension frontend)

## Known constraints and past gotchas
- Hugging Face's GitHub username field only resolves to github.com, not GitHub Enterprise instances. Link the GA repo from README.md instead, not from HF profile settings.
- Chrome Web Store does not allow custom thumbnail uploads for link-type media items in the store listing.
- Extension IDs cannot be changed once published.
- Chrome Web Store policy prohibits "review generation" services. Do not build, suggest, or integrate anything that nudges or generates reviews.
- LaTeX (used for README PDF export via Pandoc with xelatex) cannot fetch remote image URLs. Download screenshots locally before compiling. This is the reliable PDF export path; the Markdown PDF VS Code extension has fundamental limits on page break and float control. Eisvogel template is the next thing to try if further formatting issues come up.

## Current state (last known)
- Model, MGTAB benchmark validation, and Chrome extension are complete and live.
- Backend deployed and reachable at the HF Space URL above.
- README and repo are in working order.
- API key auth and rate limiting added to api/app.py (July 2026, not yet deployed). Keys are HMAC-signed with the API_SECRET env var and verified statelessly, issued per install via POST /register (limited to 5 per IP per hour). Rate limits are per key: 60 scans per minute, 1000 per day (free tier). Enforcement of missing keys only happens when REQUIRE_API_KEY=1 is also set, so the rollout order is: deploy backend with API_SECRET set, publish extension 1.1.0 (registers and sends X-API-Key, stores the key in chrome.storage.local), wait for users to update, then set REQUIRE_API_KEY=1. With API_SECRET unset (local dev), auth is disabled entirely. In-memory rate buckets reset on Space restart; acceptable at current scale.

## Open items
- Set REQUIRE_API_KEY=1 on the HF Space once extension 1.1.0 has rolled out (deployed to the Space with API_SECRET set on 31 July 2026; 1.1.0 submitted for Chrome Web Store review the same day; store listing and PRIVACY.md updated to disclose the anonymous access key; a scheduled reminder for the flip runs on 14 August 2026).
- Backend sleep addressed for now (August 2026) with a GitHub Actions keep-alive workflow (.github/workflows/keepalive.yml) that pings /health every 30 minutes; the free tier sleeps only after 48 hours of inactivity, so this keeps it awake and a failed check emails the owner. Caveat: GitHub disables scheduled workflows after 60 days without repo activity (it emails a warning first; any commit resets the clock). Proper always-on hosting behind a user-owned domain (Railway or Fly.io, roughly $5 to 15 per month plus domain) is deferred until the paid tier launches, so the extension only needs one more URL change and store review.
- Demo video may need re-recording to reflect the current production version: hosted backend, updated vocabulary, no log-odds display, no CIB clusters shown.
- Undecided: update GA repo references to point directly at the correct Hugging Face URL, or continue relying on redirects.

## Working conventions
- Full rewrites preferred over diffs or tracked changes when revising text, docs, or commit messages.
- No em dashes. No emojis.
- Formal British register in documentation and public-facing copy (programme, rigour, etc.). Concise connected prose over bullet padding or generic templates.
- Cross-check attributions carefully (e.g. which project belongs to which employer or context) before writing them into docs, READMEs, or commit messages.
- Verify URLs, repo names, and identifiers directly rather than assuming or guessing.

## Notes for future sessions
- This file does not update itself. When something changes (new deployment URL, an open item gets resolved, a new constraint is discovered), add it here and commit it.
- Run a quick scan of the repo structure at the start of a session before making assumptions about current layout, since this file does not track file-level structure.
