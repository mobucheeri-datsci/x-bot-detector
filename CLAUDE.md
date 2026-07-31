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

## Open items
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
