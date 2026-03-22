# AirBench Local Setup

This repo's runnable workflow is the AirBench pipeline under `scripts/airbench_gepa/` and `scripts/airbench_autoresearch/`.

## Recommended path: Gemini + Autoresearch

This is the most direct Gemini-supported path in the current repo.

## 1. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Authenticate Modal

Install and authenticate the Modal CLI in the same environment:

```bash
python -m modal setup
```

After this, `modal run ...` and the Python scripts that call `app.run()` should be able to talk to your Modal account.

## 3. Add API keys

Create a repo-root `.env`:

```bash
cp .env.example .env
```

For Gemini, set one of:

```env
GEMINI_API_KEY=your_key_here
```

or

```env
GOOGLE_API_KEY=your_key_here
```

The autoresearch loop auto-loads `.env` from the repo root.

## 4. Run a smoke evaluation

This checks that Modal execution works before starting the full loop:

```bash
python scripts/airbench_autoresearch/run_candidate.py --mode proxy --modal-show-output
```

## 5. Run the Gemini loop

```bash
python scripts/airbench_autoresearch/run_loop.py \
  --max-attempts 5 \
  --model gemini/gemini-3.1-flash-lite-preview \
  --modal-show-output
```

Artifacts are written under `data/airbench/autoresearch_runs/<timestamp>/`.

## Notes

- Modal is still required even if you use Gemini, because evaluation runs on a remote Modal GPU.
- `scripts/airbench_gepa/run_gepa_airbench94.py` is more explicitly wired for OpenAI auth; for Gemini, prefer the autoresearch path above.
- Current local Python in this workspace does not have `modal`, `litellm`, or `openai` installed yet, so step 1 is still required.
