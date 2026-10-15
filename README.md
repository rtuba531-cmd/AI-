# 📄 ATS Resume Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) compatibility and suggests concrete improvements, powered by Google's Gemini Flash model.

## Features

- Upload a resume as **PDF, DOCX or TXT**
- Optional **job description** for a targeted keyword match
- **ATS score (0-100)** = 70% Gemini analysis + 30% rule-based checks
- Category breakdown, strengths, missing keywords
- Prioritized improvements and sample bullet rewrites
- Downloadable JSON report

## How the score works

| Part | Weight | What it measures |
|------|--------|------------------|
| Gemini analysis | 70% | Keywords/relevance, formatting, impact, clarity |
| Rule-based checks | 30% | Email, phone, standard section headings, length, bullets, action verbs, metrics, dates |

The score is an estimate. Real ATS systems differ, so treat it as guidance, not a guarantee.

## Run locally

```bash
git clone https://github.com/<your-username>/ats-resume-checker.git
cd ats-resume-checker

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at https://aistudio.google.com/apikey, then provide it in one of two ways:

**Option A: secrets file (recommended)**
Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-key-here"
# GEMINI_MODEL = "gemini-2.5-flash"   # optional: pin a specific model
```

**Option B: environment variable**
```bash
export GEMINI_API_KEY="your-key-here"      # Windows PowerShell: $env:GEMINI_API_KEY="your-key-here"
```

If neither is set, the app shows an API key box in the sidebar.

```bash
streamlit run app.py
```

## Push to GitHub

Never commit your API key. Create a `.gitignore` first:

```bash
cat > .gitignore << 'EOT'
venv/
__pycache__/
.streamlit/secrets.toml
.env
EOT
```

Then:

```bash
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "Initial commit: ATS resume checker"
git branch -M main
```

Create an empty repo on https://github.com/new (no README or .gitignore), then:

```bash
git remote add origin https://github.com/<your-username>/ats-resume-checker.git
git push -u origin main
```

## Deploy on Streamlit Community Cloud (free)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** and choose your repository, branch `main`, and main file `app.py`.
3. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
4. Click **Deploy**. You get a public `*.streamlit.app` URL in a couple of minutes.

Every `git push` to `main` redeploys automatically. To change secrets later: app menu → **Settings → Secrets**.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Very little text was found" | The PDF is likely a scan/image. Export a text-based PDF or use DOCX. |
| `404` / "model not found" | Set `GEMINI_MODEL` to a model your key can access, e.g. `gemini-2.5-flash`. |
| `429` quota error | Free-tier rate limit. Wait a minute and retry. |
| App asks for API key on Cloud | Secrets not saved; re-check the key name is exactly `GEMINI_API_KEY`. |

## Privacy

Resume text is sent to the Gemini API for analysis. The app itself does not store uploaded files. If you deploy publicly, tell users this, and consider that free-tier Gemini API usage may be used by Google to improve its products.

## Project structure

```
app.py            # Streamlit UI + analysis logic
requirements.txt  # Python dependencies
README.md         # This file
```
