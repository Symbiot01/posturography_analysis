# Posturography Analysis Dashboard

An end-to-end Streamlit application for uploading raw posturography `.txt` exports, extracting structured metrics, visualising grouped bar and radar charts, and generating AI-assisted clinical interpretations via the Google Gemini API.

## Features
- Upload and parse multiple NeuroCom-style text reports in one session.
- Compare any metric across files with grouped Plotly bar charts and automatic change statistics.
- Build comprehensive radar profiles for a selected test condition with normalized scores.
- View tidy Pandas tables including change %, mean, and standard deviation where applicable.
- Trigger Gemini-powered narrative interpretations (requires API key).

## Quickstart
```bash
cd /workspaces/posturography_analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Secrets Configuration
Create `.streamlit/secrets.toml` (not committed) and add your Gemini key:
```toml
GEMINI_API_KEY = "your-key"
```

## Running the App
```bash
source .venv/bin/activate
streamlit run app.py
```
Visit the provided local URL, upload one or more `.txt` exports, choose an analysis mode, and (optionally) click **Generate Interpretation**.

## Testing / Smoke Check
```bash
source .venv/bin/activate
streamlit run app.py --server.headless true --server.port 8787
```
Press `Ctrl+C` (or stop the process) after confirming the app starts without errors.

## Notes
- No sample data is bundled; use your own posturography exports.
- Keep API keys out of Git by relying on `.streamlit/secrets.toml`.
- To switch Gemini models, change the `model_name` string in `app.py` (default: `gemini-1.5-flash`).