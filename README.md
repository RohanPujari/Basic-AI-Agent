# Weather + Calculator Agentic AI Playground

A small project for learning how an agent uses tools, state, planning, routing,
specialists, reflection, human review, evaluation, and basic security.

The weather tool returns **sample data**, not a live forecast.

## Setup

Use Python 3.12 for the web app. From this folder, create and activate a separate virtual environment (this leaves the existing `.venv` untouched):

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install -r requirements.txt
```

Create a `.env` file and add your Groq API key:

```text
GROQ_API_KEY=your_key_here
```

Keep `.env` private. It is excluded by `.gitignore`.

For local Streamlit secrets instead, copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and add your key there. That private file is also
excluded by `.gitignore`.

## Run the web app

```bash
streamlit run streamlit_app.py
```

Try: `What's the weather in Saratoga Springs NY in Celsius, and is it below 25?`
The page shows the plan, supervisor route, specialist results, reflection, and an
approval checkpoint. Weather is sample data, not a live forecast.

When deploying to Streamlit Community Cloud, add `GROQ_API_KEY` in the app's
Secrets settings. Never commit the actual key.

The original command-line version is still available:

```bash
python agentic_playground.py
```

## Run offline evaluations

```bash
python evaluate_agent.py
```

These checks use a fake LLM response, so they do not need an API key or live
weather. GitHub Actions runs the same evaluations when changes are pushed or a
pull request is opened.

## Project files

- `agentic_playground.py` — the learning implementation.
- `streamlit_app.py` — the visual chat interface.
- `evaluate_agent.py` — repeatable offline behavior checks.
- `requirements.txt` — pinned direct Python dependencies.
- `PROJECT_PLAN.md` — learning sequence and project notes.
