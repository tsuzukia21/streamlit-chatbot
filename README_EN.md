## Streamlit Chatbot (English)

This is a Streamlit-based chatbot supporting multimodal inputs (text/images/audio). Conversation history is persisted to Firestore and images to Cloud Storage. You can switch among models (Anthropic, Google, OpenAI) and visualize reasoning/thinking when supported.

### Key Features
- Model switching: Claude Sonnet 4.5 / Gemini 2.5 Pro / GPT‑5 (via LangChain wrapper)
- Tool augmentation: Web search tools attached to each model for up-to-date topics
- Multimodal input: image upload and speech recognition (Whisper)
- Conversation management: create new conversation, auto-generate titles, edit, branch from past messages
- Persistence: Firestore (text), Cloud Storage (images)
- Reasoning visualization: show reasoning/thinking
- Authentication: Google sign-in using `st.login()` (mount `secrets.toml` from Secret Manager in production) + optional email allowlist via `allowed_emails`

---

## Project Structure

```
streamlit-chatbot/
  main.py                     # Streamlit app
  core/
    config.py                 # App config (system prompt, token limits, etc.)
    MODEL_CONFIG.py           # Model definitions / LLM factory
    llm_handler.py            # LangChain chains and streaming
    conversation.py           # Create/load/delete conversations, title generation
    database.py               # Firestore / Cloud Storage persistence
    ui_components.py          # Message UI, editing, reasoning collapsible, etc.
  requirements.txt            # pip dependencies
  pyproject.toml, poetry.lock # Poetry dependencies
  Dockerfile                  # Container for Cloud Run
  .streamlit/
    secrets.toml              # Auth/settings (mounted from Secret Manager in prod)
  README.md                   # Japanese README
  README_EN.md                # This file
```

---

## Prerequisites
- Python
- GCP: Firestore (Native mode), Cloud Storage bucket
- API keys (as needed)
  - OpenAI: `OPENAI_API_KEY`
  - Anthropic: `ANTHROPIC_API_KEY`
  - Gemini: `GOOGLE_API_KEY`

---

## Local Setup

1) Install dependencies (pip or Poetry)
```bash
# pip
pip install -r requirements.txt

# or Poetry
poetry install
```

2) Create `.streamlit/secrets.toml` (for local run)
```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "[strong random string]"
client_id = "[Google OAuth Client ID]"
client_secret = "[Google OAuth Client Secret]"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
allowed_emails = ["your-test-user@example.com"] # Optional allowlist. Checked for exact match
```

3) Environment variables (as needed)
```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...

# Only when using a service account key locally
export GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/service-account.json
```

4) Run
```bash
streamlit run main.py
```

### Access Control (Optional Email Allowlist)
- If you set `[auth].allowed_emails` in `secrets.toml`, the app will verify `st.user.email` after login and immediately block + log out users not listed (exact match).
- If `allowed_emails` is missing or empty, this check is skipped (control relies solely on Google OAuth settings).
- Implementation location: right after the "logged-in" branch in the sidebar of `main.py`.

Example (`secrets.toml`):
```toml
[auth]
...
allowed_emails = ["your-test-user@example.com", "another@example.com"]
```

Notes:
- Google auth with `st.login()`/`st.user`/`st.logout()` and `secrets.toml` is well explained in this article (Japanese): [Zenn: Streamlit Google Login](https://zenn.dev/datum_studio/articles/c964f9e38379f4)
- Browser permissions are required for image upload and speech recognition.

---

## Secrets and Env Vars (Production)

In production (Cloud Run), do not commit `.streamlit/secrets.toml`. Instead, register the secret in Secret Manager and mount it as a file at `/app/.streamlit/secrets.toml` inside the container.

### Example Steps
1) Register in Secret Manager
```bash
gcloud secrets create streamlit-secrets --replication-policy=automatic
gcloud secrets versions add streamlit-secrets --data-file=.streamlit/secrets.toml
```

2) Mount as a file on Cloud Run
- In Cloud Console > Cloud Run > Service > Edit > Security > Secrets, configure `streamlit-secrets` to be mounted as a file at `/app/.streamlit/secrets.toml`.
- The app will then read `secrets.toml` from the same path as local.

3) Additional environment variables (as needed)
- Manage `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` in Secret Manager and expose them as Cloud Run env vars
- Setting up LangSmith is recommended

---

## GCP Setup

### 1. Project Preparation
- Enable Firestore (Native mode) with the same region as Cloud Run/Storage
- Create a Cloud Storage bucket

### 2. Service Account and Roles
Grant the Cloud Run runtime service account at least:
- `roles/datastore.user` (Firestore)
- `roles/storage.objectAdmin` (Cloud Storage)
- Recommended: `roles/logging.logWriter`

### 3. Deploy to Cloud Run
Example build and deploy:
```bash
# Build & push to Artifact Registry
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/REPO/streamlit-chatbot:latest

# Deploy (configure secret file mount in console or via manifest)
gcloud run deploy streamlit-chatbot \
  --image REGION-docker.pkg.dev/PROJECT_ID/REPO/streamlit-chatbot:latest \
  --platform managed \
  --region REGION \
  --service-account YOUR_SA@PROJECT_ID.iam.gserviceaccount.com \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=...,ANTHROPIC_API_KEY=...,GOOGLE_API_KEY=...
```

Notes:
- If you do not want the service to be public, use `--no-allow-unauthenticated` and protect it with IAP or alternatives
- If images are not displayed, check `GCS_BUCKET_NAME` configuration, bucket permissions, and CORS

---

## Data Model (Persistence)
- Firestore collections
  - `conversations/{conversationId}`: `user_id`, `title`, `total_tokens`, `is_deleted`, `created_at`, `updated_at`
  - `conversations/{conversationId}/messages/{messageId}`: `role`, `content(json)`, `reasoning`, `created_at`
- Cloud Storage
  - Saved as `images/conv{conversationId}_msg{messageId}_{index}.{ext}`
  - Replace data URI with GCS path at write time; restore data URI at read time

---

## Docker Run
The Dockerfile uses Poetry to resolve dependencies and sets `PORT=8080` for Cloud Run.
```bash
docker build -t streamlit-chatbot:local .
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=... -e ANTHROPIC_API_KEY=... -e GOOGLE_API_KEY=... \
  streamlit-chatbot:local
```

---


