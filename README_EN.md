## Streamlit Chatbot (English)

This is a Streamlit-based chatbot supporting multimodal inputs (text/images/audio). Conversation history is persisted to DynamoDB and images to S3. You can switch among models (Anthropic, Google, OpenAI) and visualize reasoning/thinking when supported.

### Key Features
- Model switching: Claude Opus 4.5 / Gemini 3.0 Pro / GPT‑5.1 (via LangChain wrapper)
- Tool augmentation: Web search tools attached to each model for up-to-date topics
- Multimodal input: image upload and speech recognition (Whisper)
- Conversation management: create new conversation, auto-generate titles, edit, branch from past messages
- Persistence: DynamoDB (text), S3 (images)
- Reasoning visualization: show reasoning/thinking
- Authentication: Google sign-in using `st.login()` + optional email allowlist via `ALLOWED_EMAILS` env var

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
    database.py               # DynamoDB / S3 persistence
    ui_components.py          # Message UI, editing, reasoning collapsible, etc.
  requirements.txt            # pip dependencies
  pyproject.toml, poetry.lock # Poetry dependencies
  Dockerfile                  # Container for App Runner
  README.md                   # Japanese README
  README_EN.md                # This file
```

---

## Prerequisites
- Python 3.11+
- AWS: DynamoDB table, S3 bucket (ap-northeast-1 recommended)
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

2) Set up AWS resources
```bash
# Create DynamoDB table
aws dynamodb create-table \
  --table-name ChatbotData \
  --attribute-definitions \
    AttributeName=pk,AttributeType=S \
    AttributeName=sk,AttributeType=S \
    AttributeName=user_id,AttributeType=S \
    AttributeName=updated_at,AttributeType=N \
  --key-schema \
    AttributeName=pk,KeyType=HASH \
    AttributeName=sk,KeyType=RANGE \
  --global-secondary-indexes '[{"IndexName":"UserConversationsIndex","KeySchema":[{"AttributeName":"user_id","KeyType":"HASH"},{"AttributeName":"updated_at","KeyType":"RANGE"}],"Projection":{"ProjectionType":"ALL"},"ProvisionedThroughput":{"ReadCapacityUnits":5,"WriteCapacityUnits":5}}]' \
  --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
  --region ap-northeast-1

# Create S3 bucket (private)
aws s3 mb s3://streamlit-chatbot-dev-images --region ap-northeast-1
```

3) Environment variables (as needed)
```bash
export AWS_REGION=ap-northeast-1
export DYNAMODB_TABLE_NAME=ChatbotData
export S3_BUCKET_NAME=streamlit-chatbot-dev-images
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
```

4) Run
```bash
streamlit run main.py
```

### Access Control (Optional Email Allowlist)
- If you set `ALLOWED_EMAILS` env var (comma-separated), the app will verify `st.user.email` after login and immediately block + log out users not listed (exact match).
- If `ALLOWED_EMAILS` is missing or empty, this check is skipped (control relies solely on Google OAuth settings).
- Implementation location: right after the "logged-in" branch in the sidebar of `main.py`.

Example:
```bash
export ALLOWED_EMAILS="your-test-user@example.com,another@example.com"
```

Notes:
- Google auth with `st.login()`/`st.user`/`st.logout()` is well explained in this article (Japanese): [Zenn: Streamlit Google Login](https://zenn.dev/datum_studio/articles/c964f9e38379f4)
- Browser permissions are required for image upload and speech recognition.

---

## Secrets and Env Vars (Production)

In production (App Runner), manage all settings via **environment variables**. Configure them in the App Runner service settings or reference from AWS Secrets Manager.

Required environment variables:
- `AWS_REGION`: AWS region (e.g., ap-northeast-1)
- `DYNAMODB_TABLE_NAME`: DynamoDB table name
- `S3_BUCKET_NAME`: S3 bucket name
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`: AI model API keys
- `ALLOWED_EMAILS`: Comma-separated allowed email addresses (optional)

---

## AWS Setup

### 1. Create DynamoDB Table
See `quickstart.md`.

### 2. Create S3 Bucket
See `quickstart.md`.

### 3. IAM Role
Grant the App Runner instance role at least:
- `dynamodb:GetItem`, `dynamodb:PutItem`, `dynamodb:UpdateItem`, `dynamodb:DeleteItem`, `dynamodb:Query` (DynamoDB)
- `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`, `s3:ListBucket` (S3)

### 4. Deploy to App Runner
See "Production Deploy (App Runner)" section in `quickstart.md`.

---

## Data Model (Persistence)
- DynamoDB
  - pk=`CONV#<conversationId>`, sk=`METADATA`: `user_id`, `title`, `total_tokens`, `is_deleted`, `created_at`, `updated_at`
  - pk=`CONV#<conversationId>`, sk=`<timestamp>#<messageId>`: `role`, `content(json)`, `reasoning`, `created_at`
- S3
  - Saved as `images/conv{conversationId}_msg{messageId}_{index}.{ext}`
  - Replace data URI with S3 path at write time; restore data URI at read time

---

## Docker Run
The Dockerfile uses Poetry to resolve dependencies and sets `PORT=8080` for App Runner.
```bash
docker build -t streamlit-chatbot:local .
docker run -p 8080:8080 \
  -e AWS_REGION=ap-northeast-1 \
  -e DYNAMODB_TABLE_NAME=ChatbotData \
  -e S3_BUCKET_NAME=streamlit-chatbot-dev-images \
  -e OPENAI_API_KEY=... -e ANTHROPIC_API_KEY=... -e GOOGLE_API_KEY=... \
  streamlit-chatbot:local
```

---
