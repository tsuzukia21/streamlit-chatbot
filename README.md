## Streamlit Chatbot
[English](./README_EN.md)

マルチモーダル入力（テキスト/画像/音声）に対応した Streamlit 製チャットボットです。会話履歴は Firestore、画像は Cloud Storage に永続化します。モデル切替（Anthropic, Google, OpenAI）や思考可視化（thinking/chain-of-thoughtの表示）に対応しています。

### 主な機能
- **モデル切替**: Claude Opus 4.5 / Gemini 3.0 Pro / GPT‑5.1（LangChain ラッパー）
- **ツール付与**: 各モデルにネット検索ツール付与。最新の話題にも対応
- **マルチモーダル入力**: 画像アップロード、音声認識（Whisper）
- **会話管理**: 新規作成、タイトル自動生成、編集（過去メッセージから再分岐）
- **永続化**: Firestore（テキスト）、Cloud Storage（画像）
- **思考可視化**: reasoning/thinking 表示
- **認証**: `st.login()` を用いた Google ログイン（`secrets.toml` を Secret Manager でマウント）＋ `allowed_emails` によるメールホワイトリスト（任意）

---

## ファイル構成

```
streamlit-chatbot/
  main.py                     # Streamlit アプリ本体
  core/
    config.py                 # アプリ設定（システムプロンプト、トークン制限等）
    MODEL_CONFIG.py           # モデル定義/LLMファクトリ
    llm_handler.py            # LangChain チェーン構築とストリーミング
    conversation.py           # 会話の新規/読込/削除、タイトル生成
    database.py               # Firestore/Cloud Storage 永続化
    ui_components.py          # メッセージ表示/編集、思考の折りたたみ等
  requirements.txt            # pip 用依存
  pyproject.toml, poetry.lock # Poetry 用依存
  Dockerfile                  # Cloud Run 用コンテナ定義
  .streamlit/
    secrets.toml              # 認証/設定（本番は Secret Manager からマウント）
  README.md                   # 本ファイル
```

---

## 動作要件
- Python
- GCP: Firestore（ネイティブモード）、Cloud Storage バケット
- API キー（必要に応じて）
  - OpenAI: `OPENAI_API_KEY`
  - Anthropic: `ANTHROPIC_API_KEY`
  - Gemini: `GOOGLE_API_KEY`

---

## セットアップ（ローカル）

1) 依存インストール（pip または Poetry）
```bash
# pip
pip install -r requirements.txt

# または Poetry
poetry install
```

2) `.streamlit/secrets.toml` を作成（ローカル実行時）
```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "[強力なランダム文字列]"
client_id = "[Google OAuth クライアントID]"
client_secret = "[Google OAuth クライアントシークレット]"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
allowed_emails = ["your-test-user@example.com"] # 任意（メールホワイトリスト）。完全一致で判定
```

3) 環境変数（必要に応じて）
```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...

# ローカルでサービスアカウント鍵を使う場合のみ
export GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/service-account.json
```

4) 実行
```bash
streamlit run main.py
```

### アクセス制御（メールホワイトリスト、任意）
- `secrets.toml` の `[auth].allowed_emails` にメールアドレスの配列を設定すると、ログイン後に `st.user.email` と完全一致で照合し、許可されていないユーザーは即時ブロック＋ログアウトします。
- `allowed_emails` が未設定または空配列の場合は、このチェックはスキップされます（Google 側の OAuth 設定のみで制御）。
- 実装箇所: `main.py` のサイドバー内、ログイン済み分岐直後。

例（`secrets.toml`）:
```toml
[auth]
...
allowed_emails = ["your-test-user@example.com", "another@example.com"]
```

補足:
- `st.login()`/`st.user`/`st.logout()` を用いた Google 認証の概要や `secrets.toml` 設定は、この記事がとても分かりやすいです。[【st.login】GoogleアカウントでログインできるStreamlitアプリの開発方法と仕組みをわかりやすく解説](https://zenn.dev/datum_studio/articles/c964f9e38379f4)）。
- 画像アップロードや音声認識にはブラウザ権限が必要です。

---

## Secrets / 環境変数の取り扱い（本番）

本番（Cloud Run）では `.streamlit/secrets.toml` を直接コミットせず、**Secret Manager に登録したシークレットをファイルとしてコンテナ内 `/app/.streamlit/secrets.toml` にマウント**します。

### 手順（例）
1) Secret Manager に登録
```bash
gcloud secrets create streamlit-secrets --replication-policy=automatic
gcloud secrets versions add streamlit-secrets --data-file=.streamlit/secrets.toml
```

2) Cloud Run でファイルとしてマウント
- Cloud Console の Cloud Run > サービス > 編集 > セキュリティ > シークレット から、
  `streamlit-secrets` を「ファイルとしてマウント」に設定し、マウント先パスを `/app/.streamlit/secrets.toml` に指定
- これによりアプリはローカルと同じパスで `secrets.toml` を参照します

3) 追加の環境変数（必要に応じて）
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` も Secret Manager で管理し、Cloud Run 環境変数として参照
- LangSmithのセットアップもお勧めします

---

## GCP 設定

### 1. プロジェクト準備
- Firestore（ネイティブモード）を有効化（リージョンは Cloud Run/Storage と合わせる）
- Cloud Storage バケット作成

### 2. サービスアカウントと権限
Cloud Run 実行サービスアカウントに最低限以下のロールを付与:
- `roles/datastore.user`（Firestore）
- `roles/storage.objectAdmin`（Cloud Storage）
- 推奨: `roles/logging.logWriter`

### 3. デプロイ（Cloud Run）
コンテナビルドとデプロイ例:
```bash
# Artifact Registry へビルド & プッシュ
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/REPO/streamlit-chatbot:latest

# デプロイ（シークレットのファイルマウントはコンソール/マニフェストで設定）
gcloud run deploy streamlit-chatbot \
  --image REGION-docker.pkg.dev/PROJECT_ID/REPO/streamlit-chatbot:latest \
  --platform managed \
  --region REGION \
  --service-account YOUR_SA@PROJECT_ID.iam.gserviceaccount.com \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=...,ANTHROPIC_API_KEY=...,GOOGLE_API_KEY=...
```

補足:
- 認証をパブリックにしたくない場合は `--no-allow-unauthenticated` とし、IAP などで保護
- 画像が表示されない場合は `GCS_BUCKET_NAME` の設定/バケット権限/CORS を確認

---

## データモデル（永続化）
- Firestore コレクション
  - `conversations/{conversationId}`: `user_id`, `title`, `total_tokens`, `is_deleted`, `created_at`, `updated_at`
  - `conversations/{conversationId}/messages/{messageId}`: `role`, `content(json)`, `reasoning`, `created_at`
- Cloud Storage
  - `images/conv{conversationId}_msg{messageId}_{index}.{ext}` に保存
  - 保存時に data URI を GCS パスへ置換、読込時に data URI に復元

---

## Docker 実行
Dockerfile は Poetry を用いて依存を解決し、`PORT=8080` を Cloud Run 既定に合わせています。
```bash
docker build -t streamlit-chatbot:local .
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=... -e ANTHROPIC_API_KEY=... -e GOOGLE_API_KEY=... \
  streamlit-chatbot:local
```

---
