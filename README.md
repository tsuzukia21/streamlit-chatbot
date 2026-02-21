## Streamlit Chatbot
[English](./README_EN.md)

マルチモーダル入力（テキスト/画像/音声）に対応した Streamlit 製チャットボットです。会話履歴は DynamoDB、画像は S3 に永続化します。モデル切替（Anthropic, Google, OpenAI）や思考可視化（thinking/chain-of-thoughtの表示）に対応しています。

### 主な機能
- **モデル切替**: Claude Opus 4.5 / Gemini 3.0 Pro / GPT‑5.1（LangChain ラッパー）
- **ツール付与**: 各モデルにネット検索ツール付与。最新の話題にも対応
- **マルチモーダル入力**: 画像アップロード、音声認識（Whisper）
- **会話管理**: 新規作成、タイトル自動生成、編集（過去メッセージから再分岐）
- **永続化**: DynamoDB（テキスト）、S3（画像）
- **思考可視化**: reasoning/thinking 表示
- **認証**: `st.login()` を用いた Google ログイン＋ `ALLOWED_EMAILS` 環境変数によるメールホワイトリスト（任意）

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
    database.py               # DynamoDB/S3 永続化
    ui_components.py          # メッセージ表示/編集、思考の折りたたみ等
  requirements.txt            # pip 用依存
  pyproject.toml, poetry.lock # Poetry 用依存
  Dockerfile                  # App Runner 用コンテナ定義
  README.md                   # 本ファイル
```

---

## 動作要件
- Python 3.11+
- AWS: DynamoDB テーブル、S3 バケット（ap-northeast-1 推奨）
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

2) AWS リソースのセットアップ
```bash
# DynamoDB テーブル作成
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

# S3 バケット作成
aws s3 mb s3://streamlit-chatbot-dev-images --region ap-northeast-1
```

3) 環境変数（必要に応じて）
```bash
export AWS_REGION=ap-northeast-1
export DYNAMODB_TABLE_NAME=ChatbotData
export S3_BUCKET_NAME=streamlit-chatbot-dev-images
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
```

4) 実行
```bash
streamlit run main.py
```

### アクセス制御（メールホワイトリスト、任意）
- 環境変数 `ALLOWED_EMAILS` にカンマ区切りでメールアドレスを設定すると、ログイン後に `st.user.email` と完全一致で照合し、許可されていないユーザーは即時ブロック＋ログアウトします。
- `ALLOWED_EMAILS` が未設定または空の場合は、このチェックはスキップされます。
- 実装箇所: `main.py` のサイドバー内、ログイン済み分岐直後。

例:
```bash
export ALLOWED_EMAILS="your-test-user@example.com,another@example.com"
```

補足:
- `st.login()`/`st.user`/`st.logout()` を用いた Google 認証の概要は、この記事が分かりやすいです。[【st.login】GoogleアカウントでログインできるStreamlitアプリの開発方法と仕組みをわかりやすく解説](https://zenn.dev/datum_studio/articles/c964f9e38379f4)
- 画像アップロードや音声認識にはブラウザ権限が必要です。

---

## Secrets / 環境変数の取り扱い（本番）

本番（App Runner）では全ての設定を**環境変数**で管理します。App Runner サービスの環境変数設定か、AWS Secrets Manager から参照してください。

必要な環境変数:
- `AWS_REGION`: AWSリージョン（例: ap-northeast-1）
- `DYNAMODB_TABLE_NAME`: DynamoDBテーブル名
- `S3_BUCKET_NAME`: S3バケット名
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`: AIモデルのAPIキー
- `ALLOWED_EMAILS`: 許可するメールアドレス（カンマ区切り、任意）

---

## AWS 設定

### 1. DynamoDB テーブル作成
`quickstart.md` 参照。

### 2. S3 バケット作成
`quickstart.md` 参照。

### 3. IAM ロール
App Runner インスタンスロールに最低限以下を付与:
- `dynamodb:GetItem`, `dynamodb:PutItem`, `dynamodb:UpdateItem`, `dynamodb:DeleteItem`, `dynamodb:Query`（DynamoDB）
- `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`, `s3:ListBucket`（S3）

### 4. App Runner へのデプロイ
`quickstart.md` の「本番デプロイ (App Runner)」セクション参照。

---

## データモデル（永続化）
- DynamoDB
  - pk=`CONV#<conversationId>`, sk=`METADATA`: `user_id`, `title`, `total_tokens`, `is_deleted`, `created_at`, `updated_at`
  - pk=`CONV#<conversationId>`, sk=`<timestamp>#<messageId>`: `role`, `content(json)`, `reasoning`, `created_at`
- S3
  - `images/conv{conversationId}_msg{messageId}_{index}.{ext}` に保存
  - 保存時に data URI を S3 パスへ置換、読込時に data URI に復元

---

## Docker 実行
Dockerfile は Poetry を用いて依存を解決し、`PORT=8080` を App Runner 規定に合わせています。
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
