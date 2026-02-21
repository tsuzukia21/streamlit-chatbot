# Research: GCPからAWSへのインフラ移行

**Date**: 2026-02-21

## 1. DynamoDB データモデリング（Firestoreサブコレクション構造の再現）

### Decision: シングルテーブルデザイン

1つのDynamoDBテーブル `ChatbotData` で会話とメッセージの両方を管理する。

**テーブル設計:**
- **PK (Partition Key)**: `pk` — `CONV#<conversationId>` 形式
- **SK (Sort Key)**: `sk` — 会話メタデータは `METADATA`、メッセージは `<timestamp>#<messageId>` 形式
- **GSI `UserConversationsIndex`**: `user_id` (PK) + `updated_at` (SK, Number/Unix ms)

**データ例:**
```
# 会話メタデータ
pk: CONV#abc123, sk: METADATA
→ user_id, title, total_tokens, is_deleted, created_at, updated_at

# メッセージ
pk: CONV#abc123, sk: 1708617600000#msg001
→ conversation_id, message_id, role, content, reasoning, created_at
```

### Rationale
- 会話とメッセージが同じパーティションに配置され、効率的なクエリが可能
- テーブル1つで管理コスト低減（RCU/WCU一元管理）
- AWSのベストプラクティス（シングルテーブルデザイン）に準拠
- このアプリの規模（個人〜少人数）では十分

### Alternatives Considered
- **2テーブル設計** (conversations + messages): テーブル間の整合性管理が複雑化、コスト増。大規模でない限り不要。

---

## 2. サーバータイムスタンプ

### Decision: クライアント側でUnixタイムスタンプ(ms)を生成

```python
import time
def get_timestamp():
    return int(time.time() * 1000)
```

### Rationale
- DynamoDBにはFirestoreの `SERVER_TIMESTAMP` に相当する機能がない
- Streamlitはサーバーサイド実行のため、Pythonの `time.time()` は実質サーバータイムスタンプ
- Lambda トリガーは過剰、アプリ規模には不要

### Alternatives Considered
- **DynamoDB Streams + Lambda**: オーバーエンジニアリング
- **Application middleware**: 複雑さが増す割にメリットなし

---

## 3. S3 画像ハンドリング

### Decision: boto3 の `put_object` / `get_object` を使用、直接アクセス（presigned URL不要）

**GCS → S3 APIマッピング:**

| GCS | S3 (boto3) |
|-----|-----------|
| `blob.upload_from_string(data, content_type=mime)` | `s3.put_object(Bucket=b, Key=k, Body=data, ContentType=mime)` |
| `blob.download_as_bytes()` | `s3.get_object(Bucket=b, Key=k)['Body'].read()` |
| `blob.exists()` | `s3.head_object()` (404 catch) |
| `bucket.list_blobs(prefix=p)` | `s3.list_objects_v2(Bucket=b, Prefix=p)['Contents']` |
| `blob.delete()` | `s3.delete_object(Bucket=b, Key=k)` |

### Rationale
- 画像はサーバーサイドでdata URIに変換するため、presigned URLは不要
- 同じパス形式 (`images/conv{id}_msg{id}_{index}.{ext}`) を維持
- `put_object` はバイナリデータの直接アップロードに最適

### Alternatives Considered
- **Presigned URLs**: ブラウザ直接アクセスなら有用だが、現アプリではdata URI変換のため不要
- **S3 Transfer**: 大ファイル向け、チャット画像には過剰

---

## 4. S3 バケット設定

### Decision: `streamlit-chatbot-{env}-images` 形式、プライベートアクセスのみ

**設定:**
- パブリックアクセス: 全ブロック
- バージョニング: 無効
- 暗号化: SSE-S3（デフォルト）
- 環境変数名: `S3_BUCKET_NAME`

### Rationale
- 画像はサーバーサイドのみアクセス、公開不要
- バージョニングはチャット画像には不要
- S3バケット名はグローバルユニーク要件あり、環境名を含めて区別

---

## 5. AWS認証（boto3）

### Decision: IAMロールベース認証（App Runner自動）

**App Runner上:**
- boto3はIAMインスタンスロールから自動的に認証情報を取得
- コード内でアクセスキー/シークレットキーの管理不要
- App Runnerサービスにインスタンスロールをアタッチするのみ

**ローカル開発:**
- `aws configure` でプロファイル設定、または環境変数 `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`

### Rationale
- IAMロールはAWSセキュリティベストプラクティス
- Cloud Runのサービスアカウント方式と概念的に同等
- 認証情報のローテーション管理が不要

---

## 6. App Runner デプロイ

### Decision: ECRベースのDockerイメージデプロイ

**Dockerfileの変更: ゼロ（そのまま互換）**
- ポート8080: App Runnerのデフォルトと一致
- `$PORT` 環境変数: App Runnerが自動設定
- ヘルスチェック: Streamlitの `/_stcore/health` エンドポイントを使用

**デプロイワークフロー:**
1. ECRリポジトリ作成
2. Docker イメージビルド・プッシュ
3. App Runnerサービス作成（CLI or コンソール）
4. 自動デプロイ設定（ECRプッシュ時）

**必要なIAMロール:**
- **アクセスロール**: ECRからイメージ取得用（`AWSAppRunnerServicePolicyForECRAccess`）
- **インスタンスロール**: DynamoDB/S3アクセス用（カスタムポリシー）
  - Trust policy: `tasks.apprunner.amazonaws.com`

### Rationale
- 既存Dockerfileがそのまま使える
- Cloud Run → App Runnerへの移行が最小限の変更で済む
- 自動スケーリング・自動デプロイ対応

### Alternatives Considered
- **ECS Fargate**: より高機能だが設定が複雑、この規模には過剰
- **Elastic Beanstalk**: 古い方式、App Runnerの方がシンプル

---

## 7. シークレット管理

### Decision: App Runner環境変数 + AWS Secrets Manager

**環境変数（非機密情報）:**
- `AWS_REGION`, `DYNAMODB_TABLE_NAME`, `S3_BUCKET_NAME`
- App Runnerのランタイム環境変数として設定

**Secrets Manager（機密情報）:**
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`
- OAuth設定（`client_id`, `client_secret`, `cookie_secret`）
- App Runnerからシークレット ARN を参照

**Streamlit secrets.toml の代替:**
- 現在GCP Secret Managerからマウントしている `.streamlit/secrets.toml`
- AWSでは環境変数 + Secrets Manager で代替
- `st.secrets` の代わりに `os.environ` を使用するよう `main.py` を更新

### Rationale
- Cloud Run のシークレットマウント方式から、App Runner の環境変数注入方式への移行
- Secrets Manager は IAMロールで自動認証
