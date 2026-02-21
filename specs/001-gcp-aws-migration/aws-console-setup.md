# AWS コンソール セットアップ手順

**対象リージョン**: アジアパシフィック（東京）ap-northeast-1

---

## 目次

1. [DynamoDB テーブル作成](#1-dynamodb-テーブル作成)
2. [S3 バケット作成](#2-s3-バケット作成)
3. [IAM ロール作成](#3-iam-ロール作成)
4. [ECR リポジトリ作成](#4-ecr-リポジトリ作成)
5. [App Runner サービス作成](#5-app-runner-サービス作成)
6. [トラブルシューティング](#6-トラブルシューティング)

---

## 1. DynamoDB テーブル作成

1. AWSコンソール → **DynamoDB** を開く
2. 右上のリージョンが **アジアパシフィック（東京）** になっていることを確認
3. 左メニュー「テーブル」→「**テーブルの作成**」をクリック

### テーブルの設定

| 項目 | 値 |
|------|-----|
| テーブル名 | `ChatbotData` |
| パーティションキー | `pk`（文字列） |
| ソートキー | `sk`（文字列） |

4. 「テーブル設定」は「**設定をカスタマイズ**」を選択

### キャパシティの設定

- キャパシティモード: **プロビジョニング済み**
- 読み込みキャパシティ: `5`
- 書き込みキャパシティ: `5`

### グローバルセカンダリインデックス（GSI）の追加

5. 「グローバルセカンダリインデックス」セクション →「**インデックスの作成**」をクリック

| 項目 | 値 |
|------|-----|
| インデックス名 | `UserConversationsIndex` |
| パーティションキー | `user_id`（文字列） |
| ソートキー | `updated_at`（数値） |
| 属性の射影 | すべて |
| 読み込みキャパシティ | `5` |
| 書き込みキャパシティ | `5` |

6. 「**テーブルの作成**」をクリック

---

## 2. S3 バケット作成

1. AWSコンソール → **S3** を開く
2. 「**バケットを作成**」をクリック

### 基本設定

| 項目 | 値 |
|------|-----|
| バケット名 | `streamlit-chatbot-prod-images`（全世界で一意な名前） |
| AWSリージョン | アジアパシフィック（東京） ap-northeast-1 |

### オブジェクト所有権

- 「**ACL 無効（推奨）**」を選択

### このバケットのパブリックアクセスをブロックする設定

- 「**パブリックアクセスをすべてブロック**」にチェック ✅（デフォルトのまま）

3. 残りはデフォルトのまま「**バケットを作成**」をクリック

---

## 3. IAM ロール作成

### 3-1. インスタンスロール（App Runner が DynamoDB・S3 にアクセスするため）

1. 引き続き IAM → 「**ロールを作成**」をクリック

#### 信頼されたエンティティの選択

- 信頼されたエンティティタイプ: **カスタム信頼ポリシー**
- 以下のJSONを貼り付け（ECRアクセスロールとは**プリンシパルが異なる**ので注意）:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "tasks.apprunner.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

2. 「次へ」をクリック

#### 許可ポリシーのアタッチ

3. ここでは既存のポリシーではなくカスタムポリシーを作成するため、「**ポリシーの作成**」をクリック（新しいタブが開く）

#### カスタムポリシーの作成

4. 「**JSON**」タブを選択して以下を貼り付け（`YOUR_ACCOUNT_ID` は自分のAWSアカウントIDに置換）:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:DeleteItem",
        "dynamodb:Query",
        "dynamodb:BatchWriteItem"
      ],
      "Resource": [
        "arn:aws:dynamodb:ap-northeast-1:YOUR_ACCOUNT_ID:table/ChatbotData",
        "arn:aws:dynamodb:ap-northeast-1:YOUR_ACCOUNT_ID:table/ChatbotData/index/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::streamlit-chatbot-prod-images",
        "arn:aws:s3:::streamlit-chatbot-prod-images/*"
      ]
    }
  ]
}
```

5. 「次へ」→ ポリシー名 `ChatbotAppPolicy` → 「**ポリシーを作成**」

6. ロール作成タブに戻り、画面を更新して `ChatbotAppPolicy` を検索してチェック ✅
7. 「次へ」をクリック

#### ロールの詳細

| 項目 | 値 |
|------|-----|
| ロール名 | `AppRunnerInstanceRole` |

8. 「**ロールを作成**」をクリック

> **AWSアカウントIDの確認方法**: コンソール右上のアカウント名をクリックすると表示される12桁の数字

---

### 3-2. GitHub Actions 用 OIDC 設定（GitHub Actions から ECR にプッシュするため）

#### IDプロバイダーの追加

1. IAM → 左メニュー「**IDプロバイダ**」→「**プロバイダを追加**」をクリック

| 項目 | 値 |
|------|-----|
| プロバイダのタイプ | **OpenID Connect** |
| プロバイダの URL | `https://token.actions.githubusercontent.com` |

2. URLを入力したら「**サムプリントを取得**」ボタンをクリック（URL入力欄の右横に表示される）
3. サムプリント取得後、「対象者」欄に `sts.amazonaws.com` を入力
4. 「**プロバイダを追加**」をクリック

#### GitHub Actions 用ロールの作成

3. IAM → 「**ロールを作成**」をクリック

- 信頼されたエンティティタイプ: **カスタム信頼ポリシー**
- 以下のJSONを貼り付け（`YOUR_ACCOUNT_ID` と `YOUR_GITHUB_USERNAME/REPO_NAME` を置換）:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::YOUR_ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:YOUR_GITHUB_USERNAME/REPO_NAME:*"
        }
      }
    }
  ]
}
```

4. 「次へ」をクリック

#### 許可ポリシーのアタッチ

5. 検索ボックスに `AmazonEC2ContainerRegistryPowerUser` と入力してチェック ✅
6. 「次へ」をクリック

#### ロールの詳細

| 項目 | 値 |
|------|-----|
| ロール名 | `GitHubActionsECRRole` |

7. 「**ロールを作成**」をクリック
8. 作成後、ロールの詳細画面で **ARN** をコピーしておく（GitHub Secrets に登録する）

---

## 4. ECR リポジトリ作成

1. AWSコンソール → **Elastic Container Registry (ECR)** を開く
2. リージョンが **東京** になっていることを確認
3. 「**リポジトリを作成**」をクリック

### リポジトリの設定

| 項目 | 値 |
|------|-----|
| 可視性設定 | **プライベート** |
| リポジトリ名 | `streamlit-chatbot` |

4. 残りはデフォルトのまま「**リポジトリを作成**」をクリック

### GitHub Actions によるビルド・プッシュの自動化

ECR へのイメージプッシュは GitHub Actions で自動化する。

#### GitHub リポジトリの Secrets 設定

5. GitHub リポジトリ → Settings → Secrets and variables → Actions
6. 以下の Repository secrets を追加:

| Secret 名 | 値 |
|------|-----|
| `AWS_ROLE_ARN` | Step 3-2 で作成した `GitHubActionsECRRole` の ARN |
| `AWS_ACCOUNT_ID` | AWSアカウントID（12桁） |

#### ワークフローファイルの作成

7. リポジトリに `.github/workflows/deploy.yml` を作成:

```yaml
name: Build and Deploy to ECR

on:
  push:
    branches: [main]

permissions:
  id-token: write
  contents: read

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: ap-northeast-1

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build, tag, and push image to ECR
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: streamlit-chatbot
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:latest .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
```

> **デプロイの流れ**: `main` ブランチに push → GitHub Actions が自動でビルド＆ECRにプッシュ → App Runner が自動デプロイ（デプロイトリガーを「自動」に設定している場合）

---

## 5. App Runner サービス作成

1. AWSコンソール → **App Runner** を開く
2. リージョンが **東京** になっていることを確認
3. 「**サービスの作成**」をクリック

### ソース設定

| 項目 | 値 |
|------|-----|
| ソース | **コンテナレジストリ** |
| プロバイダー | **Amazon ECR** |
| コンテナイメージのURI | ECRリポジトリの `streamlit-chatbot:latest` を選択 |
| デプロイトリガー | **自動** |
| ECRアクセスロール | 「**新しいサービスロールの作成**」を選択（自動で作成される） |

4. 「次へ」をクリック

### サービスの設定

| 項目 | 値 |
|------|-----|
| サービス名 | `streamlit-chatbot` |
| ポート | `8080` |

#### 環境変数の追加

「環境変数」セクションで以下を1つずつ追加:

| キー | 値 |
|------|-----|
| `AWS_REGION` | `ap-northeast-1` |
| `DYNAMODB_TABLE_NAME` | `ChatbotData` |
| `S3_BUCKET_NAME` | `streamlit-chatbot-prod-images` |
| `OPENAI_API_KEY` | `sk-...` |
| `ANTHROPIC_API_KEY` | `sk-ant-...` |
| `GOOGLE_API_KEY` | `AIza...` |
| `ALLOWED_EMAILS` | `your@example.com`（任意） |
| `GOOGLE_OAUTH_CLIENT_ID` | Google Cloud Console で作成した OAuth 2.0 クライアント ID |
| `GOOGLE_OAUTH_CLIENT_SECRET` | 同クライアントシークレット |
| `AUTH_REDIRECT_URI` | `https://xxxx.ap-northeast-1.awsapprunner.com/oauth2callback`（App Runner のデフォルトドメイン） |
| `AUTH_COOKIE_SECRET` | 任意のランダム文字列（省略時は自動生成） |

> **注意**: `AUTH_REDIRECT_URI` は App Runner デプロイ後に表示されるドメインを使う。また Google Cloud Console の OAuth クライアントの「承認済みのリダイレクト URI」にも同じ URL を追加すること。

#### インスタンスの設定

| 項目 | 値 |
|------|-----|
| vCPU | `1 vCPU` |
| メモリ | `2 GB` |
| インスタンスロール | `AppRunnerInstanceRole` |

5. 「次へ」をクリック

### ヘルスチェックの設定

| 項目 | 値 |
|------|-----|
| プロトコル | **HTTP** |
| パス | `/_stcore/health` |
| インターバル | `10` 秒 |
| タイムアウト | `5` 秒 |
| 正常しきい値 | `1` |
| 異常しきい値 | `3` |

6. 「次へ」→ 内容を確認 →「**作成してデプロイ**」をクリック

デプロイ完了後（数分）、表示されるデフォルトドメイン（`https://xxxx.ap-northeast-1.awsapprunner.com`）にアクセスして動作確認。

---

## 6. トラブルシューティング

### WebSocket エラーで画面が表示されない

App Runner 経由で Streamlit にアクセスした際、以下のようなエラーが発生する場合がある:

```
Client Error: WebSocket onerror
WebSocket connection to 'wss://xxxx.ap-northeast-1.awsapprunner.com/_stcore/stream' failed
```

**原因**: App Runner のリバースプロキシと Streamlit の WebSocket 設定（CORS、XSRF保護、圧縮）が競合している。

**対策**: `.streamlit/config.toml` を作成し、以下の設定を追加する:

```toml
[server]
enableCORS = false
enableXsrfProtection = false
enableWebsocketCompression = false

[browser]
gatherUsageStats = false
```

また、`Dockerfile` の起動コマンドにも同じフラグを追加する:

```dockerfile
CMD streamlit run main.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false --server.enableWebsocketCompression=false
```

---

## 作業順序まとめ

```
Step 1: DynamoDB テーブル作成
Step 2: S3 バケット作成
Step 3: IAM ロール作成（インスタンスロール → GitHub Actions OIDC）
Step 4: ECR リポジトリ作成 → GitHub Secrets 設定 → ワークフロー作成
Step 5: App Runner サービス作成
```
