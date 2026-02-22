# AWS コンソール セットアップ手順

**対象リージョン**: アジアパシフィック（東京）ap-northeast-1

---

## 目次

1. [DynamoDB テーブル作成](#1-dynamodb-テーブル作成)
2. [S3 バケット作成](#2-s3-バケット作成)
3. [IAM ロール作成](#3-iam-ロール作成)
4. [ECR リポジトリ作成](#4-ecr-リポジトリ作成)
5. [ECS Fargate + ALB でデプロイ](#5-ecs-fargate--alb-でデプロイ)
6. [CloudFront で HTTPS 対応](#6-cloudfront-で-https-対応)
7. [トラブルシューティング](#7-トラブルシューティング)

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

### 3-1. ECS タスクロール（アプリが DynamoDB・S3 にアクセスするため）

1. AWSコンソール → **IAM** → 左メニュー「ロール」→「**ロールを作成**」をクリック

#### 信頼されたエンティティの選択

- 信頼されたエンティティタイプ: **カスタム信頼ポリシー**
- 以下のJSONを貼り付け:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
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
| ロール名 | `ECSTaskRole` |

8. 「**ロールを作成**」をクリック

> **AWSアカウントIDの確認方法**: コンソール右上のアカウント名をクリックすると表示される12桁の数字

---

### 3-2. ECS タスク実行ロール（ECS エージェントが ECR からイメージ取得・ログ書き込みするため）

1. IAM → 「**ロールを作成**」をクリック

#### 信頼されたエンティティの選択

- 信頼されたエンティティタイプ: **AWSのサービス**
- ユースケース: **Elastic Container Service** → **Elastic Container Service Task**

2. 「次へ」をクリック

#### 許可ポリシーのアタッチ

3. 検索ボックスに `AmazonECSTaskExecutionRolePolicy` と入力してチェック ✅
4. 「次へ」をクリック

#### ロールの詳細

| 項目 | 値 |
|------|-----|
| ロール名 | `ecsTaskExecutionRole` |

5. 「**ロールを作成**」をクリック

---

### 3-3. GitHub Actions 用 OIDC 設定（GitHub Actions から ECR プッシュ＆ECS デプロイするため）

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

5. IAM → 「**ロールを作成**」をクリック

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

6. 「次へ」をクリック

#### 許可ポリシーのアタッチ

7. 検索ボックスに `AmazonEC2ContainerRegistryPowerUser` と入力してチェック ✅
8. 「次へ」をクリック

#### ロールの詳細

| 項目 | 値 |
|------|-----|
| ロール名 | `GitHubActionsECRRole` |

9. 「**ロールを作成**」をクリック
10. 作成後、ロールの詳細画面で **ARN** をコピーしておく（GitHub Secrets に登録する）

#### ECS デプロイ権限の追加

11. 作成した `GitHubActionsECRRole` の詳細画面 →「許可」タブ →「**許可を追加**」→「**インラインポリシーを作成**」
12. 「**JSON**」タブを選択して以下を貼り付け（`YOUR_ACCOUNT_ID` を置換）:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecs:UpdateService",
        "ecs:DescribeServices"
      ],
      "Resource": "arn:aws:ecs:ap-northeast-1:YOUR_ACCOUNT_ID:service/streamlit-chatbot-cluster/streamlit-chatbot-service"
    }
  ]
}
```

13. ポリシー名 `ECSDeployPolicy` → 「**ポリシーの作成**」

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

### GitHub Actions によるビルド・デプロイの自動化

ECR へのイメージプッシュと ECS へのデプロイは GitHub Actions で自動化する。

#### GitHub リポジトリの Secrets 設定

5. GitHub リポジトリ → Settings → Secrets and variables → Actions
6. 以下の Repository secrets を追加:

| Secret 名 | 値 |
|------|-----|
| `AWS_ROLE_ARN` | Step 3-3 で作成した `GitHubActionsECRRole` の ARN |

#### ワークフローファイルの作成

7. リポジトリに `.github/workflows/deploy.yml` を作成:

```yaml
name: Build and Deploy to ECS

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
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG -t $ECR_REGISTRY/$ECR_REPOSITORY:latest .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster streamlit-chatbot-cluster \
            --service streamlit-chatbot-service \
            --force-new-deployment
```

> **デプロイの流れ**: `main` ブランチに push → GitHub Actions が自動でビルド＆ECRにプッシュ → ECS サービスを強制更新 → Fargate が新しいイメージで再デプロイ

---

## 5. ECS Fargate + ALB でデプロイ

> **なぜ App Runner ではなく ECS Fargate を使うのか**: App Runner は WebSocket をサポートしていない。Streamlit は WebSocket 必須のため、WebSocket をネイティブサポートする ALB + ECS Fargate を使用する。

### 5-1. セキュリティグループの作成

AWSコンソール → **VPC** → 左メニュー「セキュリティグループ」

#### ALB 用セキュリティグループ

1. 「**セキュリティグループを作成**」をクリック

| 項目 | 値 |
|------|-----|
| セキュリティグループ名 | `streamlit-alb-sg` |
| 説明 | `ALB for Streamlit chatbot` |
| VPC | デフォルト VPC |

2. インバウンドルール:

| タイプ | ポート | ソース |
|--------|--------|--------|
| HTTP | 80 | **CloudFront マネージドプレフィックスリスト**（`com.amazonaws.global.cloudfront.origin-facing`） |

> **ポイント**: ソースに `0.0.0.0/0` ではなく CloudFront のマネージドプレフィックスリストを指定することで、ALB への直接アクセスをブロックし、CloudFront 経由のアクセスのみ許可する。ソースの入力欄で「プレフィックスリスト」を選択し、`com.amazonaws.global.cloudfront.origin-facing` を選ぶ。

3. 「**セキュリティグループを作成**」をクリック

#### ECS タスク用セキュリティグループ

4. 「**セキュリティグループを作成**」をクリック

| 項目 | 値 |
|------|-----|
| セキュリティグループ名 | `streamlit-ecs-sg` |
| 説明 | `ECS tasks for Streamlit chatbot` |
| VPC | デフォルト VPC |

5. インバウンドルール:

| タイプ | ポート | ソース |
|--------|--------|--------|
| カスタム TCP | 8080 | `streamlit-alb-sg`（ALB のセキュリティグループを選択） |

6. 「**セキュリティグループを作成**」をクリック

---

### 5-2. ターゲットグループの作成

AWSコンソール → **EC2** → 左メニュー「ターゲットグループ」

1. 「**ターゲットグループの作成**」をクリック

| 項目 | 値 |
|------|-----|
| ターゲットタイプ | **IP アドレス** |
| ターゲットグループ名 | `streamlit-chatbot-tg` |
| プロトコル | HTTP |
| ポート | `8080` |
| VPC | デフォルト VPC |
| プロトコルバージョン | HTTP1 |

2. ヘルスチェック設定:

| 項目 | 値 |
|------|-----|
| ヘルスチェックパス | `/_stcore/health` |
| 正常しきい値 | `2` |
| 異常しきい値 | `3` |
| タイムアウト | `5` 秒 |
| インターバル | `30` 秒 |
| 成功コード | `200` |

3. ターゲットの登録は**スキップ**（ECS が自動で登録する）
4. 「**ターゲットグループの作成**」をクリック

---

### 5-3. ALB（Application Load Balancer）の作成

AWSコンソール → **EC2** → 左メニュー「ロードバランサー」

1. 「**ロードバランサーの作成**」→ **Application Load Balancer** を選択

| 項目 | 値 |
|------|-----|
| ロードバランサー名 | `streamlit-chatbot-alb` |
| スキーム | **インターネット向け** |
| IPアドレスタイプ | IPv4 |

2. ネットワークマッピング: デフォルト VPC を選択し、**2つ以上のAZ**（例: ap-northeast-1a, ap-northeast-1c）のサブネットを選択

3. セキュリティグループ: `streamlit-alb-sg` を選択

4. リスナー:

| プロトコル | ポート | デフォルトアクション |
|----------|------|----------------|
| HTTP | 80 | `streamlit-chatbot-tg` に転送 |

5. 「**ロードバランサーの作成**」をクリック

> **WebSocket について**: ALB は HTTP/1.1 Upgrade リクエストを透過的に転送するため、WebSocket は追加設定なしで動作する。

---

### 5-4. ECS クラスターの作成

AWSコンソール → **ECS** → 左メニュー「クラスター」

1. 「**クラスターの作成**」をクリック

| 項目 | 値 |
|------|-----|
| クラスター名 | `streamlit-chatbot-cluster` |
| インフラストラクチャ | **AWS Fargate（サーバーレス）** のみ |

2. 「**作成**」をクリック

---

### 5-5. タスク定義の作成

AWSコンソール → **ECS** → 左メニュー「タスク定義」

1. 「**新しいタスク定義の作成**」をクリック

#### タスク定義の設定

| 項目 | 値 |
|------|-----|
| タスク定義ファミリー | `streamlit-chatbot` |
| 起動タイプ | AWS Fargate |
| OS/アーキテクチャ | Linux/X86_64 |
| タスクサイズ - CPU | `0.5 vCPU` |
| タスクサイズ - メモリ | `1 GB` |
| タスクロール | `ECSTaskRole` |
| タスク実行ロール | `ecsTaskExecutionRole` |

#### コンテナの定義

| 項目 | 値 |
|------|-----|
| コンテナ名 | `streamlit-chatbot` |
| イメージ URI | `YOUR_ACCOUNT_ID.dkr.ecr.ap-northeast-1.amazonaws.com/streamlit-chatbot:latest` |
| 必須 | はい |
| ポートマッピング | コンテナポート: `8080`、プロトコル: TCP |

#### 環境変数の追加

| キー | 値 |
|------|-----|
| `AWS_REGION` | `ap-northeast-1` |
| `DYNAMODB_TABLE_NAME` | `ChatbotData` |
| `S3_BUCKET_NAME` | `streamlit-chatbot-prod-images` |
| `OPENAI_API_KEY` | `sk-...` |
| `ANTHROPIC_API_KEY` | `sk-ant-...` |
| `GOOGLE_API_KEY` | `AIza...` |
| `ALLOWED_EMAILS` | `your@example.com`（任意） |
| `GOOGLE_OAUTH_CLIENT_ID` | Google Cloud Console の OAuth クライアント ID |
| `GOOGLE_OAUTH_CLIENT_SECRET` | 同クライアントシークレット |
| `AUTH_REDIRECT_URI` | `https://CLOUDFRONT_DOMAIN/oauth2callback`（CloudFront 作成後に更新する） |
| `AUTH_COOKIE_SECRET` | 任意のランダム文字列（省略時は自動生成） |

#### ログの設定

- ログ収集: デフォルトのまま（awslogs、CloudWatch へ自動送信）

2. 「**作成**」をクリック

---

### 5-6. ECS サービスの作成

AWSコンソール → **ECS** → クラスター `streamlit-chatbot-cluster` → 「サービス」タブ

1. 「**作成**」をクリック

#### サービスの設定

| 項目 | 値 |
|------|-----|
| 起動タイプ | **Fargate** |
| タスク定義ファミリー | `streamlit-chatbot` |
| リビジョン | LATEST |
| サービス名 | `streamlit-chatbot-service` |
| タスクの必要数 | `1` |

#### ネットワーキング

| 項目 | 値 |
|------|-----|
| VPC | デフォルト VPC |
| サブネット | ALB と同じサブネット |
| セキュリティグループ | `streamlit-ecs-sg` |
| パブリック IP | **オン**（デフォルト VPC で ECR からイメージを取得するために必要） |

#### ロードバランシング

| 項目 | 値 |
|------|-----|
| ロードバランサーの種類 | **Application Load Balancer** |
| ロードバランサー | `streamlit-chatbot-alb` |
| ロードバランス用コンテナ | `streamlit-chatbot:8080` |
| ターゲットグループ | 既存: `streamlit-chatbot-tg` |

2. 「**作成**」をクリック

---

### 5-7. 動作確認（HTTP）

ECS サービス作成後、ALB の DNS 名で HTTP アクセスして ECS が正常に動作しているか確認する。

1. EC2 コンソール → ロードバランサー → `streamlit-chatbot-alb` の DNS 名をコピー
2. ブラウザで `http://ALB_DNS_NAME` にアクセス
3. Streamlit の画面が表示されることを確認（この時点では HTTP）

> この後 CloudFront を設定して HTTPS 対応する。

---

## 6. CloudFront で HTTPS 対応

> **なぜ CloudFront を使うのか**: CloudFront はカスタムドメインなしで自動的に HTTPS URL（`https://xxxxxxxx.cloudfront.net`）と SSL 証明書を提供する。ALB 単体では SSL 証明書の設定にカスタムドメインが必要だが、CloudFront なら不要。

構成: `ユーザー → CloudFront (HTTPS) → ALB (HTTP) → ECS`

### 6-1. CloudFront ディストリビューションの作成

AWSコンソール → **CloudFront** を開く

1. 「**ディストリビューションを作成**」をクリック

#### 基本設定

| 項目 | 値 |
|------|-----|
| Description | `Streamlit Chatbot HTTPS Frontend` |

#### オリジンの設定

| 項目 | 値 |
|------|-----|
| オリジンドメイン | ALB の DNS 名（`streamlit-chatbot-alb-xxxxxx.ap-northeast-1.elb.amazonaws.com`） |
| プロトコル | **HTTP のみ** |
| HTTP ポート | `80` |

#### デフォルトのキャッシュビヘイビアの設定

| 項目 | 値 |
|------|-----|
| ビューワープロトコルポリシー | **Redirect HTTP to HTTPS** |
| 許可された HTTP メソッド | **GET, HEAD, OPTIONS, PUT, POST, PATCH, DELETE** |
| キャッシュポリシー | **CachingDisabled** |
| オリジンリクエストポリシー | **AllViewer** |

> **ポイント**: Streamlit は動的コンテンツ＆WebSocket のため、キャッシュは無効にし、すべてのヘッダー（`Upgrade`、`Connection` 含む）をオリジンに転送する必要がある。

#### Web Application Firewall (WAF)

- 「**セキュリティ保護を有効にしない**」を選択（コスト削減のため）

#### 設定

| 項目 | 値 |
|------|-----|
| 料金クラス | **北米、欧州、アジア、中東、アフリカを使用**（または「すべてのエッジロケーションを使用」） |
| デフォルトルートオブジェクト | （空欄のまま） |

2. 「**ディストリビューションを作成**」をクリック
3. 作成後、ディストリビューションの **ドメイン名** をコピー（例: `d1234567890.cloudfront.net`）

> デプロイ完了まで数分かかる。ステータスが「有効」になるまで待つ。

---

### 6-2. AUTH_REDIRECT_URI の更新

1. ECS → タスク定義 → `streamlit-chatbot` → 「**新しいリビジョンの作成**」
2. 環境変数 `AUTH_REDIRECT_URI` を `https://CLOUDFRONT_DOMAIN/oauth2callback` に更新（例: `https://d1234567890.cloudfront.net/oauth2callback`）
3. 新しいリビジョンを作成
4. ECS → クラスター → サービス → 「**サービスを更新**」→ 新しいリビジョンを選択 →「**更新**」
5. **Google Cloud Console** の OAuth クライアント設定で「承認済みのリダイレクト URI」に `https://CLOUDFRONT_DOMAIN/oauth2callback` を追加

---

### 6-3. 動作確認（HTTPS）

デプロイ完了後（数分）、CloudFront のドメイン名（`https://CLOUDFRONT_DOMAIN`）にブラウザでアクセスして動作確認。

確認項目:
- HTTPS でアクセスできること（ブラウザのアドレスバーに鍵アイコンが表示される）
- Streamlit の画面が表示されること（WebSocket エラーが出ないこと）
- Google OAuth ログインが動作すること
- チャットの送受信ができること（DynamoDB）
- 画像アップロードができること（S3）

> **注意**: ALB のセキュリティグループで CloudFront マネージドプレフィックスリストのみ許可しているため、ALB の DNS 名に直接アクセスしてもタイムアウトになる。必ず CloudFront の URL（HTTPS）を使用すること。

---

## 7. トラブルシューティング

### ECS タスクが起動しない

1. ECS → クラスター → サービス →「タスク」タブで停止したタスクを確認
2. タスクをクリック →「ログ」タブで CloudWatch ログを確認
3. よくある原因:
   - ECR イメージが見つからない → イメージ URI を確認
   - タスク実行ロールの権限不足 → `ecsTaskExecutionRole` に `AmazonECSTaskExecutionRolePolicy` がアタッチされているか確認
   - ヘルスチェック失敗 → `/_stcore/health` が 200 を返しているか確認

### ヘルスチェックが失敗する

- ターゲットグループのヘルスチェックパスが `/_stcore/health` になっているか確認
- ポートが `8080` になっているか確認
- セキュリティグループ `streamlit-ecs-sg` で ALB からの 8080 ポートが許可されているか確認

### ログの確認方法

- **CloudWatch** コンソール → ロググループ → `/ecs/streamlit-chatbot` でアプリケーションログを確認

---

## 作業順序まとめ

```
Step 1: DynamoDB テーブル作成
Step 2: S3 バケット作成
Step 3: IAM ロール作成（ECS タスクロール → タスク実行ロール → GitHub Actions OIDC）
Step 4: ECR リポジトリ作成 → GitHub Secrets 設定 → ワークフロー作成
Step 5: ECS Fargate + ALB でデプロイ
  5-1: セキュリティグループ作成
  5-2: ターゲットグループ作成
  5-3: ALB 作成
  5-4: ECS クラスター作成
  5-5: タスク定義作成
  5-6: ECS サービス作成
  5-7: 動作確認（HTTP）
Step 6: CloudFront で HTTPS 対応
  6-1: CloudFront ディストリビューション作成
  6-2: AUTH_REDIRECT_URI 更新（HTTPS）
  6-3: 動作確認（HTTPS）
```
