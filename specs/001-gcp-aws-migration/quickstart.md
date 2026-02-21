# Quickstart: GCPからAWSへのインフラ移行

**Date**: 2026-02-21

## 前提条件

- AWSアカウント（東京リージョン: ap-northeast-1）
- AWS CLI v2 インストール済み・設定済み (`aws configure`)
- Docker インストール済み
- Python 3.11+

## ローカル開発環境セットアップ

### 1. AWS認証情報の設定

```bash
# AWS CLI の設定
aws configure
# AWS_ACCESS_KEY_ID: <your-key>
# AWS_SECRET_ACCESS_KEY: <your-secret>
# Default region: ap-northeast-1
# Default output format: json
```

### 2. DynamoDB テーブル作成

```bash
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
  --global-secondary-indexes \
    '[{
      "IndexName": "UserConversationsIndex",
      "KeySchema": [
        {"AttributeName": "user_id", "KeyType": "HASH"},
        {"AttributeName": "updated_at", "KeyType": "RANGE"}
      ],
      "Projection": {"ProjectionType": "ALL"},
      "ProvisionedThroughput": {"ReadCapacityUnits": 5, "WriteCapacityUnits": 5}
    }]' \
  --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
  --region ap-northeast-1
```

### 3. S3 バケット作成

```bash
aws s3 mb s3://streamlit-chatbot-dev-images --region ap-northeast-1
aws s3api put-public-access-block \
  --bucket streamlit-chatbot-dev-images \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

### 4. 環境変数の設定

```bash
export AWS_REGION=ap-northeast-1
export DYNAMODB_TABLE_NAME=ChatbotData
export S3_BUCKET_NAME=streamlit-chatbot-dev-images
export OPENAI_API_KEY=<your-key>
export ANTHROPIC_API_KEY=<your-key>
export GOOGLE_API_KEY=<your-key>
```

### 5. 依存関係のインストール

```bash
poetry install
```

### 6. アプリケーションの起動

```bash
streamlit run main.py
```

## 本番デプロイ (App Runner)

### 1. ECR リポジトリ作成

```bash
aws ecr create-repository \
  --repository-name streamlit-chatbot \
  --region ap-northeast-1
```

### 2. Docker イメージのビルド・プッシュ

```bash
# ECR ログイン
aws ecr get-login-password --region ap-northeast-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.ap-northeast-1.amazonaws.com

# ビルド・タグ・プッシュ
docker build -t streamlit-chatbot .
docker tag streamlit-chatbot:latest <account-id>.dkr.ecr.ap-northeast-1.amazonaws.com/streamlit-chatbot:latest
docker push <account-id>.dkr.ecr.ap-northeast-1.amazonaws.com/streamlit-chatbot:latest
```

### 3. IAM ロール作成

**ECRアクセスロール** と **インスタンスロール** (DynamoDB + S3) を作成。
詳細は `research.md` の「App Runner デプロイ」セクション参照。

### 4. App Runner サービス作成

```bash
aws apprunner create-service \
  --service-name streamlit-chatbot \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "<account-id>.dkr.ecr.ap-northeast-1.amazonaws.com/streamlit-chatbot:latest",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": {
        "Port": "8080",
        "RuntimeEnvironmentVariables": {
          "AWS_REGION": "ap-northeast-1",
          "DYNAMODB_TABLE_NAME": "ChatbotData",
          "S3_BUCKET_NAME": "streamlit-chatbot-prod-images"
        }
      }
    },
    "AutoDeploymentsEnabled": true,
    "AuthenticationConfiguration": {
      "AccessRoleArn": "arn:aws:iam::<account-id>:role/AppRunnerECRAccessRole"
    }
  }' \
  --instance-configuration '{
    "Cpu": "1024",
    "Memory": "2048",
    "InstanceRoleArn": "arn:aws:iam::<account-id>:role/AppRunnerInstanceRole"
  }' \
  --health-check-configuration '{
    "Protocol": "HTTP",
    "Path": "/_stcore/health",
    "Interval": 10,
    "Timeout": 5,
    "HealthyThreshold": 1,
    "UnhealthyThreshold": 3
  }' \
  --region ap-northeast-1
```

## 検証チェックリスト

- [ ] DynamoDB テーブルが作成されている
- [ ] S3 バケットが作成されている（プライベートアクセス）
- [ ] ローカルで `streamlit run main.py` が起動する
- [ ] 新規会話の作成・メッセージ送受信が動作する
- [ ] ページリロード後に会話履歴が保持されている
- [ ] 画像付きメッセージの送信・表示が動作する
- [ ] App Runner にデプロイ後、ブラウザからアクセスできる
- [ ] Google OAuth ログイン/ログアウトが動作する
