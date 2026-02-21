# AWS コンソール セットアップ手順

**対象リージョン**: アジアパシフィック（東京）ap-northeast-1

---

## 目次

1. [DynamoDB テーブル作成](#1-dynamodb-テーブル作成)
2. [S3 バケット作成](#2-s3-バケット作成)
3. [IAM ロール作成](#3-iam-ロール作成)
4. [ECR リポジトリ作成](#4-ecr-リポジトリ作成)
5. [App Runner サービス作成](#5-app-runner-サービス作成)

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

### 3-1. ECRアクセスロール（App Runner が ECR からイメージを取得するため）

1. AWSコンソール → **IAM** を開く
2. 左メニュー「ロール」→「**ロールを作成**」をクリック

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
        "Service": "build.apprunner.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

3. 「次へ」をクリック

#### 許可ポリシーのアタッチ

4. 検索ボックスに `AWSAppRunnerServicePolicyForECRAccess` と入力してチェック ✅
5. 「次へ」をクリック

#### ロールの詳細

| 項目 | 値 |
|------|-----|
| ロール名 | `AppRunnerECRAccessRole` |

6. 「**ロールを作成**」をクリック

---

### 3-2. インスタンスロール（App Runner が DynamoDB・S3 にアクセスするため）

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

### Dockerイメージのビルド・プッシュ

5. 作成したリポジトリをクリック →「**プッシュコマンドを表示**」をクリック
6. 表示された4つのコマンドをローカル環境で順番に実行する

```
# 例（コンソールに表示される実際のコマンドをコピーして使うこと）
1. aws ecr get-login-password ...  （ECRへのログイン）
2. docker build -t streamlit-chatbot .  （イメージビルド）
3. docker tag streamlit-chatbot:latest ...  （タグ付け）
4. docker push ...  （プッシュ）
```

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
| ECRアクセスロール | `AppRunnerECRAccessRole` |

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

## 作業順序まとめ

```
Step 1: DynamoDB テーブル作成
Step 2: S3 バケット作成
Step 3: IAM ロール作成（ECRアクセスロール → インスタンスロール）
Step 4: ECR リポジトリ作成 → Docker ビルド＆プッシュ（ローカル作業）
Step 5: App Runner サービス作成
```
