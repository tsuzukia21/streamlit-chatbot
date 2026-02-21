# Data Model: GCPからAWSへのインフラ移行

**Date**: 2026-02-21

## DynamoDB テーブル設計

### テーブル: `ChatbotData`

**キー設計:**

| 属性 | 型 | 説明 |
|------|------|------|
| `pk` (Partition Key) | String | `CONV#<conversationId>` |
| `sk` (Sort Key) | String | `METADATA` (会話) / `<timestamp>#<messageId>` (メッセージ) |

**GSI: `UserConversationsIndex`**

| 属性 | 型 | 説明 |
|------|------|------|
| `user_id` (Partition Key) | String | ユーザーID |
| `updated_at` (Sort Key) | Number | Unixタイムスタンプ (ms) |

**Projection**: ALL

---

### エンティティ: Conversation（会話）

```json
{
  "pk": "CONV#<conversationId>",
  "sk": "METADATA",
  "entity_type": "conversation",
  "user_id": "<userId>",
  "title": "会話タイトル",
  "total_tokens": 0,
  "is_deleted": false,
  "created_at": 1708531200000,
  "updated_at": 1708531200000
}
```

| フィールド | 型 | 必須 | バリデーション | 説明 |
|-----------|------|------|-------------|------|
| pk | String | Yes | `CONV#` prefix | パーティションキー |
| sk | String | Yes | `METADATA` 固定 | ソートキー |
| entity_type | String | Yes | `conversation` 固定 | エンティティ種別 |
| user_id | String | Yes | 非空 | Googleログインの sub claim |
| title | String | Yes | 最大100文字 | LLM生成 or デフォルト「新しい会話」 |
| total_tokens | Number | Yes | >= 0 | マルチターン累計トークン数 |
| is_deleted | Boolean | Yes | - | 論理削除フラグ |
| created_at | Number | Yes | Unix ms | 作成日時 |
| updated_at | Number | Yes | Unix ms | 更新日時（GSIソートキー） |

---

### エンティティ: Message（メッセージ）

```json
{
  "pk": "CONV#<conversationId>",
  "sk": "1708617600000#<messageId>",
  "entity_type": "message",
  "conversation_id": "<conversationId>",
  "message_id": "<messageId>",
  "role": "human",
  "content": "{\"text\": \"Hello\"}",
  "reasoning": "",
  "created_at": 1708617600000
}
```

| フィールド | 型 | 必須 | バリデーション | 説明 |
|-----------|------|------|-------------|------|
| pk | String | Yes | `CONV#` prefix | 会話のパーティションキー |
| sk | String | Yes | `<timestamp>#<id>` | 時系列ソート用 |
| entity_type | String | Yes | `message` 固定 | エンティティ種別 |
| conversation_id | String | Yes | 非空 | 会話ID（冗長だがクエリ利便性のため） |
| message_id | String | Yes | UUID | メッセージの一意ID |
| role | String | Yes | `human` or `assistant` | 送信者ロール |
| content | String | Yes | JSON文字列 | メッセージ内容 |
| reasoning | String | No | - | AI思考プロセス（assistant のみ） |
| created_at | Number | Yes | Unix ms | 作成日時 |

**content フォーマット:**
- テキストのみ: `"Hello, how are you?"`
- マルチモーダル: `[{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "images/conv123_msg456_0.png"}}]`

---

## S3 バケット設計

### バケット: `streamlit-chatbot-{env}-images`

**構造:**
```
s3://{bucket_name}/
└── images/
    └── conv{conversationId}_msg{messageId}_{index}.{ext}
```

**例:**
```
images/conv_abc123_msg_def456_0.png
images/conv_abc123_msg_def456_1.jpg
images/conv_abc123_msg_ghi789_0.webp
```

| 設定 | 値 |
|------|------|
| パブリックアクセス | 全ブロック |
| バージョニング | 無効 |
| 暗号化 | SSE-S3 |
| リージョン | ap-northeast-1 |

---

## アクセスパターン

| # | 操作 | DynamoDB API | キー条件 | フィルタ |
|---|------|-------------|---------|--------|
| 1 | 会話作成 | `put_item` | pk=`CONV#<id>`, sk=`METADATA` | - |
| 2 | 会話一覧取得 | `query` (GSI) | user_id=`<uid>` | is_deleted=false |
| 3 | 会話取得 | `get_item` | pk=`CONV#<id>`, sk=`METADATA` | - |
| 4 | 会話更新 | `update_item` | pk=`CONV#<id>`, sk=`METADATA` | - |
| 5 | 会話論理削除 | `update_item` | pk=`CONV#<id>`, sk=`METADATA` | - |
| 6 | メッセージ保存 | `put_item` | pk=`CONV#<id>`, sk=`<ts>#<msgId>` | - |
| 7 | メッセージ一覧 | `query` | pk=`CONV#<id>`, sk > `METADATA` | - |
| 8 | メッセージ内容更新 | `update_item` | pk=`CONV#<id>`, sk=`<ts>#<msgId>` | - |
| 9 | メッセージ削除 | `delete_item` (複数) | pk=`CONV#<id>`, sk=各メッセージ | - |
| 10 | 最新reasoning取得 | `query` | pk=`CONV#<id>`, sk > `METADATA` | role=assistant, Limit=1, DESC |

---

## ID生成

| エンティティ | 現行(Firestore) | 移行後(DynamoDB) |
|------------|----------------|-----------------|
| conversation_id | Firestore自動生成 | Python `uuid.uuid4().hex` |
| message_id | Firestore自動生成 | Python `uuid.uuid4().hex` |

---

## 環境変数

| 変数名 | 説明 | 例 |
|-------|------|-----|
| `AWS_REGION` | AWSリージョン | `ap-northeast-1` |
| `DYNAMODB_TABLE_NAME` | DynamoDBテーブル名 | `ChatbotData` |
| `S3_BUCKET_NAME` | S3バケット名 | `streamlit-chatbot-prod-images` |
