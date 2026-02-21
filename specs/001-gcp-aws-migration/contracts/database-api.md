# Database API Contract: core/database.py

**Date**: 2026-02-21

## Overview

`core/database.py` の公開インターフェースは移行前後で変更なし。内部実装のみ Firestore/GCS → DynamoDB/S3 に差し替える。

---

## 初期化

### `init_db() -> None`
DynamoDB と S3 クライアントの初期化。モジュールインポート時に自動実行。

**移行前**: `firebase_admin.initialize_app()` + `storage.Client()`
**移行後**: `boto3.resource('dynamodb')` + `boto3.client('s3')`

**環境変数:**
- `AWS_REGION` (default: `ap-northeast-1`)
- `DYNAMODB_TABLE_NAME` (default: `ChatbotData`)
- `S3_BUCKET_NAME` (必須、未設定時は画像機能無効)

---

## 会話操作

### `create_conversation(user_id: str, title: str) -> str`
新しい会話を作成し、会話IDを返す。

**Input**: user_id (非空文字列), title (文字列)
**Output**: conversation_id (uuid4 hex)
**Side Effects**: DynamoDB に会話メタデータ書き込み

### `get_conversations(user_id: str) -> List[Dict[str, Any]]`
ユーザーの会話一覧を取得（論理削除除外、新しい順、最大10件）。

**Input**: user_id
**Output**: `[{id, user_id, title, total_tokens, is_deleted, created_at, updated_at}, ...]`
**Notes**: GSI `UserConversationsIndex` を使用

### `get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]`
特定の会話を取得。

**Input**: conversation_id
**Output**: `{id, user_id, title, ...}` or `None`

### `update_conversation_title(conversation_id: str, title: str) -> None`
会話タイトルを更新。updated_at も更新。

### `update_conversation_tokens(conversation_id: str, tokens: int) -> None`
会話のトークン数を更新。updated_at も更新。

### `get_conversation_tokens(conversation_id: str) -> int`
会話の総トークン数を取得。存在しない場合は 0。

### `delete_conversation(conversation_id: str) -> None`
会話を論理削除（is_deleted = True）。

### `update_conversation_timestamp(conversation_id: str) -> None`
会話の updated_at を現在時刻に更新。

---

## メッセージ操作

### `save_message(conversation_id: str, role: str, content: Any, reasoning: str = "") -> str`
メッセージを保存し、メッセージIDを返す。

**Input**: conversation_id, role (`human`/`assistant`), content (文字列 or リスト), reasoning
**Output**: message_id (uuid4 hex)
**Side Effects**: DynamDB書き込み + 会話タイムスタンプ更新

### `save_message_with_images(conversation_id: str, role: str, content: Any, reasoning: str = "") -> str`
画像を含むメッセージを保存。画像は S3 にアップロードし、content 内の data URI をパスに置換。

**Input**: 同上
**Output**: message_id
**Side Effects**: DynamoDB書き込み + S3画像アップロード + 会話タイムスタンプ更新

### `get_messages(conversation_id: str) -> List[Tuple[str, Any]]`
会話のメッセージ履歴を取得（古い順）。S3パスは data URI に変換して返す。

**Input**: conversation_id
**Output**: `[(role, content), ...]`

### `update_message_content(conversation_id: str, message_id: str, content: Any) -> None`
メッセージの content を更新。

### `get_last_reasoning(conversation_id: str) -> str`
最後の assistant メッセージの reasoning を取得。

### `delete_messages_from_index(conversation_id: str, message_index: int) -> None`
指定インデックス以降のメッセージと関連画像を削除。

---

## 画像操作

### `save_image_file(conversation_id: str, message_id: str, index: int, data_uri: str) -> str`
画像を S3 に保存し、パスを返す。

**Input**: conversation_id, message_id, index, data_uri (`data:image/png;base64,...`)
**Output**: S3パス (`images/conv{id}_msg{id}_{index}.{ext}`)

### `load_image_file(blob_path: str) -> str`
S3 から画像を読み込み、data URI に変換。

**Input**: S3パス
**Output**: `data:{mime};base64,{data}` or `""` (エラー時)

### `delete_message_images(conversation_id: str, message_id: str) -> None`
メッセージに関連する画像を S3 から削除。

---

## エラーハンドリング

| シナリオ | 現行 (GCP) | 移行後 (AWS) |
|---------|-----------|-------------|
| DB接続失敗 | `st.error("Firebase初期化エラー")` | `st.error("AWS初期化エラー")` |
| オブジェクト不存在 | `google.cloud.exceptions.NotFound` | `botocore.exceptions.ClientError (NoSuchKey/404)` |
| バケット未設定 | `st.warning("GCS_BUCKET_NAME...")` | `st.warning("S3_BUCKET_NAME...")` |
| 画像操作失敗 | `except Exception: return ""` | `except Exception: return ""` (同一) |
