# タスク: GCPからAWSへのインフラ移行

**入力**: `/specs/001-gcp-aws-migration/` の設計ドキュメント
**前提**: plan.md (必須), spec.md (必須), research.md, data-model.md, contracts/database-api.md

**テスト**: テスト未要求のため、テストタスクは含まない。

**構成**: ユーザーストーリーごとにタスクをグループ化し、各ストーリーの独立した実装・テストを可能にする。

## フォーマット: `[ID] [P?] [Story] 説明`

- **[P]**: 並列実行可能（異なるファイル、依存関係なし）
- **[Story]**: 対応するユーザーストーリー（例: US1, US2, US3）
- 説明には正確なファイルパスを含める

---

## フェーズ 1: セットアップ（共有インフラ）

**目的**: 依存関係の更新とGCPパッケージの除去

- [x] T001 pyproject.toml を更新: firebase-admin と google-cloud-storage を削除、boto3 を追加 (pyproject.toml)
- [x] T002 requirements.txt を更新: GCPパッケージを削除、boto3 を追加 (requirements.txt)
- [x] T003 更新した依存関係をローカルにインストール

---

## フェーズ 2: 基盤（ブロッキング前提条件）

**目的**: DynamoDB/S3クライアントの初期化とベースインフラ。全ユーザーストーリーの前提条件。

**⚠️ 重要**: このフェーズが完了するまで、ユーザーストーリーの作業は開始できない

- [x] T004 `init_db()` を boto3 DynamoDB リソースと S3 クライアントの初期化で実装（firebase_admin 初期化を置換、環境変数 AWS_REGION, DYNAMODB_TABLE_NAME, S3_BUCKET_NAME を読み取り） (core/database.py)
- [x] T005 Unix ミリ秒タイムスタンプを返すヘルパー関数 `get_timestamp()` を実装 (core/database.py)
- [x] T006 conversation_id と message_id 用に `uuid.uuid4().hex` による ID 生成を実装 (core/database.py)

**チェックポイント**: 基盤完了 - DynamoDB/S3クライアント初期化完了、ユーザーストーリー実装開始可能

---

## フェーズ 3: ユーザーストーリー 1 - 会話データの永続化がAWS上で動作する (優先度: P1) 🎯 MVP

**ゴール**: 会話のCRUD操作（作成・取得・更新・削除）がDynamoDBで動作する

**独立テスト**: 新規会話を作成し、メッセージを送受信し、ページリロード後に会話履歴が保持されていることを確認する

### ユーザーストーリー 1 の実装

- [x] T007 [US1] `create_conversation(user_id, title)` を DynamoDB `put_item` で実装（pk=CONV#<id>, sk=METADATA） (core/database.py)
- [x] T008 [US1] `get_conversations(user_id)` を GSI `UserConversationsIndex` クエリで実装（is_deleted フィルタ、ScanIndexForward=False、Limit=10） (core/database.py)
- [x] T009 [US1] `get_conversation(conversation_id)` を DynamoDB `get_item` で実装 (core/database.py)
- [x] T010 [P] [US1] `update_conversation_title(conversation_id, title)` を DynamoDB `update_item` で実装 (core/database.py)
- [x] T011 [P] [US1] `update_conversation_tokens(conversation_id, tokens)` を DynamoDB `update_item` で実装 (core/database.py)
- [x] T012 [P] [US1] `get_conversation_tokens(conversation_id)` を DynamoDB `get_item` で実装 (core/database.py)
- [x] T013 [P] [US1] `update_conversation_timestamp(conversation_id)` を DynamoDB `update_item` で実装 (core/database.py)
- [x] T014 [US1] `delete_conversation(conversation_id)` を is_deleted=True に設定する DynamoDB `update_item` で実装 (core/database.py)
- [x] T015 [US1] `save_message(conversation_id, role, content, reasoning)` を DynamoDB `put_item` で実装（pk=CONV#<id>, sk=<timestamp>#<msgId>） (core/database.py)
- [x] T016 [US1] `get_messages(conversation_id)` を DynamoDB `query` で実装（pk=CONV#<id>, sk > METADATA） (core/database.py)
- [x] T017 [P] [US1] `update_message_content(conversation_id, message_id, content)` を DynamoDB `update_item` で実装 (core/database.py)
- [x] T018 [US1] `get_last_reasoning(conversation_id)` を DynamoDB `query` で実装（ScanIndexForward=False、Limit=1、role=assistant フィルタ） (core/database.py)
- [x] T019 [US1] `delete_messages_from_index(conversation_id, message_index)` をメッセージをクエリしてから指定インデックス以降を一括削除で実装 (core/database.py)

**チェックポイント**: 会話のCRUD操作がDynamoDBで完全動作。メッセージの保存・取得・削除が正常動作。

---

## フェーズ 4: ユーザーストーリー 2 - 画像アップロードがAWS上で動作する (優先度: P2)

**ゴール**: 画像のS3保存・取得・削除が動作し、画像付きメッセージが正しく表示される

**独立テスト**: 画像付きメッセージを送信し、ページリロード後に画像が正しく表示されることを確認する

### ユーザーストーリー 2 の実装

- [x] T020 [US2] `save_image_file(conversation_id, message_id, index, data_uri)` を S3 `put_object` で実装 (core/database.py)
- [x] T021 [US2] `load_image_file(blob_path)` を S3 `get_object` で実装し、data URI に変換 (core/database.py)
- [x] T022 [US2] `delete_message_images(conversation_id, message_id)` を S3 `list_objects_v2` と `delete_object` で実装 (core/database.py)
- [x] T023 [US2] `save_message_with_images(conversation_id, role, content, reasoning)` を実装: data URI を抽出し S3 にアップロード、パスに置換後 DynamoDB に保存 (core/database.py)
- [x] T024 [US2] `get_messages()` を更新: S3 画像パスを `load_image_file()` で data URI に変換 (core/database.py)
- [x] T025 [US2] `delete_messages_from_index()` を更新: 削除対象メッセージの関連 S3 画像も削除 (core/database.py)

**チェックポイント**: 画像付きメッセージの送信・保存・表示・削除がS3で完全動作。

---

## フェーズ 5: ユーザーストーリー 3 - App Runner上でアプリケーションが稼働する (優先度: P3)

**ゴール**: DockerイメージがApp Runner上でデプロイ・稼働し、ブラウザからアクセスできる

**独立テスト**: App RunnerにデプロイしたアプリケーションのURLにアクセスし、チャットボットが正常に動作することを確認する

### ユーザーストーリー 3 の実装

- [x] T026 [US3] Dockerfile を更新: GCP固有の設定を削除し、PORT=8080 の互換性を確認 (Dockerfile)
- [ ] T027 [US3] 更新した依存関係（GCPパッケージなし）で Docker イメージが正常にビルドされることを確認

**チェックポイント**: Dockerイメージがビルド可能で、App Runnerデプロイ準備完了。

---

## フェーズ 6: ユーザーストーリー 4 - 認証がAWS環境で動作する (優先度: P3)

**ゴール**: シークレット管理の移行 — st.secrets から環境変数への切り替え

**独立テスト**: App Runner上のアプリケーションでGoogleアカウントによるログイン・ログアウトが正常に動作することを確認する

### ユーザーストーリー 4 の実装

- [x] T028 [US4] main.py を更新: `st.secrets` の参照を `os.environ` に置換（OAuth設定: client_id, client_secret, cookie_secret、およびAPIキー） (main.py)
- [x] T029 [US4] main.py から GCP Secret Manager の参照・インポートを削除 (main.py)

**チェックポイント**: 認証とシークレット管理が環境変数ベースで動作。

---

## フェーズ 7: 仕上げ・横断的関心事

**目的**: GCP依存の完全除去確認とクリーンアップ

- [ ] T030 コードベース全体で GCP インポート（firebase_admin, google.cloud）が完全に除去されていることを確認
- [ ] T031 全ファイルから未使用の GCP 関連コード・コメント・設定を削除
- [ ] T032 core/database.py のエラーメッセージを GCP 参照から AWS 参照に更新（例: "Firebase初期化エラー" → "AWS初期化エラー"、"GCS_BUCKET_NAME" → "S3_BUCKET_NAME"）
- [ ] T033 quickstart.md の検証を実行: ローカル開発環境のセットアップがエンドツーエンドで動作することを確認

---

## 依存関係と実行順序

### フェーズ間の依存関係

- **セットアップ（フェーズ 1）**: 依存関係なし - すぐに開始可能
- **基盤（フェーズ 2）**: セットアップ完了に依存 - 全ユーザーストーリーをブロック
- **US1（フェーズ 3）**: 基盤（フェーズ 2）に依存 - 他ストーリーへの依存なし
- **US2（フェーズ 4）**: 基盤（フェーズ 2）に依存 - US1 の `save_message`/`get_messages`/`delete_messages_from_index` と統合
- **US3（フェーズ 5）**: セットアップ（フェーズ 1）に依存 - US1/US2 と並列進行可能
- **US4（フェーズ 6）**: 他ストーリーへのコード依存なし - 並列進行可能
- **仕上げ（フェーズ 7）**: 全ユーザーストーリーの完了に依存

### 各ユーザーストーリー内の順序

- US1: 会話操作 → メッセージ操作の順（メッセージは CONV#id に依存）
- US2: 画像基本操作(T020-T022) → 画像付きメッセージ(T023) → get_messages/delete 更新(T024-T025)
- US3: Dockerfile 更新 → ビルド確認
- US4: main.py 更新 → GCP 参照除去

### 並列実行の機会

- T010, T011, T012, T013 は並列実行可能（独立した更新関数）
- T017 は他の US1 タスクと並列実行可能（独立した更新関数）
- US3（フェーズ 5）と US4（フェーズ 6）はフェーズ 2 完了後すぐに開始可能
- US3 と US4 は互いに並列実行可能

---

## 並列実行例: ユーザーストーリー 1

```bash
# 独立した会話更新関数を同時に実行:
タスク: "T010 update_conversation_title を実装 (core/database.py)"
タスク: "T011 update_conversation_tokens を実装 (core/database.py)"
タスク: "T012 get_conversation_tokens を実装 (core/database.py)"
タスク: "T013 update_conversation_timestamp を実装 (core/database.py)"
```

---

## 実装戦略

### MVP ファースト（ユーザーストーリー 1 のみ）

1. フェーズ 1 完了: セットアップ（依存関係更新）
2. フェーズ 2 完了: 基盤（DynamoDB/S3クライアント初期化）
3. フェーズ 3 完了: ユーザーストーリー 1（会話CRUD）
4. **停止して検証**: ローカルで会話の作成・メッセージ送受信・履歴表示を確認
5. MVP として動作確認完了

### インクリメンタルデリバリー

1. セットアップ + 基盤 → DynamoDB/S3 接続準備完了
2. ユーザーストーリー 1 → 会話データ永続化動作 → 検証 (MVP!)
3. ユーザーストーリー 2 → 画像アップロード動作 → 検証
4. ユーザーストーリー 3 + 4 → デプロイ準備・認証移行 → 検証
5. 仕上げ → GCP 完全除去確認 → 最終検証

---

## 備考

- 全タスクの主要変更対象は `core/database.py`（491行）— 既存インターフェースを維持し内部実装のみ差し替え
- `main.py` は US4 でのシークレット管理移行のみ
- テストフレームワークが既存でないため、手動テストで検証
- 既存データのマイグレーションはスコープ外
