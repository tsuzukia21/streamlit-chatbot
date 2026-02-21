# Feature Specification: GCPからAWSへのインフラ移行

**Feature Branch**: `001-gcp-aws-migration`
**Created**: 2026-02-21
**Status**: Draft
**Input**: User description: "GCPからAWSへ移行したい"

## Clarifications

### Session 2026-02-21

- Q: AWSデータベースサービスの選択は？ → A: DynamoDB（NoSQL、Firestoreと同系統、サーバーレス）
- Q: AWSコンテナホスティングサービスの選択は？ → A: App Runner（Cloud Run相当、シンプル、自動スケール）
- Q: GCPからの完全移行 vs 段階的移行？ → A: 完全移行（GCP依存コードを全て除去、AWSのみ）

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 会話データの永続化がAWS上で動作する (Priority: P1)

ユーザーがチャットボットでメッセージを送信すると、会話データがDynamoDBに保存され、再度アクセスした際に過去の会話履歴が正しく表示される。

**Why this priority**: データの永続化はアプリケーションの中核機能であり、Firestore→DynamoDB移行の最も重要な部分。これが動作しなければ他の機能も成立しない。

**Independent Test**: 移行後のアプリケーションで新規会話を作成し、メッセージを送受信し、ページリロード後に会話履歴が保持されていることを確認する。

**Acceptance Scenarios**:

1. **Given** ユーザーがログイン済みの状態, **When** メッセージを送信する, **Then** 会話データがDynamoDBに保存される
2. **Given** 会話が保存されている状態, **When** ページをリロードする, **Then** 過去の会話一覧と履歴が正しく表示される
3. **Given** 複数の会話が存在する状態, **When** サイドバーで会話を切り替える, **Then** 該当する会話のメッセージ履歴が正しく読み込まれる

---

### User Story 2 - 画像アップロードがAWS上で動作する (Priority: P2)

ユーザーが画像付きメッセージを送信した際、画像がS3に保存され、会話履歴を再表示した際に画像が正しく表示される。

**Why this priority**: 画像機能はチャットボットのマルチモーダル機能の重要な部分であり、Cloud Storage→S3移行が必要。

**Independent Test**: 画像付きメッセージを送信し、ページリロード後に画像が正しく表示されることを確認する。

**Acceptance Scenarios**:

1. **Given** ユーザーが会話中の状態, **When** 画像付きメッセージを送信する, **Then** 画像がS3に保存され、チャットに表示される
2. **Given** 画像を含む会話が保存済みの状態, **When** 会話履歴を再読み込みする, **Then** 画像がS3から取得されて正しく表示される

---

### User Story 3 - App Runner上でアプリケーションが稼働する (Priority: P3)

Streamlitアプリケーション全体がApp Runner上でデプロイ・稼働し、ユーザーがブラウザからアクセスして利用できる。

**Why this priority**: ホスティング環境の移行は最終的なゴールだが、先にデータ層の移行が完了していないと動作確認ができない。

**Independent Test**: App RunnerにデプロイしたアプリケーションのURLにアクセスし、チャットボットが正常に動作することを確認する。

**Acceptance Scenarios**:

1. **Given** Dockerイメージがビルドされている状態, **When** App Runnerにデプロイする, **Then** アプリケーションが起動し、ブラウザからアクセスできる
2. **Given** アプリケーションが稼働している状態, **When** ユーザーがチャット操作を行う, **Then** すべての機能（メッセージ送受信、画像アップロード、会話管理）が正常に動作する

---

### User Story 4 - 認証がAWS環境で動作する (Priority: P3)

現在のGoogle OAuth認証（Streamlitの`st.login()`）がApp Runner環境でも正常に動作し、ユーザーがGoogleアカウントでログインできる。

**Why this priority**: 認証自体はStreamlitの組み込み機能を使用しており、ホスティング環境に依存しないため優先度は低い。ただし、環境変数やシークレット管理の移行は必要。

**Independent Test**: App Runner上のアプリケーションでGoogleアカウントによるログイン・ログアウトが正常に動作することを確認する。

**Acceptance Scenarios**:

1. **Given** 未ログインの状態, **When** Googleログインボタンを押す, **Then** Google OAuth認証フローが正常に完了しログインできる
2. **Given** ログイン済みの状態, **When** ログアウトボタンを押す, **Then** 正常にログアウトできる

---

### Edge Cases

- DynamoDB接続が失敗した場合、ユーザーに適切なエラーメッセージが表示されるか？
- S3への画像アップロードがタイムアウトした場合の挙動はどうなるか？
- AWSサービスの認証情報（IAMロール等）が正しく設定されていない場合のフォールバック処理
- 大量の会話履歴がある場合のDynamoDBクエリパフォーマンス
- GCP依存コード（firebase-admin、google-cloud-storage）が完全に除去されていることの確認

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: システムは会話データ（作成、取得、更新、削除）をDynamoDBで永続化できなければならない
- **FR-002**: システムは画像ファイルのアップロード・ダウンロードをS3で処理できなければならない
- **FR-003**: システムはApp Runner上でDockerコンテナとして動作しなければならない
- **FR-004**: システムは既存のFirestore/Cloud Storageと同等のデータ操作（CRUD）をDynamoDB/S3で提供しなければならない
- **FR-005**: システムはAWSのシークレット管理サービスまたは環境変数を通じて設定情報を安全に管理しなければならない
- **FR-006**: システムは現在のデータ構造（会話コレクション、メッセージのサブコレクション構造）と同等のデータモデルをDynamoDB上で再現しなければならない
- **FR-007**: システムは画像の一意識別が可能な保存形式をS3上で使用しなければならない
- **FR-008**: システムはサーバータイムスタンプを使用した会話の時系列管理を維持しなければならない
- **FR-009**: システムからGCP依存パッケージ（firebase-admin、google-cloud-storage）を完全に除去しなければならない

### Key Entities

- **Conversation（会話）**: ユーザーごとの会話セッション。user_id、title、created_at、updated_at、is_deleted、total_tokensを持つ。DynamoDBテーブルとして管理。
- **Message（メッセージ）**: 会話内の個別メッセージ。role（human/assistant）、content（テキストまたはマルチモーダル）、reasoning、created_atを持つ。DynamoDBテーブルとして管理。
- **Image（画像）**: メッセージに添付された画像ファイル。S3バケットに保存され、パスでメッセージと関連付けられる。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 移行後のアプリケーションで、会話の作成・読み込み・更新・削除がエラーなく動作する
- **SC-002**: 画像付きメッセージの送信と再表示が、移行前と同様の体験で動作する
- **SC-003**: アプリケーションの応答時間が移行前と同等で、ユーザーが体感できる遅延の増加がない
- **SC-004**: 全てのユーザーシナリオ（会話管理、モデル切り替え、画像アップロード、認証）が移行後も正常に動作する
- **SC-005**: DockerイメージのビルドとApp RunnerへのデプロイがCI/CDパイプラインを通じて自動化できる状態になる
- **SC-006**: GCP依存パッケージがpyproject.tomlおよびコードベースから完全に除去されている

## Assumptions

- Google OAuth認証はStreamlitの`st.login()`機能を使用しており、ホスティング環境に依存しないため、認証プロバイダ自体の変更は不要
- 既存データ（Firestoreの会話履歴）のAWSへのデータマイグレーションは初期スコープに含まない（新規データからAWSを使用）
- LLM APIキー（OpenAI、Anthropic、Google等）の管理はAWSのシークレット管理サービスまたは環境変数で行う
- AWSのリージョンは東京リージョン（ap-northeast-1）を使用する
- コスト最適化はスコープ外とし、まず機能的な移行を優先する

## Dependencies

- AWSアカウントが利用可能であること
- 必要なAWSサービスへのアクセス権限（IAMロール/ポリシー）が設定されていること
- Dockerイメージのビルド環境が利用可能であること
