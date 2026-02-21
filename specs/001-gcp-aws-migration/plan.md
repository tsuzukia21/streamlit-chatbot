# Implementation Plan: GCPからAWSへのインフラ移行

**Branch**: `001-gcp-aws-migration` | **Date**: 2026-02-21 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-gcp-aws-migration/spec.md`

## Summary

GCPベースのStreamlitチャットボットアプリケーションをAWSに完全移行する。Firestore→DynamoDB、Cloud Storage→S3、Cloud Run→App Runnerへの移行を行い、GCP依存パッケージを完全に除去する。既存データのマイグレーションはスコープ外。コードベースの主要変更箇所は`core/database.py`（491行）であり、`main.py`の認証・設定部分と`Dockerfile`のデプロイ設定も更新が必要。

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: Streamlit ^1.50.0, LangChain ^1.1.2, boto3 (新規追加), langchain-openai, langchain-anthropic, langchain-google-genai
**Storage**: DynamoDB (会話・メッセージ), S3 (画像ファイル) — 現行: Firestore + Cloud Storage
**Testing**: 手動テスト（既存テストフレームワークなし）
**Target Platform**: AWS App Runner (Docker コンテナ) — 現行: Cloud Run
**Project Type**: Single (Streamlit web application)
**Performance Goals**: 移行前と同等の応答速度（体感遅延増加なし）
**Constraints**: 東京リージョン (ap-northeast-1), GCP依存コード完全除去
**Scale/Scope**: 個人〜少人数利用、会話10件表示、画像アップロード対応

## Constitution Check

*GATE: No constitution file found. Skipping gate check.*

Constitution file (`.specify/memory/constitution.md`) が存在しないため、制約チェックは省略。

## Project Structure

### Documentation (this feature)

```text
specs/001-gcp-aws-migration/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── database-api.md  # Database layer API contract
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
streamlit-chatbot/
├── main.py                   # エントリポイント（認証・セッション管理・UI）
├── core/
│   ├── MODEL_CONFIG.py       # LLMモデル設定
│   ├── config.py             # アプリ設定・システムプロンプト
│   ├── conversation.py       # 会話CRUD操作
│   ├── database.py           # データ永続化層 ★主要変更対象（Firestore/GCS → DynamoDB/S3）
│   ├── llm_handler.py        # LangChainストリーミング処理
│   └── ui_components.py      # チャットUI描画
├── Dockerfile                # コンテナ定義 ★更新対象
├── pyproject.toml            # 依存関係 ★更新対象（firebase-admin → boto3）
└── requirements.txt          # pip依存関係 ★更新対象
```

**Structure Decision**: 既存のシングルプロジェクト構造を維持。`core/database.py`の内部実装をGCP SDKからAWS SDK (boto3)に差し替えるのみで、外部インターフェースは変更しない。

## Complexity Tracking

> No constitution violations to justify.
