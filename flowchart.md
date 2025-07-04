# Streamlit Chatbot Flow

```mermaid
flowchart TD
    A[アプリケーション開始] --> B[セッション状態の初期化]
    B --> C[サイドバーUIの表示]
    C --> D[チャット履歴の表示]
    D --> E[ユーザー入力待機]
    
    E --> F{ユーザー入力あり?}
    F -->|No| E
    F -->|Yes| G[トークン制限チェック]
    
    G --> H{制限超過?}
    H -->|Yes| I[エラー表示・履歴クリア]
    I --> E
    H -->|No| J[新しいメッセージ表示]
    
    J --> K[プロンプトテンプレート作成]
    K --> L[LLMチェーン実行]
    L --> M[ストリーミング応答]
    
    M --> N{停止ボタン押下?}
    N -->|Yes| O[応答停止]
    N -->|No| P{応答完了?}
    P -->|No| M
    P -->|Yes| Q[履歴に追加]
    
    O --> Q
    Q --> R[画面再描画]
    R --> E
    
    %% 編集機能の流れ
    D --> S{編集ボタン押下?}
    S -->|Yes| T[編集モード開始]
    T --> U[テキストエリア表示]
    U --> V{保存ボタン押下?}
    V -->|No| U
    V -->|Yes| W[メッセージ更新]
    W --> X[履歴切り詰め]
    X --> Y[新しい応答生成]
    Y --> R
    
    %% サイドバー設定変更
    C --> Z{設定変更?}
    Z -->|モデル変更| AA[LLMインスタンス更新]
    Z -->|システムプロンプト変更| BB[システムプロンプト更新]
    Z -->|温度設定変更| CC[温度パラメータ更新]
    AA --> D
    BB --> D
    CC --> D
    
    %% 履歴クリア
    D --> DD{クリアボタン押下?}
    DD -->|Yes| EE[全履歴削除]
    EE --> FF[セッション状態リセット]
    FF --> R
    
    style A fill:#e1f5fe
    style F fill:#fff3e0
    style H fill:#ffebee
    style N fill:#fff3e0
    style S fill:#f3e5f5
    style V fill:#f3e5f5
    style Z fill:#e8f5e8
    style DD fill:#fff8e1
```

## 主要なコンポーネント

### セッション状態管理
- `chat_history`: 会話履歴
- `edit_states`: 編集状態の管理
- `total_tokens`: トークン使用量
- `llm`: 現在のLLMインスタンス
- `system_prompt`: システムプロンプト
- `temperature`: 温度設定

### 主要な処理フロー

1. **初期化**: セッション状態の設定、デフォルトLLMの設定
2. **UI表示**: サイドバーとメインチャット画面の描画
3. **ユーザー入力処理**: 入力検証、トークン制限チェック
4. **AI応答生成**: LangChainを使用したストリーミング応答
5. **履歴管理**: メッセージの追加、編集、削除
6. **設定変更**: モデル切り替え、パラメータ調整

### 特殊機能

- **メッセージ編集**: 任意のメッセージを編集し、以降の履歴を再生成
- **応答中断**: ストリーミング中に応答を停止可能
- **トークン管理**: 50,000トークン制限での自動履歴クリア
- **マルチモデル対応**: OpenAI、Anthropic、Google の3つのAIプロバイダー