# ベースイメージ
FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# Poetryのインストール
RUN pip install --no-cache-dir poetry

# Poetryの設定（仮想環境を作成しない）
RUN poetry config virtualenvs.create false

# 依存関係ファイルをコピー
COPY pyproject.toml ./

# 依存関係をインストール
RUN poetry install --no-interaction --no-ansi --no-root

# アプリケーションファイルをコピー
COPY main.py ./

# Cloud Runのポート（デフォルト8080）
ENV PORT=8080

# Streamlitの設定
ENV STREAMLIT_SERVER_PORT=$PORT
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# アプリケーション起動
CMD streamlit run main.py --server.port=$PORT --server.address=0.0.0.0

