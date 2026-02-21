# ベースイメージ
FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# アプリケーションファイルをコピー
COPY . /app

# Poetryのインストール
RUN pip install --no-cache-dir poetry

# Poetryの設定（仮想環境を作成しない）
RUN poetry config virtualenvs.create false

# 依存関係をインストール
RUN poetry install --no-interaction --no-ansi --no-root

# App Runnerのポート（デフォルト8080）
ENV PORT=8080

# Streamlitの設定
ENV STREAMLIT_SERVER_PORT=$PORT
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# 起動スクリプトをコピー
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# アプリケーション起動
CMD ["/app/entrypoint.sh"]
