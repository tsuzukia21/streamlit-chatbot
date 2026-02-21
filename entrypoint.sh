#!/bin/bash

# .streamlit ディレクトリを作成
mkdir -p /app/.streamlit

# config.toml を生成
cat > /app/.streamlit/config.toml <<'TOML'
[server]
enableCORS = false
enableXsrfProtection = false
enableWebsocketCompression = false

[browser]
gatherUsageStats = false
TOML

# secrets.toml を環境変数から生成（OIDC認証用）
if [ -n "$GOOGLE_OAUTH_CLIENT_ID" ] && [ -n "$GOOGLE_OAUTH_CLIENT_SECRET" ]; then
  cat > /app/.streamlit/secrets.toml <<EOF
[auth]
redirect_uri = "${AUTH_REDIRECT_URI}"
cookie_secret = "${AUTH_COOKIE_SECRET:-$(python3 -c 'import secrets; print(secrets.token_hex(32))')}"

[auth.google]
client_id = "${GOOGLE_OAUTH_CLIENT_ID}"
client_secret = "${GOOGLE_OAUTH_CLIENT_SECRET}"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
EOF
  echo "OIDC auth configured"
else
  echo "WARNING: GOOGLE_OAUTH_CLIENT_ID or GOOGLE_OAUTH_CLIENT_SECRET not set"
fi

# Streamlit 起動
exec streamlit run main.py \
  --server.port=$PORT \
  --server.address=0.0.0.0 \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --server.enableWebsocketCompression=false
