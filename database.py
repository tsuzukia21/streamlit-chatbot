"""
データベース管理モジュール
SQLiteを使用してチャット履歴と画像を永続化
"""
import sqlite3
import json
import base64
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import mimetypes

# データベースと画像の保存先
DB_DIR = Path("chat_data")
DB_PATH = DB_DIR / "chatbot.db"
IMAGES_DIR = DB_DIR / "images"

def init_db() -> None:
    """データベースとディレクトリの初期化"""
    # ディレクトリ作成
    DB_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # conversationsテーブル
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_deleted BOOLEAN DEFAULT 0
        )
    """)
    
    # messagesテーブル
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tokens INTEGER DEFAULT 0,
            reasoning TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    
    # インデックス作成
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversations_user_id 
        ON conversations(user_id, is_deleted, updated_at DESC)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_id 
        ON messages(conversation_id, created_at ASC)
    """)
    
    conn.commit()
    conn.close()

def get_extension_from_mime(mime_type: str) -> str:
    """MIMEタイプから拡張子を取得"""
    ext = mimetypes.guess_extension(mime_type)
    if ext:
        return ext
    # フォールバック
    mime_map = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp"
    }
    return mime_map.get(mime_type, ".bin")

def save_image_file(conversation_id: int, message_id: int, index: int, 
                    data_uri: str) -> str:
    """
    画像をファイルとして保存し、相対パスを返す
    
    Args:
        conversation_id: 会話ID
        message_id: メッセージID
        index: 画像のインデックス（同じメッセージ内で複数画像がある場合）
        data_uri: data:image/png;base64,... 形式のURI
    
    Returns:
        保存した画像の相対パス（例: "images/conv1_msg2_0.png"）
    """
    # data URIをパース
    if "," in data_uri:
        header, base64_data = data_uri.split(",", 1)
        # MIMEタイプを取得
        mime_type = header.split(";")[0].replace("data:", "")
        ext = get_extension_from_mime(mime_type)
    else:
        base64_data = data_uri
        ext = ".png"  # デフォルト
    
    # ファイル名を生成
    filename = f"conv{conversation_id}_msg{message_id}_{index}{ext}"
    filepath = IMAGES_DIR / filename
    
    # Base64デコードして保存
    image_bytes = base64.b64decode(base64_data)
    filepath.write_bytes(image_bytes)
    
    # 相対パスを返す（DBに保存する用）
    return f"images/{filename}"

def load_image_file(relative_path: str) -> str:
    """
    ファイルパスから画像を読み込んでdata URIに変換
    
    Args:
        relative_path: 相対パス（例: "images/conv1_msg2_0.png"）
    
    Returns:
        data:image/png;base64,... 形式のURI
    """
    filepath = DB_DIR / relative_path
    if not filepath.exists():
        # ファイルが存在しない場合はエラー画像を返すなど
        return ""
    
    # 拡張子からMIMEタイプを推測
    mime_type, _ = mimetypes.guess_type(str(filepath))
    if not mime_type:
        mime_type = "image/png"
    
    # ファイルを読み込んでBase64エンコード
    image_bytes = filepath.read_bytes()
    base64_data = base64.b64encode(image_bytes).decode('utf-8')
    
    return f"data:{mime_type};base64,{base64_data}"

def create_conversation(user_id: str, title: str) -> int:
    """
    新しい会話を作成
    
    Returns:
        作成した会話のID
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversations (user_id, title)
        VALUES (?, ?)
    """, (user_id, title))
    conversation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return conversation_id

def get_conversations(user_id: str) -> List[Dict[str, Any]]:
    """
    ユーザーの会話一覧を取得（論理削除されていないもののみ）
    
    Returns:
        会話のリスト（新しい順）
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, title, created_at, updated_at
        FROM conversations
        WHERE user_id = ? AND is_deleted = 0
        ORDER BY updated_at DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def get_conversation(conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    特定の会話情報を取得
    
    Returns:
        会話情報の辞書、存在しない場合はNone
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, user_id, title, created_at, updated_at, is_deleted
        FROM conversations
        WHERE id = ?
    """, (conversation_id,))
    row = cursor.fetchone()
    conn.close()
    
    return dict(row) if row else None

def update_conversation_timestamp(conversation_id: int) -> None:
    """会話の更新日時を現在時刻に更新"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE conversations
        SET updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (conversation_id,))
    conn.commit()
    conn.close()

def delete_conversation(conversation_id: int) -> None:
    """会話を論理削除"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE conversations
        SET is_deleted = 1
        WHERE id = ?
    """, (conversation_id,))
    conn.commit()
    conn.close()

def save_message(conversation_id: int, role: str, content: Any, 
                 tokens: int = 0, reasoning: str = "") -> int:
    """
    メッセージを保存
    
    Args:
        conversation_id: 会話ID
        role: 'human' or 'assistant'
        content: メッセージ内容（文字列 or リスト）
        tokens: トークン数
        reasoning: 思考プロセス（assistantのみ）
    
    Returns:
        保存したメッセージのID
    """
    # contentを一時的にJSON文字列として保存（後で画像パスを更新する）
    content_json = json.dumps(content, ensure_ascii=False)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO messages (conversation_id, role, content, tokens, reasoning)
        VALUES (?, ?, ?, ?, ?)
    """, (conversation_id, role, content_json, tokens, reasoning))
    message_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    # 会話の更新日時を更新
    update_conversation_timestamp(conversation_id)
    
    return message_id

def update_message_content(message_id: int, content: Any) -> None:
    """メッセージの内容を更新（画像保存後にパスを更新する用）"""
    content_json = json.dumps(content, ensure_ascii=False)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE messages
        SET content = ?
        WHERE id = ?
    """, (content_json, message_id))
    conn.commit()
    conn.close()

def save_message_with_images(conversation_id: int, role: str, content: Any,
                              tokens: int = 0, reasoning: str = "") -> int:
    """
    画像を含むメッセージを保存（画像はファイルとして保存）
    
    Args:
        conversation_id: 会話ID
        role: 'human' or 'assistant'
        content: メッセージ内容（文字列 or 画像URLを含むリスト）
        tokens: トークン数
        reasoning: 思考プロセス
    
    Returns:
        保存したメッセージのID
    """
    # まずメッセージをDBに保存してIDを取得
    message_id = save_message(conversation_id, role, content, tokens, reasoning)
    
    # contentがリストで画像を含む場合、画像をファイルに保存してパスを更新
    if isinstance(content, list):
        updated_content = []
        image_index = 0
        
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                data_uri = item["image_url"]["url"]
                # data URIならファイルに保存
                if data_uri.startswith("data:"):
                    relative_path = save_image_file(
                        conversation_id, message_id, image_index, data_uri
                    )
                    # パスに置き換え
                    updated_content.append({
                        "type": "image_url",
                        "image_url": {"url": relative_path}
                    })
                    image_index += 1
                else:
                    # すでにパスの場合はそのまま
                    updated_content.append(item)
            else:
                updated_content.append(item)
        
        # 画像パスに更新したcontentで再保存
        update_message_content(message_id, updated_content)
    
    return message_id

def get_messages(conversation_id: int) -> List[Tuple[str, Any]]:
    """
    会話のメッセージ履歴を取得（古い順）
    
    Returns:
        (role, content)のタプルのリスト
        contentは画像がある場合はdata URIに変換して返す
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content, reasoning
        FROM messages
        WHERE conversation_id = ?
        ORDER BY created_at ASC
    """, (conversation_id,))
    rows = cursor.fetchall()
    conn.close()
    
    messages = []
    for row in rows:
        role = row["role"]
        content_json = row["content"]
        
        try:
            content = json.loads(content_json)
        except json.JSONDecodeError:
            content = content_json
        
        # contentがリストで画像パスを含む場合、data URIに変換
        if isinstance(content, list):
            converted_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    # ファイルパスならdata URIに変換
                    if url.startswith("images/"):
                        data_uri = load_image_file(url)
                        converted_content.append({
                            "type": "image_url",
                            "image_url": {"url": data_uri}
                        })
                    else:
                        converted_content.append(item)
                else:
                    converted_content.append(item)
            content = converted_content
        
        messages.append((role, content))
    
    return messages

def get_last_reasoning(conversation_id: int) -> str:
    """最後のアシスタントメッセージの推論過程を取得"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT reasoning
        FROM messages
        WHERE conversation_id = ? AND role = 'assistant'
        ORDER BY created_at DESC
        LIMIT 1
    """, (conversation_id,))
    row = cursor.fetchone()
    conn.close()
    
    return row[0] if row and row[0] else ""

def delete_message_images(conversation_id: int, message_id: int) -> None:
    """
    メッセージに関連する画像ファイルを削除
    
    Args:
        conversation_id: 会話ID
        message_id: メッセージID
    """
    import glob
    pattern = str(IMAGES_DIR / f"conv{conversation_id}_msg{message_id}_*")
    for filepath in glob.glob(pattern):
        try:
            Path(filepath).unlink()
        except Exception:
            pass

def delete_messages_from_index(conversation_id: int, message_index: int) -> None:
    """
    指定したインデックス以降のメッセージをDBから削除
    
    Args:
        conversation_id: 会話ID
        message_index: 削除開始インデックス（このインデックス以降を削除）
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # conversation_idのメッセージをcreated_atでソートしてIDを取得
    cursor.execute("""
        SELECT id FROM messages 
        WHERE conversation_id = ? 
        ORDER BY created_at ASC
    """, (conversation_id,))
    
    message_ids = [row[0] for row in cursor.fetchall()]
    
    # インデックスが範囲内かチェック
    if message_index < len(message_ids):
        # 削除対象のIDリスト
        ids_to_delete = message_ids[message_index:]
        
        # メッセージを削除
        placeholders = ','.join('?' * len(ids_to_delete))
        cursor.execute(f"""
            DELETE FROM messages 
            WHERE id IN ({placeholders})
        """, ids_to_delete)
        
        # 関連する画像ファイルも削除
        for msg_id in ids_to_delete:
            delete_message_images(conversation_id, msg_id)
        
        # updated_atを更新
        cursor.execute("""
            UPDATE conversations 
            SET updated_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        """, (conversation_id,))
    
    conn.commit()
    conn.close()

# データベース初期化（モジュールインポート時）
init_db()

