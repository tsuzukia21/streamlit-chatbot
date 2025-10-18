"""
データベース管理モジュール
FirestoreとCloud Storageを使用してチャット履歴と画像を永続化
"""
import os
import json
import base64
import mimetypes
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import storage
from google.cloud.firestore_v1.base_query import FieldFilter
import streamlit as st

# Firestore と Cloud Storage クライアント
_db = None
_storage_client = None
_bucket = None

def init_db() -> None:
    """Firestore と Cloud Storage の初期化"""
    global _db, _storage_client, _bucket
    
    if _db is not None:
        return  # 既に初期化済み
    
    # Firebase Admin SDK の初期化
    if not firebase_admin._apps:
        # Cloud Run上では自動的に認証される
        # ローカル開発の場合は GOOGLE_APPLICATION_CREDENTIALS 環境変数を設定
        try:
            # デフォルトの認証情報を使用
            firebase_admin.initialize_app()
        except Exception as e:
            st.error(f"Firebase初期化エラー: {e}")
            raise
    
    _db = firestore.client()
    
    # Cloud Storage クライアントの初期化
    _storage_client = storage.Client()
    
    # バケット名を環境変数から取得（デフォルト値も設定）
    bucket_name = os.environ.get('GCS_BUCKET_NAME', '')
    if not bucket_name:
        # Streamlit secrets から取得を試みる
        try:
            bucket_name = st.secrets.get('GCS_BUCKET_NAME', '')
        except:
            pass
    
    if bucket_name:
        _bucket = _storage_client.bucket(bucket_name)
    else:
        st.warning("GCS_BUCKET_NAME が設定されていません。画像機能は使用できません。")

def get_db():
    """Firestoreクライアントを取得"""
    if _db is None:
        init_db()
    return _db

def get_bucket():
    """Cloud Storageバケットを取得"""
    if _bucket is None:
        init_db()
    return _bucket

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

def save_image_file(conversation_id: str, message_id: str, index: int, 
                    data_uri: str) -> str:
    """
    画像をCloud Storageに保存し、パスを返す
    
    Args:
        conversation_id: 会話ID
        message_id: メッセージID
        index: 画像のインデックス
        data_uri: data:image/png;base64,... 形式のURI
    
    Returns:
        保存した画像のGCSパス（例: "images/conv123_msg456_0.png"）
    """
    bucket = get_bucket()
    if not bucket:
        return ""
    
    # data URIをパース
    if "," in data_uri:
        header, base64_data = data_uri.split(",", 1)
        mime_type = header.split(";")[0].replace("data:", "")
        ext = get_extension_from_mime(mime_type)
    else:
        base64_data = data_uri
        ext = ".png"
    
    # ファイル名を生成
    filename = f"conv{conversation_id}_msg{message_id}_{index}{ext}"
    blob_path = f"images/{filename}"
    
    # Base64デコードしてアップロード
    image_bytes = base64.b64decode(base64_data)
    blob = bucket.blob(blob_path)
    blob.upload_from_string(image_bytes, content_type=mime_type if "," in data_uri else "image/png")
    
    return blob_path

def load_image_file(blob_path: str) -> str:
    """
    Cloud Storageから画像を読み込んでdata URIに変換
    
    Args:
        blob_path: GCSパス（例: "images/conv123_msg456_0.png"）
    
    Returns:
        data:image/png;base64,... 形式のURI
    """
    bucket = get_bucket()
    if not bucket:
        return ""
    
    try:
        blob = bucket.blob(blob_path)
        if not blob.exists():
            return ""
        
        # ファイル名から拡張子を取得してMIMEタイプを推測
        mime_type = blob.content_type or "image/png"
        
        # ダウンロードしてBase64エンコード
        image_bytes = blob.download_as_bytes()
        base64_data = base64.b64encode(image_bytes).decode('utf-8')
        
        return f"data:{mime_type};base64,{base64_data}"
    except Exception:
        return ""

def create_conversation(user_id: str, title: str) -> str:
    """
    新しい会話を作成
    
    Returns:
        作成した会話のID（Firestoreの自動生成ID）
    """
    db = get_db()
    
    doc_ref = db.collection('conversations').document()
    doc_ref.set({
        'user_id': user_id,
        'title': title,
        'created_at': firestore.SERVER_TIMESTAMP,
        'updated_at': firestore.SERVER_TIMESTAMP,
        'is_deleted': False
    })
    
    return doc_ref.id

def get_conversations(user_id: str) -> List[Dict[str, Any]]:
    """
    ユーザーの会話一覧を取得（論理削除されていないもののみ）
    
    Returns:
        会話のリスト（新しい順、最大10件）
    """
    db = get_db()
    
    conversations_ref = db.collection('conversations')
    query = conversations_ref.where(filter=FieldFilter('user_id', '==', user_id))\
                             .where(filter=FieldFilter('is_deleted', '==', False))\
                             .order_by('updated_at', direction=firestore.Query.DESCENDING)\
                             .limit(10)
    
    docs = query.stream()
    
    conversations = []
    for doc in docs:
        data = doc.to_dict()
        data['id'] = doc.id
        # Timestampをstringに変換
        if data.get('created_at'):
            data['created_at'] = data['created_at'].isoformat() if hasattr(data['created_at'], 'isoformat') else str(data['created_at'])
        if data.get('updated_at'):
            data['updated_at'] = data['updated_at'].isoformat() if hasattr(data['updated_at'], 'isoformat') else str(data['updated_at'])
        conversations.append(data)
    
    return conversations

def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    特定の会話情報を取得
    
    Returns:
        会話情報の辞書、存在しない場合はNone
    """
    db = get_db()
    
    doc_ref = db.collection('conversations').document(conversation_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return None
    
    data = doc.to_dict()
    data['id'] = doc.id
    # Timestampをstringに変換
    if data.get('created_at'):
        data['created_at'] = data['created_at'].isoformat() if hasattr(data['created_at'], 'isoformat') else str(data['created_at'])
    if data.get('updated_at'):
        data['updated_at'] = data['updated_at'].isoformat() if hasattr(data['updated_at'], 'isoformat') else str(data['updated_at'])
    
    return data

def update_conversation_timestamp(conversation_id: str) -> None:
    """会話の更新日時を現在時刻に更新"""
    db = get_db()
    
    doc_ref = db.collection('conversations').document(conversation_id)
    doc_ref.update({
        'updated_at': firestore.SERVER_TIMESTAMP
    })

def update_conversation_title(conversation_id: str, title: str) -> None:
    """会話のタイトルを更新"""
    db = get_db()
    
    doc_ref = db.collection('conversations').document(conversation_id)
    doc_ref.update({
        'title': title,
        'updated_at': firestore.SERVER_TIMESTAMP
    })

def delete_conversation(conversation_id: str) -> None:
    """会話を論理削除"""
    db = get_db()
    
    doc_ref = db.collection('conversations').document(conversation_id)
    doc_ref.update({
        'is_deleted': True
    })

def save_message(conversation_id: str, role: str, content: Any, 
                 tokens: int = 0, reasoning: str = "") -> str:
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
    db = get_db()
    
    # contentをJSON文字列として保存
    content_json = json.dumps(content, ensure_ascii=False)
    
    messages_ref = db.collection('conversations').document(conversation_id).collection('messages')
    doc_ref = messages_ref.document()
    
    doc_ref.set({
        'role': role,
        'content': content_json,
        'tokens': tokens,
        'reasoning': reasoning,
        'created_at': firestore.SERVER_TIMESTAMP
    })
    
    # 会話の更新日時を更新
    update_conversation_timestamp(conversation_id)
    
    return doc_ref.id

def update_message_content(conversation_id: str, message_id: str, content: Any) -> None:
    """メッセージの内容を更新（画像保存後にパスを更新する用）"""
    db = get_db()
    
    content_json = json.dumps(content, ensure_ascii=False)
    
    doc_ref = db.collection('conversations').document(conversation_id)\
                .collection('messages').document(message_id)
    doc_ref.update({
        'content': content_json
    })

def save_message_with_images(conversation_id: str, role: str, content: Any,
                              tokens: int = 0, reasoning: str = "") -> str:
    """
    画像を含むメッセージを保存（画像はCloud Storageに保存）
    
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
    
    # contentがリストで画像を含む場合、画像をCloud Storageに保存してパスを更新
    if isinstance(content, list):
        updated_content = []
        image_index = 0
        
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                data_uri = item["image_url"]["url"]
                # data URIならCloud Storageに保存
                if data_uri.startswith("data:"):
                    blob_path = save_image_file(
                        conversation_id, message_id, image_index, data_uri
                    )
                    # パスに置き換え
                    updated_content.append({
                        "type": "image_url",
                        "image_url": {"url": blob_path}
                    })
                    image_index += 1
                else:
                    # すでにパスの場合はそのまま
                    updated_content.append(item)
            else:
                updated_content.append(item)
        
        # 画像パスに更新したcontentで再保存
        update_message_content(conversation_id, message_id, updated_content)
    
    return message_id

def get_messages(conversation_id: str) -> List[Tuple[str, Any]]:
    """
    会話のメッセージ履歴を取得（古い順）
    
    Returns:
        (role, content)のタプルのリスト
        contentは画像がある場合はdata URIに変換して返す
    """
    db = get_db()
    
    messages_ref = db.collection('conversations').document(conversation_id).collection('messages')
    query = messages_ref.order_by('created_at', direction=firestore.Query.ASCENDING)
    
    docs = query.stream()
    
    messages = []
    for doc in docs:
        data = doc.to_dict()
        role = data['role']
        content_json = data['content']
        
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
                    # Cloud Storageのパスならdata URIに変換
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

def get_last_reasoning(conversation_id: str) -> str:
    """最後のアシスタントメッセージの推論過程を取得"""
    db = get_db()
    
    messages_ref = db.collection('conversations').document(conversation_id).collection('messages')
    query = messages_ref.where(filter=FieldFilter('role', '==', 'assistant'))\
                        .order_by('created_at', direction=firestore.Query.DESCENDING)\
                        .limit(1)
    
    docs = query.stream()
    
    for doc in docs:
        data = doc.to_dict()
        return data.get('reasoning', '')
    
    return ""

def delete_message_images(conversation_id: str, message_id: str) -> None:
    """
    メッセージに関連する画像ファイルをCloud Storageから削除
    
    Args:
        conversation_id: 会話ID
        message_id: メッセージID
    """
    bucket = get_bucket()
    if not bucket:
        return
    
    # パターンに一致するblobを検索して削除
    prefix = f"images/conv{conversation_id}_msg{message_id}_"
    blobs = bucket.list_blobs(prefix=prefix)
    
    for blob in blobs:
        try:
            blob.delete()
        except Exception:
            pass

def delete_messages_from_index(conversation_id: str, message_index: int) -> None:
    """
    指定したインデックス以降のメッセージをFirestoreから削除
    
    Args:
        conversation_id: 会話ID
        message_index: 削除開始インデックス（このインデックス以降を削除）
    """
    db = get_db()
    
    # メッセージを取得（作成日時順）
    messages_ref = db.collection('conversations').document(conversation_id).collection('messages')
    query = messages_ref.order_by('created_at', direction=firestore.Query.ASCENDING)
    
    docs = list(query.stream())
    
    # インデックスが範囲内かチェック
    if message_index < len(docs):
        # 削除対象のドキュメント
        docs_to_delete = docs[message_index:]
        
        # メッセージと関連画像を削除
        for doc in docs_to_delete:
            # 関連する画像を削除
            delete_message_images(conversation_id, doc.id)
            # メッセージを削除
            doc.reference.delete()
        
        # updated_atを更新
        update_conversation_timestamp(conversation_id)

# データベース初期化（モジュールインポート時）
init_db()
