"""
データベース管理モジュール
DynamoDBとS3を使用してチャット履歴と画像を永続化
"""
import os
import json
import base64
import mimetypes
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import boto3
from boto3.dynamodb.conditions import Key, Attr
import streamlit as st

# DynamoDB と S3 クライアント
_dynamodb_table = None
_s3_client = None
_s3_bucket_name = None


def get_timestamp() -> int:
    """現在時刻のUnixミリ秒タイムスタンプを返す"""
    return int(datetime.now().timestamp() * 1000)


def _generate_id() -> str:
    """uuid.uuid4().hex でIDを生成"""
    return uuid.uuid4().hex


def init_db() -> None:
    """DynamoDB と S3 クライアントの初期化"""
    global _dynamodb_table, _s3_client, _s3_bucket_name

    if _dynamodb_table is not None:
        return  # 既に初期化済み

    region = os.environ.get('AWS_REGION')
    table_name = os.environ.get('DYNAMODB_TABLE_NAME')
    _s3_bucket_name = os.environ.get('S3_BUCKET_NAME')

    try:
        dynamodb = boto3.resource('dynamodb', region_name=region)
        _dynamodb_table = dynamodb.Table(table_name)
        _s3_client = boto3.client('s3', region_name=region)
    except Exception as e:
        st.error(f"AWS初期化エラー: {e}")
        raise

def get_db():
    """DynamoDBテーブルを取得"""
    if _dynamodb_table is None:
        init_db()
    return _dynamodb_table


def get_s3():
    """(S3クライアント, バケット名)のタプルを取得"""
    if _s3_client is None:
        init_db()
    return _s3_client, _s3_bucket_name


def get_bucket():
    """S3バケット名を返す（画像関数から呼ばれる後方互換インタフェース）"""
    if _s3_client is None:
        init_db()
    return _s3_bucket_name or None

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
        作成した会話のID
    """
    table = get_db()

    conversation_id = _generate_id()
    now = get_timestamp()

    table.put_item(Item={
        'pk': f'CONV#{conversation_id}',
        'sk': 'METADATA',
        'entity_type': 'conversation',
        'user_id': user_id,
        'title': title,
        'total_tokens': 0,
        'is_deleted': False,
        'created_at': now,
        'updated_at': now,
    })

    return conversation_id

def get_conversations(user_id: str) -> List[Dict[str, Any]]:
    """
    ユーザーの会話一覧を取得（論理削除されていないもののみ）

    Returns:
        会話のリスト（新しい順、最大10件）
    """
    table = get_db()

    response = table.query(
        IndexName='UserConversationsIndex',
        KeyConditionExpression=Key('user_id').eq(user_id),
        FilterExpression=Attr('is_deleted').eq(False),
        ScanIndexForward=False,
        Limit=10,
    )

    conversations = []
    for item in response.get('Items', []):
        conversations.append({
            'id': item['pk'].replace('CONV#', ''),
            'user_id': item.get('user_id'),
            'title': item.get('title', ''),
            'total_tokens': item.get('total_tokens', 0),
            'is_deleted': item.get('is_deleted', False),
            'created_at': item.get('created_at'),
            'updated_at': item.get('updated_at'),
        })

    return conversations

def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    特定の会話情報を取得

    Returns:
        会話情報の辞書、存在しない場合はNone
    """
    table = get_db()

    response = table.get_item(Key={
        'pk': f'CONV#{conversation_id}',
        'sk': 'METADATA',
    })

    item = response.get('Item')
    if not item:
        return None

    return {
        'id': conversation_id,
        'user_id': item.get('user_id'),
        'title': item.get('title', ''),
        'total_tokens': item.get('total_tokens', 0),
        'is_deleted': item.get('is_deleted', False),
        'created_at': item.get('created_at'),
        'updated_at': item.get('updated_at'),
    }

def update_conversation_timestamp(conversation_id: str) -> None:
    """会話の更新日時を現在時刻に更新"""
    table = get_db()

    table.update_item(
        Key={
            'pk': f'CONV#{conversation_id}',
            'sk': 'METADATA',
        },
        UpdateExpression='SET updated_at = :updated_at',
        ExpressionAttributeValues={
            ':updated_at': get_timestamp(),
        },
    )

def update_conversation_title(conversation_id: str, title: str) -> None:
    """会話のタイトルを更新"""
    table = get_db()

    table.update_item(
        Key={
            'pk': f'CONV#{conversation_id}',
            'sk': 'METADATA',
        },
        UpdateExpression='SET title = :title, updated_at = :updated_at',
        ExpressionAttributeValues={
            ':title': title,
            ':updated_at': get_timestamp(),
        },
    )

def update_conversation_tokens(conversation_id: str, tokens: int) -> None:
    """会話のトークン数を更新（マルチターン全体のトークン数）"""
    table = get_db()

    table.update_item(
        Key={
            'pk': f'CONV#{conversation_id}',
            'sk': 'METADATA',
        },
        UpdateExpression='SET total_tokens = :tokens, updated_at = :updated_at',
        ExpressionAttributeValues={
            ':tokens': tokens,
            ':updated_at': get_timestamp(),
        },
    )

def get_conversation_tokens(conversation_id: str) -> int:
    """会話の総トークン数を取得（マルチターン全体のトークン数）"""
    table = get_db()

    response = table.get_item(
        Key={
            'pk': f'CONV#{conversation_id}',
            'sk': 'METADATA',
        },
        ProjectionExpression='total_tokens',
    )

    item = response.get('Item')
    if item:
        return int(item.get('total_tokens', 0))
    return 0

def delete_conversation(conversation_id: str) -> None:
    """会話を論理削除"""
    db = get_db()
    
    doc_ref = db.collection('conversations').document(conversation_id)
    doc_ref.update({
        'is_deleted': True
    })

def save_message(conversation_id: str, role: str, content: Any, 
                 reasoning: str = "") -> str:
    """
    メッセージを保存
    
    Args:
        conversation_id: 会話ID
        role: 'human' or 'assistant'
        content: メッセージ内容（文字列 or リスト）
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
                              reasoning: str = "") -> str:
    """
    画像を含むメッセージを保存（画像はCloud Storageに保存）
    
    Args:
        conversation_id: 会話ID
        role: 'human' or 'assistant'
        content: メッセージ内容（文字列 or 画像URLを含むリスト）
        reasoning: 思考プロセス
    
    Returns:
        保存したメッセージのID
    """
    # まずメッセージをDBに保存してIDを取得
    message_id = save_message(conversation_id, role, content, reasoning)
    
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
