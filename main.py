import streamlit as st
from core.MODEL_CONFIG import MODEL_CONFIG
from core.config import DEFAULT_SYSTEM_PROMPT, TOKEN_LIMIT, MESSAGE_LIMIT, DEFAULT_TEMPERATURE, DEFAULT_MODEL
import math
from typing import List, Dict, Any
import base64
from st_chat_input_multimodal import multimodal_chat_input
import core.database as db
import core.ui_components as ui
import core.conversation as conv
import core.llm_handler as llm
import time
st.set_page_config(layout="wide", page_title="streamlit chatbot",page_icon=":material/chat:")

def get_user_id() -> str:
    """
    ユーザーIDを取得
    """
    # ログイン済みユーザーのIDを優先
    try:
        if st.user.is_logged_in:
            # Googleプロバイダーではsubフィールドを使用
            user_id = getattr(st.user, 'sub', None)
            if user_id:
                return user_id
    except Exception:
        pass
    
    # 既存のフォールバック処理（互換性のため残す）
    try:
        cookies = st.context.cookies
        user_id = cookies.get("ajs_user_id") or cookies.get("ajs_anonymous_id")
        if user_id:
            return user_id
    except Exception:
        pass
    
    # 最終フォールバック：セッションベースのID
    if "fallback_user_id" not in st.session_state:
        import uuid
        st.session_state.fallback_user_id = f"user_{uuid.uuid4().hex[:16]}"
    return st.session_state.fallback_user_id

def initialize_session_state():
    """セッション状態の初期化を一元管理"""
    defaults = {
        "done": True,
        "save": False,
        "stop": False,
        "edit_states": {},
        "total_tokens": 0,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "temperature": DEFAULT_TEMPERATURE,
        "error_message": "",
        "model_index": 0,
        "chat_history": [],
        "model": DEFAULT_MODEL,
        "reasoning": "",  # 推論過程の保存
        "current_conversation_id": None,  # 現在の会話ID
        "user_id": None,  # ユーザーID
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # ユーザーIDの取得
    if st.session_state.user_id is None:
        st.session_state.user_id = get_user_id()
    
    # LLM初期化（modelが設定された後）
    if "llm" not in st.session_state:
        model_name = st.session_state.model
        config = MODEL_CONFIG[model_name]
        st.session_state.llm = config["llm_factory"]

def check_token() -> bool:
    def limit_error(msg: str) -> bool:
        st.error(msg, icon="🚨")
        st.session_state.done = True
        st.session_state.save = False
        return False
    
    if st.session_state.total_tokens > TOKEN_LIMIT:
        persent=min(100, math.floor(100 * st.session_state.total_tokens / TOKEN_LIMIT))
        return limit_error(f'Error: Text volume is {persent}% of the limit.  \nPlease delete unnecessary parts or reset the conversation')
    if len(st.session_state.chat_history) > MESSAGE_LIMIT:
        return limit_error('Error: Conversation limit exceeded. Please reset the conversation')
    return True

def update_system_prompt():
    st.session_state.system_prompt = st.session_state.new_system_prompt

def update_model():
    """モデル切り替え時にLLMインスタンスとインデックスを更新"""
    model_name = st.session_state.model
    config = MODEL_CONFIG.get(model_name)
    if config:
        st.session_state.llm = config["llm_factory"]
        st.session_state.model_index = config["index"]

# セッション状態を初期化
initialize_session_state()

# サイドバー：ログイン機能
with st.sidebar:
    # ログイン状態の確認
    if not st.user.is_logged_in:
        st.warning("⚠️ ログインが必要です")
        if st.button("🔐 Googleアカウントでログイン", use_container_width=True, type="primary", key="login_button"):
            st.login()
        st.stop()  # ログインするまでここで停止
    else:
        # 許可されたメールアドレスのホワイトリスト検証（secrets.toml の [auth].allowed_emails）
        try:
            allowed_emails = st.secrets.get("auth", {}).get("allowed_emails", [])
        except Exception:
            allowed_emails = []
        user_email = getattr(st.user, "email", None)
        if allowed_emails and (user_email not in allowed_emails):
            st.error("アクセス権がありません")
            time.sleep(3)
            st.logout()
            st.stop()
        # ログイン済みユーザー情報表示
        st.success(f"👤 {st.user.name}")
        if st.button("ログアウト", use_container_width=True, key="logout_button"):
            st.logout()
    st.divider()

with st.sidebar.container():
    st.markdown(":material/settings: モデル設定")
    model_options = list(MODEL_CONFIG.keys())
    st.selectbox("model",
                 options=model_options,
                 format_func=lambda x: MODEL_CONFIG[x]["display_name"],
                 help="推論モデルを選択できます。全てのモデルが思考プロセスに対応しています。",
                 index=st.session_state.model_index,
                 key="model",
                 on_change=update_model)
    st.text_area("system prompt",value=st.session_state.system_prompt,on_change=update_system_prompt,key="new_system_prompt",
                                 help="You can provide a prompt to the system. This is only effective at the first message transmission.")
    st.divider()
    st.markdown(":material/message: 会話管理")
    
    # 新しい会話ボタン
    if st.button(":material/add: 新しい会話", use_container_width=True, type="primary"):
        conv.create_new_conversation()
    
    # 会話削除ボタン（現在の会話がある場合のみ表示）
    if st.session_state.current_conversation_id:
        if st.button(":material/delete: この会話を削除", use_container_width=True):
            conv.delete_current_conversation()
    
    conversations = db.get_conversations(st.session_state.user_id)
    
    if conversations:
        for conversation in conversations:
            conv_id = conversation["id"]
            title = conversation["title"]
            updated_at = conversation["updated_at"]
            
            # 現在の会話をハイライト
            is_current = conv_id == st.session_state.current_conversation_id
            button_type = "primary" if is_current else "secondary"
            
            # 会話ボタン
            if st.button(
                f"{':material/push_pin: ' if is_current else ''}{title}",
                key=f"conv_{conv_id}",
                use_container_width=True,
                type=button_type,
                disabled=is_current
            ):
                conv.load_conversation(conv_id)
    

st.title("Streamlit ChatBot")

user_input = multimodal_chat_input(
    placeholder="Send a message",
    enable_voice_input=True,
    voice_recognition_method="openai_whisper",
    voice_language="ja-JP",
    key="chat_input"
)

ui.show_chat_history(messages=st.session_state.chat_history,edit=True, error_message=st.session_state.error_message, show_copy_button=True)

if user_input is not None:
    st.session_state.error_message = ""
    st.session_state.done = False
    ok = check_token()
    if ok:
        # 会話がない場合は新規作成
        if not st.session_state.current_conversation_id:
            user_id = st.session_state.user_id
            title = "新しい会話"  # 最初のメッセージで自動更新される
            conversation_id = db.create_conversation(user_id, title)
            st.session_state.current_conversation_id = conversation_id
        
        # Extract text from multimodal input
        input_text = user_input.get("text", "")
        input_files = user_input.get("files", [])
        
        # Use text content for LangChain input
        llm_input = input_text if input_text else "Image uploaded"
        
        # Display user message with images
        with st.chat_message("human", avatar=":material/mood:"):
            col1, col2 = st.columns([9, 1])
            with col1:
                if input_text:
                    st.markdown(ui.render_markdown(input_text), unsafe_allow_html=True)
                
                # Display images
                ui.render_uploaded_images(input_files)
        
        # Extract image URLs for prompt template
        image_urls: List[str] = []
        for file in input_files:
            if file.get("type", "").startswith("image/"):
                try:
                    base64_data = file['data'].split(',')[1] if ',' in file['data'] else file['data']
                    # Validate base64 data
                    base64.b64decode(base64_data)
                    image_urls.append(file["data"])
                except (ValueError, base64.binascii.Error) as e:
                    st.warning(f"画像 '{file.get('name', 'unknown')}' の読み込みに失敗しました: {str(e)}")
        
        # Add human message to history
        if image_urls:
            human_payload: List[Dict[str, Any]] = []
            if input_text:
                human_payload.append({"type": "text", "text": input_text})
            for url in image_urls:
                human_payload.append({"type": "image_url", "image_url": {"url": url}})
            st.session_state.chat_history.append(("human", human_payload))
        else:
            st.session_state.chat_history.append(("human", input_text))
        
        # DBに人間のメッセージを保存
        conversation_id = st.session_state.current_conversation_id
        human_content = human_payload if image_urls else input_text
        db.save_message_with_images(conversation_id, "human", human_content)
        
        # 最初のメッセージの場合、会話タイトルを更新
        if len(st.session_state.chat_history) == 1:
            title = conv.generate_title_from_message(human_content)
            db.update_conversation_title(conversation_id, title)
        
        # Execute chat turn
        response, tokens = llm.run_chat_turn(
            prompt=llm_input,
            conversation_history=st.session_state.chat_history[:-1],
            image_urls=image_urls if image_urls else None
        )
        
        # 会話の累積トークン数を更新
        db.update_conversation_tokens(conversation_id, tokens)
        st.session_state.total_tokens = db.get_conversation_tokens(conversation_id)
        
        st.session_state.chat_history.append(("assistant", response))
        
        # DBにアシスタントのメッセージを保存
        db.save_message_with_images(
            conversation_id, 
            "assistant", 
            response, 
            reasoning=st.session_state.reasoning
        )
        
        # Reset state and rerun
        st.session_state.done = True
        st.session_state.stop = False
        st.rerun()

if st.session_state.save:
    st.session_state.error_message = ""
    st.session_state.done = False
    prompt = st.session_state.new_message
    ui.show_chat_history(messages=st.session_state.chat_history, edit=False, error_message=st.session_state.error_message, new_message=prompt, show_copy_button=False)
    ok = check_token()
    if not ok:
        st.session_state.save = False
    else:
        # 会話がない場合は新規作成
        if not st.session_state.current_conversation_id:
            user_id = st.session_state.user_id
            title = "新しい会話"  # 最初のメッセージで自動更新される
            conversation_id = db.create_conversation(user_id, title)
            st.session_state.current_conversation_id = conversation_id
        
        # Add edited human message to history
        st.session_state.chat_history.append(("human", prompt))
        
        # DBに人間のメッセージを保存
        conversation_id = st.session_state.current_conversation_id
        db.save_message_with_images(conversation_id, "human", prompt)
        
        # Execute chat turn (text only, no images for edited messages)
        response, tokens = llm.run_chat_turn(
            prompt=prompt,
            conversation_history=st.session_state.chat_history[:-1],
            image_urls=None
        )
        
        # 会話の累積トークン数を更新
        db.update_conversation_tokens(conversation_id, tokens)
        st.session_state.total_tokens = db.get_conversation_tokens(conversation_id)
        
        st.session_state.chat_history.append(("assistant", response))
        
        # DBにアシスタントのメッセージを保存
        db.save_message_with_images(
            conversation_id, 
            "assistant", 
            response, 
            reasoning=st.session_state.reasoning
        )
        
        # Reset state and rerun
        st.session_state.done = True
        st.session_state.save = False
        st.session_state.stop = False
        st.rerun()

