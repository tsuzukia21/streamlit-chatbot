import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
from st_txt_copybutton import txt_copy
import math
from typing import Tuple, Union, List, Dict, Any, Optional
import base64
from st_chat_input_multimodal import multimodal_chat_input
import database as db
from langchain_core.messages import HumanMessage

st.set_page_config(layout="wide", page_title="streamlit chatbot",page_icon=":material/chat:")

# モデル設定の一元管理
MODEL_CONFIG = {
    "claude-sonnet-4.5": {
        "provider": "anthropic",
        "display_name": "Claude Sonnet 4.5",
        "index": 0,
        "llm_factory": lambda temp: ChatAnthropic(
            temperature=1.0,
            model_name="claude-sonnet-4-5-20250929",
            max_tokens=16384,
            timeout=120,
            max_retries=3,
            thinking={"type": "enabled","budget_tokens": 8192}
        )
    },
    "gemini-2.5-pro": {
        "provider": "google",
        "display_name": "Gemini 2.5 Pro",
        "index": 1,
        "llm_factory": lambda temp: ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=1.0,
            thinking_budget=16000,
            include_thoughts=True
        )
    },
    "gpt-5": {
        "provider": "openai",
        "display_name": "GPT 5",
        "index": 2,
        "llm_factory": lambda temp: ChatOpenAI(
            model="gpt-5-chat-latest",
            temperature=1.0,
        )
    }
}

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
        "system_prompt": "あなたは優秀なAIアシスタントです。",
        "temperature": 1.0,
        "error_message": "",
        "model_index": 0,
        "chat_history": [],
        "model": "claude-sonnet-4.5",
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

def get_current_provider() -> str:
    """現在のモデルのプロバイダを取得"""
    model = st.session_state.get("model", "claude-sonnet-4.5")
    return MODEL_CONFIG.get(model, MODEL_CONFIG["claude-sonnet-4.5"])["provider"]

def copy_button(text: str, key_suffix: Union[int, str]) -> None:
    copy_button = txt_copy(label="copy", text_to_copy=text.replace("\\n", "\n"), key=f"text_clipboard_chat_{key_suffix}")
    if copy_button:
        st.toast("Copied!")

def check_token() -> bool:
    token_limit = 50000
    message_limit = 30
    def limit_error(msg: str) -> bool:
        st.error(msg, icon="🚨")
        st.session_state.done = True
        st.session_state.save = False
        return False
    
    if st.session_state.total_tokens > token_limit:
        persent=min(100, math.floor(100 * st.session_state.total_tokens / token_limit))
        return limit_error(f'Error: Text volume is {persent}% of the limit.  \nPlease delete unnecessary parts or reset the conversation')
    if len(st.session_state.chat_history) > message_limit:
        return limit_error('Error: Conversation limit exceeded. Please reset the conversation')
    return True

def clear_chat():
    st.session_state.chat_history = []
    st.session_state.total_tokens = 0
    st.session_state.done = True
    st.session_state.error_message = ""
    st.session_state.edit_states = {}
    st.session_state.reasoning = ""  # 推論過程もクリア
    st.rerun()

def create_new_conversation() -> None:
    """新しい会話セッションを開始（DBには最初のメッセージ送信時に作成）"""
    st.session_state.current_conversation_id = None
    st.session_state.chat_history = []
    st.session_state.reasoning = ""
    st.session_state.total_tokens = 0
    st.session_state.error_message = ""
    st.session_state.edit_states = {}
    st.rerun()

def load_conversation(conversation_id: str) -> None:
    """既存の会話を読み込む"""
    # 会話が存在するか確認
    conv = db.get_conversation(conversation_id)
    if not conv or conv["is_deleted"]:
        st.error("会話が見つかりません")
        return
    
    # メッセージを読み込み
    messages = db.get_messages(conversation_id)
    
    st.session_state.current_conversation_id = conversation_id
    st.session_state.chat_history = messages
    st.session_state.reasoning = db.get_last_reasoning(conversation_id)
    st.session_state.total_tokens = db.get_conversation_tokens(conversation_id)
    st.session_state.error_message = ""
    st.session_state.edit_states = {}
    st.rerun()

def delete_current_conversation() -> None:
    """現在の会話を削除（論理削除）"""
    if st.session_state.current_conversation_id:
        db.delete_conversation(st.session_state.current_conversation_id)
        st.session_state.current_conversation_id = None
        st.session_state.chat_history = []
        st.session_state.reasoning = ""
        st.rerun()

def generate_title_from_message(message: str) -> str:
    """最初のメッセージから会話タイトルをLLMで生成"""
    # メッセージからテキストを抽出
    text_content = ""
    if isinstance(message, list):
        # 画像付きメッセージの場合、テキスト部分を抽出
        for item in message:
            if isinstance(item, dict) and item.get("type") == "text":
                text_content = item["text"]
                break
        if not text_content:
            return "画像付き会話"
    else:
        text_content = message
    
    # フォールバック用の簡易タイトル生成関数
    def fallback_title(text: str) -> str:
        if len(text) > 15:
            return text[:15] + "..."
        return text
    
    # LLMでタイトル生成を試みる
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            thinking_budget=0,
        )
        
        prompt = f"""以下のメッセージに対して、15文字以内の簡潔な会話タイトルを生成してください。
タイトルのみを出力し、句読点や記号は不要です。

メッセージ: {text_content[:200]}

タイトル:"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        title = response.content.strip()
        
        # 15文字を超える場合は切り詰め
        if len(title) > 15:
            title = title[:15] + "..."
        
        # 空の場合はフォールバック
        if not title:
            return fallback_title(text_content)
        
        return title
        
    except Exception as e:
        # エラー時は従来の方式にフォールバック
        return fallback_title(text_content)

def update_system_prompt():
    st.session_state.system_prompt = st.session_state.new_system_prompt

def update_model():
    """モデル切り替え時にLLMインスタンスとインデックスを更新"""
    model_name = st.session_state.model
    config = MODEL_CONFIG.get(model_name)
    if config:
        st.session_state.llm = config["llm_factory"]
        st.session_state.model_index = config["index"]

def on_stop() -> None:
    """stop押下時に停止フラグを立てる。streamlitの仕様上ループ中断後の処理は実行されないのでstate、chat_history更新する。"""
    response = st.session_state.get("response", "")
    if response:
        st.session_state.chat_history.append(("assistant", response))
    st.session_state.stop = True
    st.session_state.done = True
    st.session_state.save = False

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
        create_new_conversation()
    
    # 会話削除ボタン（現在の会話がある場合のみ表示）
    if st.session_state.current_conversation_id:
        if st.button(":material/delete: この会話を削除", use_container_width=True):
            delete_current_conversation()
    
    conversations = db.get_conversations(st.session_state.user_id)
    
    if conversations:
        for conv in conversations:
            conv_id = conv["id"]
            title = conv["title"]
            updated_at = conv["updated_at"]
            
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
                load_conversation(conv_id)
    
def modify_message(messages, i):
    del messages[i:]
    return messages

def render_markdown(text: str) -> str:
    """
    テキストをMarkdown表示用にエスケープする。
    改行を<br>に変換し、特殊文字をエスケープする。
    """
    return text.replace("\n", "<br>").replace("$", "\\$").replace("#", "\\#").replace("_", "\\_")

def render_uploaded_images(files: List[Dict]) -> None:
    """
    アップロードされた画像ファイルを表示する。
    """
    for file in files:
        if file.get("type", "").startswith("image/"):
            try:
                base64_data = file['data'].split(',')[1] if ',' in file['data'] else file['data']
                image_bytes = base64.b64decode(base64_data)
                st.image(image_bytes, caption=file['name'], width=200)
            except (ValueError, base64.binascii.Error, KeyError) as e:
                st.warning(f"画像 '{file.get('name', 'unknown')}' の表示に失敗しました")
                st.write(f"📎 {file.get('name', 'unknown')}")

def build_prompt_template(image_urls: Optional[List[str]] = None) -> ChatPromptTemplate:
    """
    プロンプトテンプレートを構築する。
    image_urls が None または空の場合はテキストのみ、それ以外は画像付きテンプレートを返す。
    """
    if image_urls:
        # 画像付きの場合
        human_content: List[Dict[str, Any]] = [{"type": "text", "text": "{input}"}]
        for url in image_urls:
            human_content.append({
                "type": "image_url",
                "image_url": {"url": url}
            })
        return ChatPromptTemplate.from_messages(
            [
                ("system", st.session_state.system_prompt),
                MessagesPlaceholder(variable_name="conversation"),
                ("human", human_content),
            ]
        )
    else:
        # テキストのみの場合
        return ChatPromptTemplate.from_messages(
            [
                ("system", st.session_state.system_prompt),
                MessagesPlaceholder(variable_name="conversation"),
                ("human", "{input}"),
            ]
        )

def build_chain(prompt_template: ChatPromptTemplate):
    """プロンプトテンプレートからチェーンを構築する。"""
    # LLMインスタンスを取得
    llm_instance = st.session_state.llm(st.session_state.temperature)
    
    provider = get_current_provider()
    if provider == "openai":
        llm_instance = llm_instance.bind_tools([{"type": "web_search"}])
    elif provider == "google":
        llm_instance = llm_instance.bind_tools([GenAITool(google_search={})])
    elif provider == "anthropic":
        llm_instance = llm_instance.bind_tools([{"type": "web_search_20250305","name": "web_search", "max_uses": 5}])
    
    return (prompt_template | llm_instance).with_config({"run_name": "Chat", "tags": ["Chat"]})

def stream_response_anthropic(chain, input_text: str, conversation_history: List[Tuple[str, Union[str, List[Dict[str, Any]]]]], status_container, reasoning_placeholder, message_placeholder) -> Tuple[str, int]:
    """
    Anthropic用のストリーミング処理（thinking対応）
    """
    st.session_state.response = ""
    st.session_state.reasoning = ""
    total_tokens: int = 0
    last_usage: Optional[Dict[str, Any]] = None
    
    for chunk in chain.stream({"input": input_text, "conversation": conversation_history}):
        if st.session_state.stop:
            break
        if isinstance(chunk.content, list) and len(chunk.content) > 0:
            content_item = chunk.content[0]
            if content_item.get("thinking"):
                status_container.update(label="AIは考えています...", state="running", expanded=True)
                st.session_state.reasoning += content_item["thinking"]
                reasoning_placeholder.markdown(st.session_state.reasoning.replace("\n", "  \n"), unsafe_allow_html=True)
            elif content_item.get("text"):
                status_container.update(label="出力中...", state="running", expanded=False)
                st.session_state.response += content_item["text"]
                message_placeholder.markdown(st.session_state.response.replace("\n", "  \n") + "▌", unsafe_allow_html=True)
        try:
            if getattr(chunk, "usage_metadata", None):
                last_usage = chunk.usage_metadata
        except Exception:
            pass
    
    if last_usage:
        total_tokens = last_usage.get("total_tokens", 0)
    
    status_container.update(label="完了", state="complete", expanded=False)
    message_placeholder.markdown(st.session_state.response.replace("\n", "  \n"))
    return st.session_state.response, total_tokens

def stream_response_google(chain, input_text: str, conversation_history: List[Tuple[str, Union[str, List[Dict[str, Any]]]]], status_container, reasoning_placeholder, message_placeholder) -> Tuple[str, int]:
    """
    Google用のストリーミング処理（thinking対応）
    """
    st.session_state.response = ""
    st.session_state.reasoning = ""
    total_tokens: int = 0
    
    for chunk in chain.stream({"input": input_text, "conversation": conversation_history}):
        if st.session_state.stop:
            break
        if isinstance(chunk.content, list) and len(chunk.content) > 0:
            content_item = chunk.content[0]
            if content_item.get("thinking"):
                status_container.update(label="AIは考えています...", state="running", expanded=True)
                st.session_state.reasoning += content_item["thinking"]
                reasoning_placeholder.markdown(st.session_state.reasoning.replace("\n", "  \n"), unsafe_allow_html=True)
        else:
            status_container.update(label="出力中...", state="running", expanded=False)
            st.session_state.response += chunk.content
            message_placeholder.markdown(st.session_state.response.replace("\n", "  \n") + "▌", unsafe_allow_html=True)
        try:
            total_tokens += (chunk.usage_metadata or {}).get("total_tokens", 0)
        except Exception:
            pass
    
    status_container.update(label="完了", state="complete", expanded=False)
    message_placeholder.markdown(st.session_state.response.replace("\n", "  \n"))
    return st.session_state.response, total_tokens

def stream_response_openai(chain, input_text: str, conversation_history: List[Tuple[str, Union[str, List[Dict[str, Any]]]]], status_container, message_placeholder) -> Tuple[str, int]:
    """
    OpenAI用のストリーミング処理（非推論モデル用）
    """
    st.session_state.response = ""
    st.session_state.reasoning = ""
    total_tokens: int = 0
    first_chunk_received = False
    
    # 最初は「AIは考えています...」
    status_container.update(label="AIは考えています...", state="running", expanded=False)
    
    for chunk in chain.stream({"input": input_text, "conversation": conversation_history}):
        if st.session_state.stop:
            break
        
        # 最初のチャンクを受け取ったら「出力中」に変更
        if not first_chunk_received:
            first_chunk_received = True
            status_container.update(label="出力中...", state="running", expanded=False)
            
        if isinstance(chunk.content, list) and len(chunk.content) > 0:
            content_item = chunk.content[0]
            if content_item.get("text"):
                st.session_state.response += content_item["text"]
                message_placeholder.markdown(st.session_state.response.replace("\n", "  \n") + "▌", unsafe_allow_html=True)
        
        # トークン数の取得
        try:
            if getattr(chunk, "usage_metadata", None):
                usage = chunk.usage_metadata
                total_tokens = usage.get("total_tokens", 0)
        except Exception:
            pass
    
    # 完了
    status_container.update(label="完了", state="complete", expanded=False)
    message_placeholder.markdown(st.session_state.response.replace("\n", "  \n"))
    return st.session_state.response, total_tokens

def run_chat_turn(
    prompt: str, 
    conversation_history: List[Tuple[str, Union[str, List[Dict[str, Any]]]]], 
    image_urls: Optional[List[str]] = None
) -> Tuple[str, int]:
    """
    チャットターンのストリーミング実行を行う共通関数（推論対応）
    
    Args:
        prompt: ユーザー入力テキスト
        conversation_history: これまでの会話履歴
        image_urls: 画像のdata URI リスト（Noneの場合はテキストのみ）
    
    Returns:
        (response_text, total_tokens): アシスタントの応答とトークン数
    """
    # プロンプトテンプレート構築
    prompt_template = build_prompt_template(image_urls)
    
    # チェーン構築
    chain = build_chain(prompt_template)
    
    # プロバイダ取得
    provider = get_current_provider()
    
    # ストリーミング実行とUI表示
    st.session_state.response = ""
    st.session_state.reasoning = ""
    
    with st.chat_message("assistant", avatar=":material/psychology:"):
        col1, col2 = st.columns([9, 1])
        with col2:
            _pressed = st.button("stop", on_click=on_stop)
            st.session_state.stop = _pressed
        with col1:
            if provider == "anthropic":
                with st.status(label="メッセージを送信", state="complete", expanded=False) as status_container:
                    reasoning_placeholder = st.empty()
                message_placeholder = st.empty()
                response, tokens = stream_response_anthropic(
                    chain=chain,
                    input_text=prompt,
                    conversation_history=conversation_history,
                    status_container=status_container,
                    reasoning_placeholder=reasoning_placeholder,
                    message_placeholder=message_placeholder,
                )
            elif provider == "google":
                with st.status(label="AIは考えています...", state="running", expanded=False) as status_container:
                    reasoning_placeholder = st.empty()
                message_placeholder = st.empty()
                response, tokens = stream_response_google(
                    chain=chain,
                    input_text=prompt,
                    conversation_history=conversation_history,
                    status_container=status_container,
                    reasoning_placeholder=reasoning_placeholder,
                    message_placeholder=message_placeholder,
                )
            elif provider == "openai":
                status_container = st.status(label="AIは考えています...", state="running", expanded=False)
                message_placeholder = st.empty()
                response, tokens = stream_response_openai(
                    chain=chain,
                    input_text=prompt,
                    conversation_history=conversation_history,
                    status_container=status_container,
                    message_placeholder=message_placeholder,
                )
            else:
                # フォールバック
                message_placeholder = st.empty()
                message_placeholder.markdown("Unknown provider")
                response, tokens = "", 0
    
    return response, tokens

def render_human_message(message: Tuple[str, Union[str, List[Dict[str, Any]]]], index: int, edit: bool) -> None:
    """
    Render user-side messages.
    """
    
    with st.chat_message("human", avatar=":material/mood:"):
        col1, col2 = st.columns([9, 1])
        with col1:
            if isinstance(message[1], list):
                for item in message[1]:
                    if item["type"] == "text":
                        msg_content = item["text"]
                        st.markdown(render_markdown(msg_content), unsafe_allow_html=True)
                    elif item["type"] == "image_url":
                        st.image(item["image_url"]["url"])
            else:
                msg_content = message[1]
                st.markdown(render_markdown(msg_content), unsafe_allow_html=True)
                
        with col2:
            if edit:
                if st.button("edit", key=f"edit_{index}"):
                    st.session_state.edit_states[index] = True
            else:
                st.button("edit", key=f"dummy_{index}")
                
        if edit and st.session_state.edit_states.get(index):
            st.session_state.new_message = st.text_area("編集したらsaveしてください。", value=msg_content, key=f"new_message_{index}")
            left, right = st.columns([9, 1])
            with right:
                if st.button("save", key=f"save_{index}", type="primary"):
                    st.session_state.edit_states[index] = False
                    
                    # DBから該当インデックス以降のメッセージを削除
                    if st.session_state.current_conversation_id:
                        db.delete_messages_from_index(
                            st.session_state.current_conversation_id, 
                            index
                        )
                    
                    # session_stateからも削除
                    modify_message(st.session_state.chat_history, index)
                    st.session_state.save = True

def render_assistant_message(message: Tuple[str, str], index: int, show_copy_button: bool) -> None:
    """
    Render assistant-side messages.
    """
    col1, col2 = st.columns([9, 1])
    with col1:
        with st.chat_message("assistant", avatar=":material/psychology:"):
            # 最新のメッセージで推論過程がある場合は表示
            if index == len(st.session_state.chat_history) - 1 and hasattr(st.session_state, "reasoning") and st.session_state.reasoning:
                with st.expander("AIの思考プロセス", expanded=False):
                    st.markdown(st.session_state.reasoning.replace("\n", "  \n"), unsafe_allow_html=True)
            st.markdown(message[1].replace("\n","  \n"), unsafe_allow_html=True)
    with col2:
        if show_copy_button and index == len(st.session_state.chat_history) - 1:
            copy_button(message[1], index)

def show_chat_history(
    messages: List[Tuple[str, Union[str, List[Dict[str, Any]]]]],
    edit: bool,
    error_message: str,
    new_message: Optional[str] = None,
    show_copy_button: bool = True,
) -> None:
    """
    Display the entire chat history and render new messages as needed.
    """
    for i, message in enumerate(messages):
        if message[0] == "human":
            render_human_message(message, i, edit)
        elif message[0] == "assistant":
            render_assistant_message(message, i, show_copy_button)
    if new_message:
        with st.chat_message("user", avatar=":material/mood:"):
            st.markdown(render_markdown(new_message), unsafe_allow_html=True)
    if error_message:
        st.error(f"エラーが発生しました。  \n{st.session_state.error_message}。  \nモデルを変更するか再度試してみてください。",icon=":material/warning:")

st.title("Streamlit ChatBot")

user_input = multimodal_chat_input(
    placeholder="Send a message",
    enable_voice_input=True,
    voice_recognition_method="openai_whisper",
    voice_language="ja-JP",
    key="chat_input"
)

show_chat_history(messages=st.session_state.chat_history,edit=True, error_message=st.session_state.error_message, show_copy_button=True)

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
                    st.markdown(render_markdown(input_text), unsafe_allow_html=True)
                
                # Display images
                render_uploaded_images(input_files)
        
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
            title = generate_title_from_message(human_content)
            db.update_conversation_title(conversation_id, title)
        
        # Execute chat turn
        response, tokens = run_chat_turn(
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
    show_chat_history(messages=st.session_state.chat_history, edit=False, error_message=st.session_state.error_message, new_message=prompt, show_copy_button=False)
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
        response, tokens = run_chat_turn(
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

