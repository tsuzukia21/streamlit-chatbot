import streamlit as st
from core.MODEL_CONFIG import MODEL_CONFIG
from typing import List, Dict, Any, Optional, Tuple, Union
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool

def on_stop() -> None:
    """stop押下時に停止フラグを立てる。streamlitの仕様上ループ中断後の処理は実行されないのでstate、chat_history更新する。"""
    response = st.session_state.get("response", "")
    if response:
        st.session_state.chat_history.append(("assistant", response))
    st.session_state.stop = True
    st.session_state.done = True
    st.session_state.save = False

def get_current_provider() -> str:
    """現在のモデルのプロバイダを取得"""
    model = st.session_state.get("model", "claude-sonnet-4.5")
    return MODEL_CONFIG.get(model, MODEL_CONFIG["claude-sonnet-4.5"])["provider"]

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