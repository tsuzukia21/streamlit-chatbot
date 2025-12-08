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
    model = st.session_state.get("model", "claude-opus-4.5")
    return MODEL_CONFIG.get(model, MODEL_CONFIG["claude-opus-4.5"])["provider"]

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

def stream_response(chain, input_text: str, conversation_history: List[Tuple[str, Union[str, List[Dict[str, Any]]]]], status_container, reasoning_placeholder, message_placeholder) -> Tuple[str, int]:
    """
    LangChain v1: content_blocksによるプロバイダー統一ストリーミング処理
    """
    st.session_state.response = ""
    st.session_state.reasoning = ""
    last_usage: Optional[Dict[str, Any]] = None
    
    for chunk in chain.stream({"input": input_text, "conversation": conversation_history}):
        if st.session_state.stop:
            break
        
        if hasattr(chunk, 'content_blocks') and chunk.content_blocks:
            for block in chunk.content_blocks:
                if block["type"] == "reasoning":
                    reasoning_text = block.get("reasoning") or ""
                    if not reasoning_text and block.get("summary"):
                        reasoning_text = "\n".join(str(s) for s in block["summary"])
                    if reasoning_text:
                        status_container.update(label="AIは考えています...", state="running", expanded=True)
                        st.session_state.reasoning += reasoning_text
                        reasoning_placeholder.markdown(st.session_state.reasoning.replace("\n", "  \n"), unsafe_allow_html=True)
                elif block["type"] == "text":
                    status_container.update(label="出力中...", state="running", expanded=False)
                    st.session_state.response += block["text"]
                    message_placeholder.markdown(st.session_state.response.replace("\n", "  \n") + "▌", unsafe_allow_html=True)
        
        try:
            if getattr(chunk, "usage_metadata", None):
                last_usage = chunk.usage_metadata
        except Exception:
            pass
    
    total_tokens = last_usage.get("total_tokens", 0) if last_usage else 0
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
    
    # ストリーミング実行とUI表示
    st.session_state.response = ""
    st.session_state.reasoning = ""
    
    with st.chat_message("assistant", avatar=":material/psychology:"):
        col1, col2 = st.columns([9, 1])
        with col2:
            _pressed = st.button("stop", on_click=on_stop)
            st.session_state.stop = _pressed
        with col1:
            # LangChain v1: content_blocksによるプロバイダー統一処理
            with st.status(label="メッセージを送信", state="complete", expanded=False) as status_container:
                reasoning_placeholder = st.empty()
            message_placeholder = st.empty()
            response, tokens = stream_response(
                chain=chain,
                input_text=prompt,
                conversation_history=conversation_history,
                status_container=status_container,
                reasoning_placeholder=reasoning_placeholder,
                message_placeholder=message_placeholder,
            )
    
    return response, tokens