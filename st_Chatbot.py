import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from st_txt_copybutton import txt_copy
import math
from typing import Tuple, Union, List, Dict, Any, Optional
import base64
from st_chat_input_multimodal import multimodal_chat_input

st.set_page_config(layout="wide", page_title="chat bot",page_icon=":material/chat:")

def get_current_provider() -> str:
    model = st.session_state.get("model", "")
    if model == "gemini-2.5-pro":
        return "google"
    if model == "claude-sonnet-4":
        return "anthropic"
    if model == "gpt-4.1":
        return "openai"
    return "openai"

if not hasattr(st.session_state, "done"):
    st.session_state.done = True

attrs=["Clear","save","stop"]
for attr in attrs:
    if attr not in st.session_state:
        st.session_state[attr] = False

if "edit_states" not in st.session_state:
    st.session_state.edit_states = {}

if not hasattr(st.session_state, "total_tokens"):
    st.session_state.total_tokens = 0

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are an excellent AI assistant."

if not hasattr(st.session_state, "temperature"):
    st.session_state.temperature = 0.7

if "error_message" not in st.session_state:
    st.session_state.error_message = ""

if "model_index" not in st.session_state:
    st.session_state.model_index = 1
    st.session_state.llm = ChatAnthropic(temperature=st.session_state.temperature, model_name="claude-sonnet-4-20250514",max_tokens=4096,timeout=120,max_retries=3)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def copy_button(text: str, key_suffix: Union[int, str]) -> None:
    copy_button = txt_copy(label="copy", text_to_copy=text.replace("\\n", "\n"), key=f"text_clipboard_chat_{key_suffix}")
    if copy_button:
        st.toast("Copied!")

def check_token() -> bool:
    token_limit = 50000
    message_limit = 30
    def limit_error(msg: str) -> bool:
        st.error(msg, icon="🚨")
        st.session_state.Clear = True
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
    st.session_state.Clear = False
    st.session_state.total_tokens = 0
    st.session_state.done = True
    st.session_state.error_message = ""
    st.session_state.edit_states = {}
    st.rerun()

def update_system_prompt():
    st.session_state.system_prompt = st.session_state.new_system_prompt

def update_temperature():
    st.session_state.temperature = st.session_state.new_temperature

def update_model():
    if st.session_state.model == "gpt-4.1":
        st.session_state.llm = ChatOpenAI(model = "gpt-4.1",temperature=st.session_state.temperature)
        st.session_state.model_index = 0
    elif st.session_state.model == "claude-sonnet-4":
        st.session_state.llm = ChatAnthropic(temperature=st.session_state.temperature, model_name="claude-sonnet-4-20250514",max_tokens=4096,timeout=120,max_retries=3)
        st.session_state.model_index = 1
    elif st.session_state.model == "gemini-2.5-pro":
        st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro",temperature=st.session_state.temperature)
        st.session_state.model_index = 2

def on_stop() -> None:
    """stop押下時に停止フラグを立てる。streamlitの仕様上ループ中断後の処理は実行されないのでstate、chat_history更新する。"""
    response = st.session_state.get("response", "")
    if response:
        st.session_state.chat_history.append(("assistant", response))
    st.session_state.stop = True
    st.session_state.done = True
    st.session_state.Clear = True
    st.session_state.save = False

with st.sidebar.container():
    st.selectbox("model",
                 ("gpt-4.1","claude-sonnet-4","gemini-2.5-pro"),
                 help="You can select the model.",index=st.session_state.model_index,key="model",on_change=update_model)
    st.text_area("system prompt",value=st.session_state.system_prompt,on_change=update_system_prompt,key="new_system_prompt",
                                 help="You can provide a prompt to the system. This is only effective at the first message transmission.")
    st.slider(label="temperature",min_value=0.0, max_value=1.0,on_change=update_temperature,key="new_temperature",
                            value=st.session_state.temperature,help="Controls the randomness of the generated text.")
    
def modify_message(messages, i):
    del messages[i:]
    return messages

def build_prompt_template_text_only() -> ChatPromptTemplate:
    """
    テキストのみのプロンプトテンプレートを構築する。
    system -> conversation -> human("{input}") の順。
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", st.session_state.system_prompt),
            MessagesPlaceholder(variable_name="conversation"),
            ("human", "{input}"),
        ]
    )

def build_prompt_template_with_images(image_urls: List[str]) -> ChatPromptTemplate:
    """
    画像付きのプロンプトテンプレートを構築する。
    human は [{text:{"{input}"}}, {image_url: {url: data-uri}}, ...] のリストにする。
    """
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

def build_chain(prompt_template: ChatPromptTemplate):
    """プロンプトテンプレートからチェーンを構築する。"""
    return (prompt_template | st.session_state.llm).with_config({"run_name": "Chat", "tags": ["Chat"]})

def stream_response(chain, input_text: str, conversation_history: List[Tuple[str, Union[str, List[Dict[str, Any]]]]], provider: str, message_placeholder) -> Tuple[str, int]:
    """
    共通のストリーミング処理。
    - チャンク結合
    - stop 押下チェック
    - トークン集計（google は逐次、他は最終チャンク）
    """
    st.session_state.response = ""
    total_tokens: int = 0
    last_chunk: Any = None
    for chunk in chain.stream({"input": input_text, "conversation": conversation_history}):
        last_chunk = chunk
        if not st.session_state.stop:
            st.session_state.response += chunk.content
            message_placeholder.markdown(st.session_state.response.replace("\n", "  \n") + "▌", unsafe_allow_html=True)
        if provider == "google":
            total_tokens += (getattr(chunk, "usage_metadata", {}) or {}).get("total_tokens", 0)
    if provider != "google" and last_chunk is not None:
        try:
            total_tokens = last_chunk.usage_metadata.get("total_tokens", 0)
        except Exception:
            pass
    message_placeholder.markdown(st.session_state.response.replace("\n", "  \n"))
    return st.session_state.response, total_tokens

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
                        st.markdown(msg_content.replace("\n", "<br>").replace("$", "\\$").replace("#", "\\#").replace("_", "\\_"),unsafe_allow_html=True)
                    elif item["type"] == "image_url":
                        st.image(item["image_url"]["url"])
            else:
                msg_content = message[1]
                st.markdown(msg_content.replace("\n", "<br>").replace("$", "\\$").replace("#", "\\#").replace("_", "\\_"),unsafe_allow_html=True)
                
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
                    modify_message(st.session_state.chat_history, index)
                    st.session_state.save = True

def render_assistant_message(message: Tuple[str, str], index: int, show_copy_button: bool) -> None:
    """
    Render assistant-side messages.
    """
    col1, col2 = st.columns([9, 1])
    with col1:
        with st.chat_message("assistant", avatar=":material/psychology:"):
            st.markdown(message[1].replace("\n","  \n"), unsafe_allow_html=True)
    with col2:
        if show_copy_button and index == len(st.session_state.chat_history) - 1:
            copy_button(st.session_state.response, index)

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
            st.markdown(new_message.replace("\n", "<br>").replace("$", "\\$").replace("#", "\\#").replace("_", "\\_"),unsafe_allow_html=True)
    if error_message:
        st.error(f"エラーが発生しました。  \n{st.session_state.error_message}。  \nモデルを変更するか再度試してみてください。",icon=":material/warning:")

st.title("Streamlit Chatbot")

st.write("**You can converse with the selected model. You can pause the conversation midway and edit the conversation history.**")

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
        # Extract text from multimodal input
        input_text = user_input.get("text", "")
        input_files = user_input.get("files", [])
        
        # Create message content for LangChain
        message_content = []
        if input_text:
            message_content.append({"type": "text", "text": input_text})
        
        # Handle image files
        for file in input_files:
            if file.get("type", "").startswith("image/"):
                try:
                    base64_data = file['data'].split(',')[1] if ',' in file['data'] else file['data']
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": file['data']}
                    })
                except:
                    pass
        
        # Use text content for LangChain input
        llm_input = input_text if input_text else "Image uploaded"
        
        with st.chat_message("human",avatar = ":material/mood:"):
            col1,  col2 = st.columns([9,  1])
            with col1:
                if input_text:
                    st.markdown(input_text.replace("\n", "<br>").replace("$", "\\$").replace("#", "\\#").replace("_", "\\_"),unsafe_allow_html=True)
                
                # Display images
                for file in input_files:
                    if file.get("type", "").startswith("image/"):
                        try:
                            base64_data = file['data'].split(',')[1] if ',' in file['data'] else file['data']
                            image_bytes = base64.b64decode(base64_data)
                            st.image(image_bytes, caption=file['name'], width=200)
                        except:
                            st.write(f"📎 {file['name']}")

            # Create prompt based on whether images are included (unified)
            image_urls: List[str] = []
            for file in input_files:
                if file.get("type", "").startswith("image/"):
                    image_urls.append(file["data"])  # expect data URI

            if image_urls:
                prompt_template = build_prompt_template_with_images(image_urls)
            else:
                prompt_template = build_prompt_template_text_only()

        chain = build_chain(prompt_template)

        st.session_state.response = ""
        if image_urls:
            human_payload: List[Dict[str, Any]] = []
            if input_text:
                human_payload.append({"type": "text", "text": input_text})
            for url in image_urls:
                human_payload.append({"type": "image_url", "image_url": {"url": url}})
            st.session_state.chat_history.append(("human", human_payload))
        else:
            st.session_state.chat_history.append(("human", input_text))
        provider = get_current_provider()
        with st.chat_message("assistant",avatar = ":material/psychology:"):
            col1,  col2 = st.columns([9,  1])
            with col1:
                message_placeholder = st.empty()
                message_placeholder.markdown("thinking...")
            with col2:
                _pressed = st.button("stop", on_click=on_stop)
                st.session_state.stop = _pressed
            with col1:
                st.session_state.total_tokens = 0
                st.session_state.response, st.session_state.total_tokens = stream_response(
                    chain=chain,
                    input_text=llm_input,
                    conversation_history=st.session_state.chat_history[:-1],
                    provider=provider,
                    message_placeholder=message_placeholder,
                )
                st.session_state.chat_history.append(("assistant", st.session_state.response))
        st.session_state.done = True
        st.session_state.Clear=True
        st.rerun()

if st.session_state.save:
    st.session_state.error_message = ""
    st.session_state.done = False
    prompt = st.session_state.new_message
    show_chat_history(messages=st.session_state.chat_history,edit=False,error_message=st.session_state.error_message,new_message=prompt, show_copy_button=False)
    ok = check_token()
    if not ok:
        st.session_state.save = False
    else:
        prompt_template = build_prompt_template_text_only()
        chain = build_chain(prompt_template)
        st.session_state.response = ""
        st.session_state.chat_history.append(("human", prompt))
        provider = get_current_provider()
        with st.chat_message("assistant",avatar = ":material/psychology:"):
            col1,  col2 = st.columns([9,  1])
            with col1:
                message_placeholder = st.empty()
                message_placeholder.markdown("thinking...")
            with col2:
                _pressed = st.button("stop", on_click=on_stop)
                st.session_state.stop = _pressed
            with col1:
                # Append human first, then stream with history excluding last human
                st.session_state.total_tokens = 0
                st.session_state.response, st.session_state.total_tokens = stream_response(
                    chain=chain,
                    input_text=prompt,
                    conversation_history=st.session_state.chat_history[:-1],
                    provider=provider,
                    message_placeholder=message_placeholder,
                )
                st.session_state.chat_history.append(("assistant", st.session_state.response))
        st.session_state.done = True
        st.session_state.Clear=True
        st.session_state.save = False
        st.rerun()

if st.session_state.Clear:
    left, spacer, right = st.columns([1, 3, 1])
    with right:
        button_clear_chat = st.button("clear chat history",type="primary")
        if button_clear_chat:
            clear_chat()
