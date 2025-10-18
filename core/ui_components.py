import streamlit as st
import base64
from typing import List, Dict, Any, Optional, Tuple, Union
import core.database as db
from st_txt_copybutton import txt_copy

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

def copy_button(text: str, key_suffix: Union[int, str]) -> None:
    copy_button = txt_copy(label="copy", text_to_copy=text.replace("\\n", "\n"), key=f"text_clipboard_chat_{key_suffix}")
    if copy_button:
        st.toast("Copied!")