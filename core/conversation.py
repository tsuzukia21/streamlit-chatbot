import streamlit as st
import core.database as db
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

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
