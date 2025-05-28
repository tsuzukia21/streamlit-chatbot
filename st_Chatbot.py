import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from st_txt_copybutton import txt_copy
import tiktoken
import math
from tiktoken.core import Encoding

encoding: Encoding = tiktoken.encoding_for_model("gpt-4o")
st.set_page_config(layout="wide", page_title="chat bot",page_icon=":material/chat:")

if not hasattr(st.session_state, "done"):
    st.session_state.done = True
attrs=["clear_button","edit","save","stop"]
for attr in attrs:
    if attr not in st.session_state:
        st.session_state[attr] = False
if not hasattr(st.session_state, "total_tokens"):
    st.session_state.total_tokens = 0
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are an excellent AI assistant."
if not hasattr(st.session_state, "temperature"):
    st.session_state.temperature = 0.7
if "model_index" not in st.session_state:
    st.session_state.model_index = 1
    st.session_state.llm = ChatAnthropic(temperature=st.session_state.temperature, model_name="claude-sonnet-4-20250514",max_tokens=4096,timeout=120,max_retries=3)

def copy_button(text):
    copy_button = txt_copy(label="copy", text_to_copy=text.replace("\\n", "\n"), key="text_clipboard")
    if copy_button:
        st.toast("Copied!")

def check_token():
    if st.session_state.total_tokens>50000:
        persent=math.floor(st.session_state.total_tokens/500)
        if st.button('clear chat history'):
            clear_chat()
        st.error(f'Error: Text volume is {persent}% of the limit.  \nPlease delete unnecessary parts or reset the conversation', icon="🚨")
        st.stop()
    if len(st.session_state.chat_history) > 30:
        if st.button('clear chat history'):
            clear_chat()
        st.error('Error: Conversation limit exceeded. Please reset the conversation', icon="🚨")
        st.stop()

def clear_chat():
    st.session_state.chat_history = []
    st.session_state.clear_button=False
    st.session_state.total_tokens=0
    st.session_state.done = True
    st.rerun()

def update_system_prompt():
    st.session_state.system_prompt = st.session_state.new_system_prompt

def update_temperature():
    st.session_state.temperature = st.session_state.new_temperature

def update_model():
    if st.session_state.model == "gpt-4.1":
        if openai_api_key == "":
            st.toast("Please enter the OpenAI API Key.",icon=":material/error:")
        st.session_state.llm = ChatOpenAI(model = "gpt-4.1",temperature=st.session_state.temperature,api_key=openai_api_key)
        st.session_state.model_index = 0
    elif st.session_state.model == "claude-sonnet-4":
        if anthropic_api_key =="":
            st.toast("Please enter the Claude API Key.",icon=":material/error:")
        st.session_state.llm = ChatAnthropic(temperature=st.session_state.temperature, model_name="claude-sonnet-4-20250514",max_tokens=4096,timeout=120,max_retries=3,api_key=anthropic_api_key)
        st.session_state.model_index = 1
    elif st.session_state.model == "gemini-2.5-pro":
        if google_api_key == "":
            st.toast("Please enter the Google API Key.",icon=":material/error:")
        st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-preview-05-06",temperature=st.session_state.temperature,google_api_key=google_api_key)
        st.session_state.model_index = 2
    elif st.session_state.model == "gpt-4.5-preview":
        if openai_api_key == "":
            st.toast("Please enter the OpenAI API Key.",icon=":material/error:")
        st.session_state.llm = ChatOpenAI(model = "gpt-4.5-preview",temperature=st.session_state.temperature,api_key=openai_api_key)
        st.session_state.model_index = 3

with st.sidebar.container():
    st.selectbox("model",
                 ("gpt-4.1","claude-sonnet-4","gemini-2.5-pro","gpt-4.5-preview"),
                 help="You can select the model.",index=st.session_state.model_index,key="model",on_change=update_model)
    st.text_area("system prompt",value=st.session_state.system_prompt,on_change=update_system_prompt,key="new_system_prompt",
                                 help="You can provide a prompt to the system. This is only effective at the first message transmission.")
    st.slider(label="temperature",min_value=0.0, max_value=1.0,on_change=update_temperature,key="new_temperature",
                            value=st.session_state.temperature,help="Controls the randomness of the generated text.")
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    google_api_key = st.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
    anthropic_api_key = st.text_input("Claude API Key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))

st.markdown(
    """
    <style>
    /* Target buttons outside the sidebar */
    div.stButton>button:first-child {
        float: right;
    }
    div[data-testid="stButtonGroup"] {
        display: flex;
        justify-content: flex-end;
    }
    /* Reset styles for sidebar buttons */
    [data-testid="stSidebar"] div.stButton>button:first-child {
        float: none;
    }
    [data-testid="stSidebar"] div[data-testid="stButtonGroup"] {
        justify-content: flex-start;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Streamlit Chatbot")

st.write("**You can converse with the selected model. You can pause the conversation midway and edit the conversation history.**")

user_input = st.chat_input(
    "Send a message",
    accept_file=False,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
def modify_message(messages, i):
    del messages[i:]
    return messages

def render_human_message(message, index, edit):
    """
    Renders the user's message.
    """
    if isinstance(message[1], list):
        for item in message[1]:
            if item["type"] == "text":
                st.session_state.total_tokens += len(encoding.encode(item["text"]))
    else:
        st.session_state.total_tokens += len(encoding.encode(message[1]))
    
    with st.chat_message("human", avatar=":material/mood:"):
        col1, col2 = st.columns([9, 1])
        with col1:
            msg_content = message[1]
            st.markdown(msg_content.replace("\n", "<br>").replace("$", "\\$").replace("#", "\\#").replace("_", "\\_"),unsafe_allow_html=True)
                
        with col2:
            if edit:
                if st.button("edit", key=f"edit_{index}"):
                    st.session_state.edit = True
            else:
                st.button("edit", key=f"dummy_{index}")
                
        if edit and st.session_state.edit:
            st.session_state.new_message = st.text_area("Please save after editing.", value=msg_content)
            if st.button("save", key=f"save_{index}", type="primary"):
                st.session_state.edit = False
                modify_message(st.session_state.chat_history, index)
                st.session_state.save = True

def render_assistant_message(message, index):
    """
    Renders the assistant's message.
    """
    col1, col2 = st.columns([9, 1])
    st.session_state.total_tokens += len(encoding.encode(message[1]))
    with col1:
        with st.chat_message("assistant", avatar=":material/psychology:"):
            st.markdown(message[1].replace("\n","  \n"), unsafe_allow_html=True)
    with col2:
        if index == len(st.session_state.chat_history) - 1:
            copy_button(st.session_state.response)

def show_chat_history(messages, edit, new_message=None):
    """
    Displays the entire chat history and renders new messages if needed.
    """
    st.session_state.total_tokens = 0
    for i, message in enumerate(messages):
        if message[0] == "human":
            render_human_message(message, i, edit)
        elif message[0] == "assistant":
            render_assistant_message(message, i)
    if new_message:
        st.session_state.total_tokens += len(encoding.encode(new_message))
        with st.chat_message("user", avatar=":material/mood:"):
            st.markdown(new_message.replace("\n", "<br>").replace("$", "\\$").replace("#", "\\#").replace("_", "\\_"),unsafe_allow_html=True)

show_chat_history(messages=st.session_state.chat_history,edit=True)

if user_input is not None:
    st.session_state.done = False
    st.session_state.edit = False   
    st.session_state.total_tokens += len(encoding.encode(user_input))
    check_token()
    with st.chat_message("human",avatar = ":material/mood:"):
        col1,  col2 = st.columns([9,  1])
        with col1:
            st.markdown(user_input.replace("\n", "<br>").replace("$", "\\$").replace("#", "\\#").replace("_", "\\_"),unsafe_allow_html=True)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", st.session_state.system_prompt),
                MessagesPlaceholder(variable_name="conversation"),
                ("human", "{input}"),
            ]
        )

    chain = (
        prompt_template
        | st.session_state.llm
    ).with_config({"run_name": "Chat","tags": ["Chat"]})

    st.session_state.response = ""
    with st.chat_message("assistant",avatar = ":material/psychology:"):
        col1,  col2 = st.columns([9,  1])
        with col1:
            message_placeholder = st.empty()
            message_placeholder.markdown("thinking...")
        with col2:
            st.session_state.stop = st.button("stop")
        with col1:
            for chunk in chain.stream({"input": user_input,"conversation": st.session_state.chat_history}):
                st.session_state.response += chunk.content
                message_placeholder.markdown(st.session_state.response.replace("\n","  \n") + "▌",unsafe_allow_html=True)
            message_placeholder.markdown(st.session_state.response.replace("\n","  \n"))
        
    st.session_state.done = True

    st.session_state.chat_history.append(("human", user_input))
    st.session_state.chat_history.append(("assistant", st.session_state.response))
    st.session_state.clear_button=True
    st.session_state.clear_button_holder = st.empty()
    st.rerun()

if st.session_state.save:
    st.session_state.done = False
    prompt = st.session_state.new_message
    show_chat_history(messages=st.session_state.chat_history,edit=False,new_message=prompt)
    check_token()
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", st.session_state.system_prompt),
            MessagesPlaceholder(variable_name="conversation"),
            ("human", "{input}"),
        ]
    )

    chain = (
        prompt_template
        | st.session_state.llm
    ).with_config({"run_name": "Chat","tags": ["Chat","reasoning"]})

    st.session_state.response = ""
    with st.chat_message("assistant",avatar = ":material/psychology:"):
        col1,  col2 = st.columns([9,  1])
        with col1:
            message_placeholder = st.empty()
            message_placeholder.markdown("thinking...")
        with col2:
            st.session_state.stop = st.button("stop")
        with col1:
            for chunk in chain.stream({"input": prompt,"conversation": st.session_state.chat_history}):
                st.session_state.response += chunk.content
                message_placeholder.markdown(st.session_state.response.replace("\n","  \n") + "▌",unsafe_allow_html=True)
        
    st.session_state.done = True
    st.session_state.chat_history.append(("human", prompt))
    st.session_state.chat_history.append(("assistant", st.session_state.response))
    st.session_state.clear_button=True
    st.session_state.clear_button_holder = st.empty()
    st.session_state.save = False
    st.rerun()

if st.session_state.clear_button:
    with st.session_state.clear_button_holder:
        button_clear_chat = st.button("clear chat history",type="primary")
        if button_clear_chat:
            clear_chat()