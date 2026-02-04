import streamlit as st
from src.qabot import ask

st.set_page_config(
    page_title="Rice AI Assistant",
    page_icon="🌾",
    layout="centered"
)

def ui_css():
    bg = "radial-gradient(1000px 800px at 50% 50%, #A5D6A7 0%, #E8F5E9 50%, #C8E6C9 100%)"
    st.markdown(f"""
        <style>
            html, body, [data-testid="stAppViewContainer"] {{
                font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
                background: {bg};
                color: #1B1B1B;
            }}
        </style>
    """, unsafe_allow_html=True)
ui_css()

st.markdown(
            "<h1 style='text-align: center;'>AI tư vấn trồng lúa</h1>",unsafe_allow_html=True
        )
st.markdown(
            "<h5 style='text-align: center;'>Chào bạn! Tui sẽ giải đáp những thắc mắc về nông nghiệp cho bạn!</h5>"
            ,unsafe_allow_html=True
        )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("Nhập câu hỏi...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Đang trả lời..."):
            answer = ask(user_input)
            st.markdown(answer)
            
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })