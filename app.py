import os
import json
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load API key
load_dotenv(dotenv_path=".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Page config
st.set_page_config(page_title="GenAI Chatbot", page_icon="🤖")
st.title("🤖 GenAI Chatbot")
st.caption("🚀 Built with OpenAI + Streamlit")

# Sidebar controls
with st.sidebar:
    st.title("⚙️ Settings")

    role = st.selectbox(
        "Choose Assistant Role",
        ["General Assistant", "Data Analyst", "Career Coach"]
    )

    temperature = st.slider("Creativity", 0.0, 1.0, 0.7)

    if st.button("Clear Chat"):
        st.session_state.messages = []

    if st.button("Download Chat"):
        st.download_button(
            label="Download",
            data=json.dumps(st.session_state.get("messages", []), indent=2),
            file_name="chat.json",
            mime="application/json"
        )

# Initialize chat with system prompt
if "messages" not in st.session_state:
    if role == "Data Analyst":
        system_prompt = "You are a professional data analyst who explains insights clearly."
    elif role == "Career Coach":
        system_prompt = "You help users with resumes, interviews, and career advice."
    else:
        system_prompt = "You are a helpful assistant."

    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]

# Display chat history
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Input box
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # API call with spinner + error handling
    try:
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages,
                temperature=temperature
            )

        reply = response.choices[0].message.content

        st.session_state.messages.append({"role": "assistant", "content": reply})

        with st.chat_message("assistant"):
            st.markdown(reply)

    except Exception as e:
        st.error(f"Error: {e}")