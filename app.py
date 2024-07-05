import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag_app import App

st.set_page_config(page_title="Chat")

def display_messages():
    print("---DISPLAY MSGS---")
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    print("---PROCESS INPUT---")
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        print("user_text:", user_text)
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].invoke(user_text)
        print("agent_text:", agent_text)
        for output in agent_text:
            for key, value in output.items():
                print(f"Finished running: {key}:{value}")
        print(value["generation"])
        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((value["generation"], False))

def process_url():
    print("---PROCESS URL---")
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""
    if st.session_state["url_input"] and len(st.session_state["url_input"].strip()) > 0:
        url = st.session_state["url_input"].strip()
        print("url:", url)
        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting"):
            st.session_state["assistant"].ingest(urls=[url])

def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""
    st.session_state["url_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest(pdf_file_paths=[file_path])
        os.remove(file_path)

def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = App()

    st.header("Chat")

    st.subheader("Upload a document or enter an URL")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )
    st.text_input("Enter an URL", key="url_input", on_change=process_url)

    st.session_state["ingestion_spinner"] = st.empty()
    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

if __name__ == "__main__":
    page()