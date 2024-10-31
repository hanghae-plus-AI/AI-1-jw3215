import bs4
import streamlit as st
import os
from dotenv import load_dotenv
from data_loader import DataLoader 
from vectorstore import VectorStoreManager
from chatbot import ChatBot
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("무엇을 도와드릴까요?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 벡터스토어 생성
    vector_manager = VectorStoreManager()
    retriever = vector_manager.get_retriever()

    # 챗봇 생성 및 응답
    chatbot = ChatBot(retriever=retriever, llm=llm)
    response = chatbot.retrieve_answer(prompt)
    
    vector_manager.clear_cache()

    with st.chat_message("assistant"):
        st.markdown(response)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })