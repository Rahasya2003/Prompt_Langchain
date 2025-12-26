from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
import streamlit as st
from dotenv import load_dotenv  

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.8
)
model = ChatHuggingFace(llm = llm)

st.header("Research Tool")

user_input = st.text_input("Enter your Prompt")

if st.button("Sumarize"):
    result = model.invoke(user_input)
    st.write(result.content)