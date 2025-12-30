import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("Stress Buddy 💔😉")
st.caption("Represented By Abhi BABA👾")

# 🔒 Backend-controlled personality
DEFAULT_PERSONALITY = "Funny 😁"

if st.button("Create Knowledgebase"):
    create_vector_db()
    st.success("Knowledgebase created successfully!")

with st.form("qa_form"):
    question = st.text_input("Say whatever is on your mind 💭")
    submit = st.form_submit_button("Send")

if submit and question:
    chain = get_qa_chain(DEFAULT_PERSONALITY)
    response = chain.invoke(question)

    st.header("Reply")
    st.write(response)
