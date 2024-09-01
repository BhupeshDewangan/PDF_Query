import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3


# from phi.assistant import Assistant
# from phi.llm.openai import OpenAIChat

from pydub import AudioSegment
from gtts import gTTS

from app import get_conversational_chain, get_vector_store
from app import get_pdf_text, get_text_chunks, text_to_speech

os.environ['GOOGLE_API_KEY'] = "AIzaSyA33GBXRny_UjzJEd_sNz9VlCIiEdX5SZ8"

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

if 'messages' not in st.session_state:
    st.session_state.messages = []

for messages in st.session_state.messages:
    with st.chat_message(messages['role']):
        st.markdown(messages['content'])

def input_user(prompt_input):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(prompt_input)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": prompt_input}
        ,return_only_outputs=True)

    text_to_speech(response['output_text'])
    return response['output_text']



# st.title("PDF QUERY")

with st.sidebar:
        st.title("PDF QUERY 📃")
        st.divider()

        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


prompt_input = st.chat_input("what's up ?")

if prompt_input:
    with st.chat_message('user'):
        st.markdown(prompt_input)
    
    st.session_state.messages.append({'role': 'user', 'content' : prompt_input})

    with st.chat_message('assistant'):
        message_placeholder = st.empty()

        response = input_user(prompt_input)
        # response = st.write_stream(input_user(prompt_input))

        message_placeholder.markdown(response)
    
    st.session_state.messages.append({'role': 'assistant', 'content' : response})

    st.audio("response.mp3", format="audio/mpeg")

    # with st.spinner("Going"):
    #     st.session_state.messages.append({'role': 'assistant', 'content' : response})

    #     st.audio("response.mp3", format="audio/mpeg")

