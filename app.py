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

from pydub import AudioSegment
from gtts import gTTS


# os.getenv("GOOGLE_API_KEY")

os.environ['GOOGLE_API_KEY'] = "AIzaSyA33GBXRny_UjzJEd_sNz9VlCIiEdX5SZ8"

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def text_to_speech(response_text):
   print(response_text)
   tts = gTTS(text=response_text, lang="en")

   # Saves the spoken text to an mp3 file
   tts.save("response.mp3")

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    # prompt_template = """
    # summerize the question in 50 words from the provided context, make sure to provide all the details, if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer \n\n
    # Context:\n {context}?\n
    # Question: \n{question}\n

    # Answer:
    # """


    # prompt_template = """
    # Answer the question as detailed as possible from the provided context, summerize it and write it every point in new line, make sure to provide all the details, if the answer is not in  provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    # Context:\n {context}?\n
    # Question: \n{question}\n

    # Answer:
    # """


    prompt_template = """
    Please provide a detailed response based on the provided context. Summarize the information and present each point on a new line. If the answer is not available in the context, state 'answer is not available in the context.' Avoid providing incorrect information\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                            temperature=0.7)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents":docs, "question": user_question}
        ,return_only_outputs=True)

    st.write("Response: ") 
    st.write(response["output_text"])

    text_to_speech(response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if st.button('Submit'):
        user_input(user_question)
        st.audio("response.mp3", format="audio/mpeg")


    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()