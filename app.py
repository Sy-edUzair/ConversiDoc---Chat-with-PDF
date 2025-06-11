import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def load_pdf(docs):
    text = ""
    for pdf in docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

def create_db(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = Chroma.from_texts(chunks, embedding=embeddings, persist_directory="chroma_db")
    vector_store.persist()

def get_conversational_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,output_key="answer")

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    return chain

def main():
    st.set_page_config("ConversiDoc")
    st.header("ConversiDoc💬📄")

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = get_conversational_chain()

    query = st.text_input("Ask a question based on the uploaded PDFs:")

    if query:
        result = st.session_state.qa_chain({"question": query})
        st.write("Reply:", result["answer"])

    with st.sidebar:
        st.title("📁 Menu")
        pdf_docs = st.file_uploader("Upload your PDF files and click 'Submit & Process'", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = load_pdf(pdf_docs)
                text_chunks = get_chunks(raw_text)
                create_db(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
