import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()


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

def get_conversational_chain_from_chunks(chunks):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    if not openai_api_key:
        raise EnvironmentError("OpenAI API key not found. Please check your .env file.")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(chunks, embeddings)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

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

    query = st.text_input("Ask a question based on the uploaded PDFs:")

    if query and "qa_chain" in st.session_state:
        result = st.session_state.qa_chain({"question": query})
        st.write("Reply:", result["answer"])

    with st.sidebar:
        st.title("📁 Menu")
        pdf_docs = st.file_uploader("Upload your PDF files and click 'Submit & Process'", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = load_pdf(pdf_docs)
                    text_chunks = get_chunks(raw_text)
                    st.session_state.qa_chain = get_conversational_chain_from_chunks(text_chunks)
                    st.success("Processing complete! You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF.")

if __name__ == "__main__":
    main()
