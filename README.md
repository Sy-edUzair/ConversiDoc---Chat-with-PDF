Here's a clear and beginner-friendly **`README.md`** for your **ConversiDoc** project:

---

# 📄 ConversiDoc - Chat with Your PDFs using OpenAI and LangChain

ConversiDoc is a simple Streamlit application that allows you to upload PDF documents and interact with their contents using natural language. Powered by OpenAI's GPT-4o-mini model and LangChain, it extracts, embeds, and retrieves information from your documents to answer your questions.

---

## 🚀 Features

* Upload one or multiple PDF files
* Ask questions based on the content of those files
* Keeps track of previous conversations (chat memory)
* Retrieves relevant source documents from vector store
* Lightweight and easy to run locally

---

## 🧰 Tech Stack

* **Python**
* **Streamlit** – For building the web UI
* **LangChain** – For building the conversational AI pipeline
* **OpenAI GPT-4o-mini** – For answering your questions
* **Chroma** – Vector store to persist PDF embeddings
* **PyPDF2** – For PDF text extraction
* **Dotenv** – For managing API keys securely

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ConversiDoc.git
cd ConversiDoc
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your OpenAI API Key

Create a `.env` file in the root directory with the following content:

```
OPENAI_API_KEY=your_openai_api_key_here
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📁 How It Works

1. **Upload PDFs**: Use the sidebar to upload one or multiple PDFs.
2. **Extract Text**: Text is extracted from the PDFs using PyPDF2.
3. **Chunk Text**: Text is split into manageable chunks using LangChain's text splitter.
4. **Embed & Store**: Chunks are converted into vectors and stored in a persistent Chroma database.
5. **Ask Questions**: Ask questions based on uploaded content. GPT-4o-mini will respond using memory and document context.

---

## 🧠 Memory System

This app uses **ConversationBufferMemory**, which retains the full chat history in memory. This allows the chatbot to give context-aware responses during your session.

---

## 💡 Example Questions

After uploading a PDF on YOLO (You Only Look Once – object detection), try asking:

* "What is YOLO?"
* "Who developed YOLO?"
* "What makes YOLO different from R-CNN?"
* "Explain YOLO’s architecture."
* "Summarize the key contributions of the paper."

---

## 🛠 To-Do / Improvements

* [ ] Add support for TokenBufferMemory
* [ ] UI improvements (Dark Mode, chat bubbles)
* [ ] Save chat history between sessions
* [ ] Multi-user support

---

## 📄 License

This project is open-source and free to use under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

* [LangChain](https://github.com/langchain-ai/langchain)
* [Streamlit](https://streamlit.io)
* [OpenAI](https://openai.com/)
* [ChromaDB](https://www.trychroma.com/)

---

Let me know if you'd like me to generate the `requirements.txt` as well.
