import os
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda


load_dotenv()

def build_vectorstore(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    faiss_index = FAISS.from_documents(docs, embeddings)
    
    if os.path.exists("faiss_nutrition"):
        faiss_index.load_local("faiss_nutrition", embeddings=embeddings, allow_dangerous_deserialization=True)
    else:
        faiss_index.save_local("faiss_nutrition")

    return faiss_index


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def extract_last_human_message(chat_history: List[Tuple[str, str]]):
    for role, message in reversed(chat_history):
        if role == "human":
            return message
    return ""


def setup_chain(vectorstore):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert nutritionist. Use the following context to answer questions:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
    )

    retriever = vectorstore.as_retriever()

    rag_chain = (
        {
            "context": RunnableLambda(lambda x: extract_last_human_message(x["chat_history"]))
            | retriever
            | format_docs,
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt_template
        | llm
    )

    return rag_chain