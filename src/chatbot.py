from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv
from langchain import hub
from pathlib import Path
import streamlit as st
import os

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = InMemoryVectorStore(embedding=embeddings)

def load_data():
    files = [
        r"./data/department_snapshot.json",
        r"./data/exec_summary_org.json",
        r"./data/hierarchy_level_snapshot.json",
        r"./data/legal_entity_snapshot.json",
        r"./data/location_snapshot.json",
        r"./data/org_engagement.json",
        r"./data/overall_module_stats.json",
    ]

    docs = []
    for file in files:
        loader = JSONLoader(
            file_path=file,
            jq_schema=".",  # Extracts the entire JSON object
            text_content=False  # Prevents expecting page_content as a string
        )
        data = loader.load()
        docs.extend(data)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    
    _ = vector_store.add_documents(documents=all_splits)    

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def chat(query, thread_id="chatbot"):
    # Specify an ID for the thread
    config = {"configurable": {"thread_id": thread_id}}
    response = graph.invoke(
        {"question": query},
        config=config
    )
    return response["answer"]