from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
print("Loaded Google API Key")
    
def load_documents(dirpath):
    try:
        print("Loading documents...")
        loader = DirectoryLoader(dirpath, glob="**/*.txt")
        docs = loader.load()
        print(f"Loaded {len(docs)} documents.")
        
        return docs
    except Exception as e:
        return f"Error: {e}"
    
def create_vectorstore():
    try:
        print("Calling load_documents...")
        docs = load_documents(r"C:\Orglens_Official\chatbot\backend\data")
        
        if isinstance(docs, str):
            return docs

        print("Creating vector store...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vector_store = Chroma(
            embedding_function=embeddings
        )
        _ = vector_store.add_documents(all_splits)
        print("Vector store created successfully.")
    
        return "Vector store created successfully."
    except Exception as e:
        return f"Error: {e}"