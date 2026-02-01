from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

file_path = "data"
vector_db_path = "vectorstore"

def creat_db_from_files():
    loader = DirectoryLoader(file_path, glob="*.pdf", loader_cls= PyPDFLoader)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    
    embedding_models = HuggingFaceEmbeddings(
        model_name="models/vietnamese-document-embedding",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    db = FAISS.from_documents(chunks, embedding_models)
    db.save_local(vector_db_path)
    return db

creat_db_from_files()