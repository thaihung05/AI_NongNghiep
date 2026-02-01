import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0.2,
)

prompt = PromptTemplate.from_template(
    """Bạn là trợ lý hỗ trợ tra cứu kiến thức về cây lúa dựa trên tài liệu đã cung cấp (RAG), không được bịa.
    QUY TẮC BẮT BUỘC:
    1) Chỉ sử dụng thông tin xuất hiện TRỰC TIẾP trong NGỮ CẢNH. Không suy diễn, không bổ sung kiến thức ngoài.
    2) Nếu trong NGỮ CẢNH không có thông tin TRỰC TIẾP trả lời câu hỏi,hoặc thông tin chỉ liên quan một phần,
        bạn PHẢI trả lời đúng một câu duy nhất:
        "Không tìm thấy thông tin trong tài liệu."
        KHÔNG giải thích thêm.  
    3) Nếu trả lời, hãy bám sát từng ý có trong ngữ cảnh.

    NGỮ CẢNH: {context}
    CÂU HỎI: {question}
    """
)

@st.cache_resource
def load_vectorstore():
    embedding = HuggingFaceEmbeddings(
        model_name="models/vietnamese-document-embedding",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = FAISS.load_local("vectorstore", embedding, allow_dangerous_deserialization=True)
    return db

db = load_vectorstore()
retriever = db.as_retriever(search_kwargs={"k": 5})

@st.cache_resource
def load_chain():
    memory = ConversationBufferMemory(
        k=2,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type="stuff",
        return_source_documents=True,
        rephrase_question=False,
        combine_docs_chain_kwargs={
            "prompt": prompt,
        }
    )
    return chain

chain = load_chain()

def ask(question):
    result = chain.invoke({"question": question})
    return result["answer"]

def ask_debug(question):
    result = chain.invoke({"question": question})
    answer = result["answer"]
    docs = result.get("source_documents")
    sources = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source")
        page = d.metadata.get("page_label")
        content_preview = " ".join(d.page_content.split())[:300]
        sources.append({
            "chunk_id": i,
            "source": source,
            "page": page,
            "preview": content_preview
        })

    return {
        "question": question,
        "sources": sources,
        "answer": answer
    }

if __name__=="__main__":
    print(ask_debug("Gieo sạ lúa với mật độ quá dày sẽ gây ra những vấn đề gì trong quá trình sinh trưởng và phòng trừ sâu bệnh?"))
# if __name__=="__main__":
#     print(ask("Gieo sạ lúa với mật độ quá dày sẽ gây ra những vấn đề gì trong quá trình sinh trưởng và phòng trừ sâu bệnh?"))