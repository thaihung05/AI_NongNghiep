import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

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
    2) Nếu NGỮ CẢNH không đủ để trả lời câu hỏi, hãy trả lời đúng một câu:
       "Không tìm thấy thông tin trong tài liệu."
       Sau đó gợi ý người dùng cung cấp thêm tài liệu liên quan.
    3) Nếu trả lời, hãy bám sát từng ý có trong ngữ cảnh.

    NGỮ CẢNH: {context}
    CÂU HỎI: {question}
    CÁCH TRẢ LỜI:
    - Trả lời ngắn gọn, rõ ràng, có thể gạch đầu dòng, thân thiện với người nông dân.
    """
)

embedding = HuggingFaceEmbeddings(
    model_name="models/vietnamese-document-embedding",
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
)

db = FAISS.load_local("vectorstore", embedding, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={"k": 5})
chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt | llm | StrOutputParser()
)

def ask(question: str):
    return chain.invoke(question)

def ask_debug(question: str, preview_chars: int = 300):
    docs = retriever.invoke(question)
    answer = chain.invoke(question)

    sources = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page_label", d.metadata.get("page", "N/A"))
        content_preview = d.page_content.replace("\n", " ").strip()[:preview_chars]
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
    print(ask_debug("Theo tài liệu, sản xuất lúa ở Đồng Bằng Sông Cửu Long đã thay đổi như thế nào trong các giai đoạn gần đây?"))