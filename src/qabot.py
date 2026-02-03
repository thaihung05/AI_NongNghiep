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
    model="gemini-3-flash-preview",
    google_api_key=API_KEY,
    temperature=0.2,
)

prompt = PromptTemplate.from_template(
    """Bạn là trợ lý chuyên môn nông nghiệp, có nhiệm vụ trả lời câu hỏi về kỹ thuật canh tác cây lúa dựa CHỈ trên các tài liệu được cung cấp.
    NGỮ CẢNH ĐƯỢC CUNG CẤP:
    {context}
    CÂU HỎI CỦA NGƯỜI DÙNG:
    {question}

    Lưu ý khi trả lời:
    1. Chỉ sử dụng thông tin CÓ trong ngữ cảnh trên. Tuyệt đối không tự nghĩ ra hoặc dùng kiến thức bên ngoài.
    2. Xử lý thông tin thiếu: 
       - Nếu ngữ cảnh hoàn toàn không có thông tin: Trả lời "Tài liệu không đề cập đến vấn đề này.". Và hướng dẫn tìm nội dung trên những phương thức khác.
       - Nếu ngữ cảnh chỉ trả lời được một phần câu hỏi nhưng vẫn "đúng ý của câu hỏi": Hãy trả lời phần đó và ghi chú rõ "Tài liệu chỉ cung cấp thông tin về [nội dung có], chưa có thông tin về [nội dung thiếu]."
    3. Trích dẫn (Quan trọng): Khi đưa ra số liệu (ví dụ: lượng phân bón, năng suất, mật độ sạ, ...), hãy cố gắng nhắc đến nguồn nếu có trong ngữ cảnh (ví dụ: "Theo luận án tiến sĩ..." hoặc "Theo Sổ tay hướng dẫn 1 triệu ha...").
    4. Trình bày: Sử dụng gạch đầu dòng cho các quy trình kỹ thuật hoặc danh sách để dễ đọc.
    5. Không được suy luận, tổng hợp, hoặc khái quát vượt quá thông tin xuất hiện trực tiếp trong ngữ cảnh, kể cả khi bạn "biết" kiến thức đó là đúng.
    
    CÂU TRẢ LỜI:
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
            "preview": content_preview
        })

    return {
        "question": question,
        "sources": sources,
        "answer": answer
    }

if __name__=="__main__":
    question_input = "Theo tài liệu, việc gieo sạ không đúng thời vụ có thể dẫn đến những rủi ro gì trong sản xuất lúa?"
    print(ask_debug(question_input))
# if __name__=="__main__":
#     print(ask("Gieo sạ lúa với mật độ quá dày sẽ gây ra những vấn đề gì trong quá trình sinh trưởng và phòng trừ sâu bệnh?"))