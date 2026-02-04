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
    """Bạn là trợ lý chuyên môn nông nghiệp. Hãy trả lời câu hỏi về kỹ thuật canh tác lúa chỉ dựa trên ngữ cảnh được cung cấp.

        NGỮ CẢNH ĐƯỢC CUNG CẤP: {context}
        CÂU HỎI CỦA NGƯỜI DÙNG: {question}

        Quy tắc trả lời:
        - Chỉ dùng thông tin trong ngữ cảnh. Không suy đoán, không dùng kiến thức ngoài.
        - Nếu ngữ cảnh không có thông tin: trả lời đúng câu "Tài liệu không đề cập đến vấn đề này.".
        - Nếu ngữ cảnh chỉ trả lời được một phần: trả lời phần có trong tài liệu và ghi rõ
        "Tài liệu chỉ cung cấp thông tin về [nội dung có], chưa có thông tin về [nội dung thiếu]."
        - Trình bày rõ ràng; dùng gạch đầu dòng cho quy trình hoặc danh sách.
        - Giọng điệu thân thiện, gần gũi như kỹ sư nông nghiệp, nhưng không bịa.
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
        text = " ".join(d.page_content.split())
        content_preview = text[:300]
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
    question_input = "Bệnh đạo ôn cổ bông là gì?"
    print(ask_debug(question_input))
    # print(ask(question_input))
