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
        """Bạn là chuyên gia nông nghiệp thân thiện, giải thích đầy đủ thông tin và dễ hiểu bằng tiếng Việt.
            Trả lời dựa trên NGỮ CẢNH dưới đây. Nếu không có thông tin thì nói rõ là không thấy trong tài liệu, đừng đoán.
            Ngữ cảnh: {context}
            Câu hỏi: {question}
            Trả lời (đầy đủ, dễ hiểu, có thể kèm khuyến nghị thực hành nếu cần):
        """
)

embedding = HuggingFaceEmbeddings(
    model_name="models/vietnamese-document-embedding",
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
)

db = FAISS.load_local("vectorstore", embedding, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={"k": 4})
chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt | llm | StrOutputParser()
)

def ask(question: str):
    return chain.invoke(question)
