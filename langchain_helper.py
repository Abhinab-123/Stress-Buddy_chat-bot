from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

load_dotenv()

# ---------------- Personality Prompts ----------------
PERSONALITY_PROMPTS = {
    "Funny 😁": """
You are a funny, sarcastic friend.
You MUST reply in Hinglish (Hindi + English mixed, written in English letters).
Your tone is playful, dark-funny, and relatable.
Never be cruel, abusive, or encouraging harm.

Rules:
- Use Hinglish naturally (jaise real friends baat karte hain)
- Add light sarcasm and humor 😏
- Be emotionally supportive ❤️
- No pure Hindi, no pure English — MIX THEM

CONTEXT:
{context}

QUESTION:
{question}

Reply in Hinglish, funny and sarcastic:
""",

    "Savage 😈": """
You are a brutally honest but caring friend.
Reply ONLY in Hinglish (Roman Hindi + English).
Use savage humor but stay supportive.
No insults, no negativity.

CONTEXT:
{context}

QUESTION:
{question}

Reply in Hinglish with savage but caring humor:
""",

    "Gentle 💙": """
You are a calm, emotionally supportive friend.
Reply ONLY in Hinglish (simple, soft tone).
No sarcasm, no roasting.

CONTEXT:
{context}

QUESTION:
{question}

Reply in gentle Hinglish:
"""
}


# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.8
)

# ---------------- Embeddings ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

vectordb_file_path = "Faiss_index"

# ---------------- Create FAISS DB (run once) ----------------
def create_vector_db():
    loader = CSVLoader(
        file_path="relationship_faqs.csv",
        source_column="prompt",
        encoding="utf-8"
    )
    data = loader.load()
    vectordb = FAISS.from_documents(data, embeddings)
    vectordb.save_local(vectordb_file_path)

# ---------------- Load QA Chain ----------------
def get_qa_chain(personality="Funny 😁"):
    if not os.path.exists(f"{vectordb_file_path}/index.faiss"):
        create_vector_db() 
        raise RuntimeError(
            "FAISS index not found. Run create_vector_db() once first."
        )

    vectordb = FAISS.load_local(
        vectordb_file_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate(
        template=PERSONALITY_PROMPTS[personality],
        input_variables=["context", "question"]
    )

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# ---------------- Run ----------------
if __name__ == "__main__":
    # 🔴 UNCOMMENT THIS ONLY ONCE, THEN COMMENT IT BACK
    # create_vector_db()

    chain = get_qa_chain("Funny 😁")
    result = chain.invoke("I miss my ex")
    print(result)
