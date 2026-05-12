import streamlit as st
import torch

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==============================
# 🔧 DEVICE CHECK
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="Seha RAG System",
    layout="wide"
)

# ==============================
# 📌 SIDEBAR
# ==============================
st.sidebar.title("🛠 System Status")

if device == "cuda":
    st.sidebar.success("🚀 Running on GPU")
else:
    st.sidebar.warning("⚠️ Running on CPU")

st.sidebar.markdown("---")
st.sidebar.info("RAG-based Healthcare Assistant")

# ==============================
# 🧠 LOAD & PROCESS DATA
# ==============================
@st.cache_resource
def load_data():

    # Load PDFs
    loader = PyPDFDirectoryLoader("DATA/")
    docs = loader.load()

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )

    # Create FAISS vector database
    db = FAISS.from_documents(chunks, embeddings)

    return db

with st.spinner("📚 Loading healthcare knowledge base..."):
    db = load_data()

st.success("✅ Knowledge Base Ready")

# ==============================
# 🤖 LOAD FLAN-T5 MODEL
# ==============================
@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-base"
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base"
    ).to(device)

    return tokenizer, model

tokenizer, model = load_model()

# ==============================
# 🎯 MAIN UI
# ==============================
st.title("🩺 AI Health Assistant (RAG System)")

st.info(
    "Ask about symptoms and receive healthcare guidance "
    "based on medical documents."
)

# Example questions
st.markdown("### 💡 Example Questions")
st.write("- I have back pain")
st.write("- I have fever and cough")
st.write("- I feel anxious")
st.write("- I have chest pain")

# User input
query = st.text_input(
    "Enter your symptoms or question:"
)

# ==============================
# 🔍 QUERY PROCESSING
# ==============================
if query:

    with st.spinner("🔍 Analyzing your question..."):

        # --------------------------------
        # 1. RETRIEVE RELEVANT DOCUMENTS
        # --------------------------------
        results = db.similarity_search(query, k=3)

        # Reduce context size for better generation
        context = " ".join(
            [doc.page_content[:300] for doc in results]
        )

        # --------------------------------
        # 2. CREATE PROMPT
        # --------------------------------
        prompt = f"""
You are a healthcare AI assistant.

Use ONLY the provided context.
Do NOT invent or assume information.

Explain the answer clearly for a normal patient.

Structure the response EXACTLY as:

Possible condition:
...

Recommended actions:
- ...
- ...
- ...

When to see a doctor:
...

Context:
{context}

Question:
{query}

Answer:
"""

        # --------------------------------
        # 3. TOKENIZE INPUT
        # --------------------------------
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        ).to(device)

        # --------------------------------
        # 4. GENERATE RESPONSE
        # --------------------------------
        try:

            outputs = model.generate(
                **inputs,
                max_length=256,
                min_length=80,
                num_beams=4,
                early_stopping=True
            )

            answer = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

        except Exception as e:

            answer = f"Error generating response: {str(e)}"

    # ==============================
    # 🧠 DISPLAY AI RESPONSE
    # ==============================
    st.subheader("🧠 AI Medical Response")

    st.write(answer)

    # ==============================
    # 📄 DISPLAY SOURCES
    # ==============================
    st.subheader("📄 Retrieved References")

    for i, doc in enumerate(results):

        source = doc.metadata.get(
            "source",
            "Unknown Source"
        )

        with st.expander(
            f"Reference {i+1} - {source}"
        ):

            st.write(doc.page_content)

# ==============================
# ⚠️ DISCLAIMER
# ==============================
st.markdown("---")

st.warning(
    "⚠️ This AI system provides general healthcare guidance "
    "based on medical documents. "
    "It is NOT a replacement for professional medical advice."
)