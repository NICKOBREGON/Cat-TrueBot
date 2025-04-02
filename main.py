import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import tempfile

st.set_page_config(page_title="Bot de Protocolos", layout="centered")
st.title("🧠 Bot de Protocolos Hospitalarios")
st.markdown("Sube un protocolo en PDF y haz preguntas sobre cómo actuar en cada situación.")

hf_token = st.text_input("🔑 HuggingFace Token", type="password")
if not hf_token:
    st.stop()

uploaded_file = st.file_uploader("📄 Sube tu protocolo en PDF", type="pdf")
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load_and_split()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(pages, embeddings)
    retriever = vectorstore.as_retriever()

    llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    task="text-generation",
    model_kwargs={"temperature": 0.3, "max_new_tokens": 512},
    huggingfacehub_api_token=hf_token
)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    question = st.text_input("❓ ¿Qué quieres saber?")
    if question:
        with st.spinner("Pensando..."):
            answer = qa_chain.run(question)
        st.success(answer)
