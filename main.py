import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
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

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_documents(pages, embeddings)
    retriever = vectorstore.as_retriever()

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.3, "max_length": 512},
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
