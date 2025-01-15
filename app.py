import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from datasets import load_dataset

# --- Modell und Vektorspeicher vorbereiten ---
@st.cache_resource
def load_model():
    model_name = "EleutherAI/gpt-neo-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
    
    text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    return llm

@st.cache_resource
def build_vectorstore():
    dataset = load_dataset("json", data_files={"train": "koerber_data.jsonl"})
    documents = [Document(page_content=doc["completion"]) for doc in dataset["train"]]
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    return vectorstore

# Modell und Vektorindex laden
llm = load_model()
vectorstore = build_vectorstore()

# RAG-Chain erstellen
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# --- Streamlit UI ---
st.title("🔎 Körber AI Assistant")
st.subheader("Stelle deine Fragen zum Unternehmen Körber")

# Texteingabe
query = st.text_input("Deine Frage:", placeholder="Was macht Körber im Bereich Supply Chain?")

# Button
if st.button("Antwort generieren"):
    if query:
        with st.spinner("Antwort wird generiert..."):
            response = rag_chain.invoke({"query": query})
            st.success("Antwort:")
            st.write(response["result"])

            # Quellen anzeigen
            st.info("Genutzte Quellen:")
            for idx, doc in enumerate(response["source_documents"]):
                st.write(f"**Quelle {idx+1}:** {doc.page_content}")
    else:
        st.warning("Bitte eine Frage eingeben.")
