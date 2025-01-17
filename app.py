import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datasets import load_dataset
import re

# --- 1. Umgebungsvariablen sicher laden ---
def load_api_keys():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("API-Schlüssel fehlt. Bitte die .env-Datei prüfen.")
        st.stop()
    return api_key

# --- 2. JSONL-Daten laden ---
def load_koerber_data():
    dataset = load_dataset("json", data_files={"train": "koerber_data.jsonl"})
    return [{
        "content": doc["completion"],
        "url": doc["meta"].get("url", ""),
        "timestamp": doc["meta"].get("timestamp", ""),
        "title": doc["meta"].get("title", "Kein Titel")
    } for doc in dataset["train"]]

# --- 3. Vektorspeicher (FAISS) erstellen ---
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        return FAISS.from_texts(texts=[chunk["content"] for chunk in text_chunks], embedding=embeddings)
    except Exception as e:
        st.error(f"Fehler beim Erstellen des Vektorspeichers: {e}")
        return None

# --- 4. Schlagwörter mit Gemini extrahieren ---
def extract_keywords_with_llm(model, query):
    prompt = f"Extrahiere relevante Schlagwörter aus der folgenden Anfrage:\n\n{query}\n\nNur Schlagwörter ohne Erklärungen."
    try:
        response = model.generate_content(prompt)
        return re.findall(r'\b\w{3,}\b', response.text)
    except Exception as e:
        st.error(f"Fehler bei der Schlagwort-Extraktion: {e}")
        return []

# --- 5. JSONL-Dokumente nach Schlagwörtern durchsuchen ---
def search_documents(documents, keywords):
    for doc in documents:
        if any(keyword.lower() in doc["content"].lower() for keyword in keywords):
            return f"{doc['content']}\n\n🔗 [Quelle]({doc['url']})"
    return "Keine passenden Informationen gefunden."

# --- 6. Fallback: Antwort mit Vektorspeicher (RAG) generieren ---
def generate_response(context, question, model):
    prompt = f"""
    Beantworte die folgende Frage basierend auf diesem Kontext strukturiert und mit 3 Beispielen:
    
    Kontext: {context}
    
    Frage: {question}
    """
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        st.error(f"Fehler bei der Generierung: {e}")
        return ""

# --- 7. Hauptprozess der App ---
def main():
    api_key = load_api_keys()
    genai.configure(api_key=api_key)

    st.set_page_config(page_title="Körber AI Chatbot", page_icon=":factory:")
    st.header("🔍 Stell deine Fragen")

    generation_config = {
        "temperature": 0.2,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 8000,
    }

    # --- Session State initialisieren ---
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "query" not in st.session_state:
        st.session_state.query = ""

    # --- JSONL-Daten laden und Vektorspeicher erstellen ---
    if st.session_state.vectorstore is None:
        with st.spinner("Daten werden geladen..."):
            documents = load_koerber_data()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
            text_chunks = [{"content": chunk, "url": doc["url"]} for doc in documents for chunk in text_splitter.split_text(doc["content"])]
            st.session_state.vectorstore = get_vector_store(text_chunks)
            st.session_state.documents = text_chunks

    # --- 1. Benutzereingabe ---
    query_input = st.text_input("Frag Körber", value=st.session_state.query)

    if st.button("Antwort generieren") and query_input:
        st.session_state.query = query_input

    # --- 2. Schlagwörter mit Gemini extrahieren und 3. JSONL durchsuchen ---
    if st.session_state.query:
        with st.spinner("Antwort wird generiert..."):
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config)

            # Schlagwörter aus der Benutzereingabe extrahieren
            keywords = extract_keywords_with_llm(model, st.session_state.query)

            # JSONL-Daten nach den Schlagwörtern durchsuchen
            result = search_documents(st.session_state.documents, keywords)

            # Fallback auf Vektorspeicher (RAG), falls nichts gefunden wurde
            if result == "Keine passenden Informationen gefunden.":
                relevant_content = st.session_state.vectorstore.similarity_search(st.session_state.query, k=5)
                context = "\n".join([getattr(doc, "page_content", getattr(doc, "content", "")) for doc in relevant_content])
                result = generate_response(context, st.session_state.query, model)

            # --- 4. Antwort ausgeben ---
            st.success("Antwort:")
            st.write(result)

if __name__ == "__main__":
    main()
