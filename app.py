import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datasets import load_dataset
import re
from datetime import datetime

# --- Körber-Daten aus der JSON-Datei laden ---
def load_koerber_data():
    dataset = load_dataset("json", data_files={"train": "koerber_data.jsonl"})
    documents = [{
        "content": doc["completion"],
        "url": doc["meta"]["url"],
        "timestamp": doc["meta"].get("timestamp", datetime.now().isoformat()),
        "title": doc["meta"].get("title", "Kein Titel")
    } for doc in dataset["train"]]
    return documents

# --- Vektorspeicher erstellen ---
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vectorstore = FAISS.from_texts(texts=[chunk["content"] for chunk in text_chunks], embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.warning(f"Fehler beim Erstellen des Vektorspeichers: {e}")

# --- Schlagwörter extrahieren (max. 3 anzeigen) ---
def extract_keywords(text, max_keywords=3):
    stopwords = ["der", "die", "das", "und", "in", "auf", "von", "zu", "mit", "für", "an", "bei"]
    words = re.findall(r'\b[A-ZÄÖÜ][a-zäöüß]+\b', text)
    keywords = [word for word in words if word.lower() not in stopwords]
    return list(set(keywords))[:max_keywords]  # Begrenzung auf 3 Schlagwörter

# --- Antwort generieren ---
def get_response(context, question, model):
    prompt_template = f"""
    Du bist ein hilfreicher Assistent (Experte für Logistik, Ingenieurwesen und Personalwesen). 
    Beantworte die folgende Frage basierend auf dem bereitgestellten Kontext ausführlich und professionell. 
    Zeige zuerst **maximal 3 relevante Schlagwörter** an und gib anschließend 3 konkrete und umsetzbare Beispiele, falls möglich.

    Kontext: {context}\n
    Frage: {question}\n

    Antwortstruktur:
    1. **Schlagwörter:** (Maximal 3 relevante Begriffe)
    2. **Antwort:** (Detaillierte und präzise Antwort)
    3. **Beispiele:** (3 konkrete Handlungsvorschläge, falls möglich)
    """
    try:
        response = model.generate_content(prompt_template)
        return response.text
    except Exception as e:
        st.warning(f"Fehler bei der Generierung: {e}")
        return ""

# --- Hauptprozess ---
def main():
    load_dotenv()
    st.set_page_config(page_title="Körber AI Chatbot", page_icon=":factory:")
    st.header("🔍 Stell deine Fragen")

    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

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

    # --- Vektorspeicher laden ---
    if st.session_state.vectorstore is None:
        with st.spinner("Daten werden geladen..."):
            documents = load_koerber_data()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
            text_chunks = [{"content": chunk, "url": doc["url"], "timestamp": doc["timestamp"], "title": doc["title"]}
                           for doc in documents for chunk in text_splitter.split_text(doc["content"])]
            st.session_state.vectorstore = get_vector_store(text_chunks)
            st.session_state.documents = text_chunks

    # --- Benutzerabfrage ---
    query_input = st.text_input("Frag Körber", value=st.session_state.query)

    # --- Button für direkte Anfrage ---
    if st.button("Antwort generieren") and query_input:
        st.session_state.query = query_input  # Speichere die Anfrage

    # --- Generiere Antwort, wenn eine Anfrage existiert ---
    if st.session_state.query:
        with st.spinner("Antwort wird generiert..."):
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", gener
