import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datasets import load_dataset
import re

"""
Lädt den API-Schlüssel aus der .env-Datei, um den Zugriff auf die Gemini API zu ermöglichen.
Beendet die Anwendung, wenn der API-Schlüssel fehlt.
"""
def load_api_keys():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("API-Schlüssel fehlt. Bitte die .env-Datei prüfen.")
        st.stop()
    return api_key

"""
Lädt die Daten aus der JSONL-Datei und bereitet sie für die Suche vor.
"""
def load_koerber_data():
    dataset = load_dataset("json", data_files={"train": "koerber_data.jsonl"})
    return [{
        "content": doc["completion"],
        "url": doc["meta"].get("url", ""),
        "timestamp": doc["meta"].get("timestamp", ""),
        "title": doc["meta"].get("title", "Kein Titel")
    } for doc in dataset["train"]]

"""
Erstellt den Vektorspeicher (FAISS) aus den geladenen Textdaten.
"""
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        return FAISS.from_texts(texts=[chunk["content"] for chunk in text_chunks], embedding=embeddings)
    except Exception as e:
        st.error(f"Fehler beim Erstellen des Vektorspeichers: {e}")
        return None

"""
Extrahiert relevante Schlagwörter aus dem Benutzerprompt mithilfe von Gemini.
"""
def extract_keywords_with_llm(model, query):
    prompt = f"Extrahiere relevante Schlagwörter aus der folgenden Anfrage:\n\n{query}\n\nNur Schlagwörter ohne Erklärungen."
    try:
        response = model.generate_content(prompt)
        return re.findall(r'\b\w{3,}\b', response.text)
    except Exception as e:
        st.error(f"Fehler bei der Schlagwort-Extraktion: {e}")
        return []

"""
Durchsucht den Vektorspeicher basierend auf den extrahierten Schlagwörtern und der ursprünglichen Anfrage.
"""
def search_vectorstore(vectorstore, keywords, query, k=5):
    combined_query = " ".join(keywords + [query])
    relevant_content = vectorstore.similarity_search(combined_query, k=k)
    return "\n".join([getattr(doc, "page_content", getattr(doc, "content", "")) for doc in relevant_content])

"""
Erzeugt eine strukturierte Antwort basierend auf dem gefundenen Kontext.
"""
def generate_response(context, question, model):
    prompt = f"Beantworte folgende Frage basierend auf diesem Kontext strukturiert mit Beispielen:\n\nKontext: {context}\nFrage: {question}"
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        st.error(f"Fehler bei der Generierung: {e}")
        return ""

"""
Hauptfunktion zur Steuerung des Chatbots.
1. API-Schlüssel laden
2. Vektorspeicher initialisieren
3. Benutzeranfrage verarbeiten
4. Schlagwörter extrahieren und Suche ausführen
5. Antwort generieren und anzeigen
"""
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

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "query" not in st.session_state:
        st.session_state.query = ""

    if st.session_state.vectorstore is None:
        with st.spinner("Daten werden geladen..."):
            documents = load_koerber_data()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
            text_chunks = [{"content": chunk, "url": doc["url"]} for doc in documents for chunk in text_splitter.split_text(doc["content"])]
            st.session_state.vectorstore = get_vector_store(text_chunks)
            st.session_state.documents = text_chunks

    query_input = st.text_input("Frag Körber", value=st.session_state.query)

    if st.button("Antwort generieren") and query_input:
        st.session_state.query = query_input

    if st.session_state.query:
        with st.spinner("Antwort wird generiert..."):
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config)
            keywords = extract_keywords_with_llm(model, st.session_state.query)
            context = search_vectorstore(st.session_state.vectorstore, keywords, st.session_state.query)
            result = generate_response(context, st.session_state.query, model)
            st.success("Antwort:")
            st.write(result)

if __name__ == "__main__":
    main()
