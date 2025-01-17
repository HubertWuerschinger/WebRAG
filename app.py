import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datasets import load_dataset
import re

# --- 1. Körber-Daten aus der JSON-Datei laden ---
def load_koerber_data():
    dataset = load_dataset("json", data_files={"train": "koerber_data.jsonl"})
    documents = [{
        "content": doc["completion"],
        "url": doc["meta"]["url"],
        "timestamp": doc["meta"].get("timestamp", ""),
        "title": doc["meta"].get("title", "Kein Titel")
    } for doc in dataset["train"]]
    return documents

# --- 2. Vektorspeicher erstellen ---
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vectorstore = FAISS.from_texts(texts=[chunk["content"] for chunk in text_chunks], embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Fehler beim Erstellen des Vektorspeichers: {e}")
        return None

# --- 3. Schlagwörter mit Gemini extrahieren ---
def extract_keywords_with_gemini(question, model):
    prompt_template = f"""
    Extrahiere die 3 wichtigsten Schlagwörter aus folgender Frage. Gib nur die Schlagwörter als Liste zurück:

    Frage: {question}
    """
    try:
        response = model.generate_content(prompt_template)
        keywords = re.findall(r'\b\w+\b', response.text)
        return keywords[:3]  # Maximal 3 Schlagwörter
    except Exception as e:
        st.error(f"Fehler bei der Schlagwort-Extraktion: {e}")
        return []

# --- 4. Dynamische Suche in den Dokumenten ---
def search_dynamic_content_with_keywords(documents, keywords, query):
    combined_search_terms = keywords + query.split()
    for doc in documents:
        content_lower = doc["content"].lower()
        if any(term.lower() in content_lower for term in combined_search_terms):
            return doc["content"]
    return "Keine passenden Informationen gefunden."

# --- 5. Antwort generieren ---
def get_response(context, question, model, documents):
    extracted_keywords = extract_keywords_with_gemini(question, model)
    fallback_result = search_dynamic_content_with_keywords(documents, extracted_keywords, question)

    if fallback_result != "Keine passenden Informationen gefunden.":
        return f"Direkt gefunden: {fallback_result}"

    prompt_template = f"""
    Du bist ein Experte für Logistik, Ingenieurwesen und Personalwesen. Beantworte die folgende Frage basierend auf dem Kontext:

    Kontext: {context}\n
    Frage: {question}\n

    Antworte strukturiert und liefere 3 praxisnahe Beispiele.
    """
    try:
        response = model.generate_content(prompt_template)
        return response.text
    except Exception as e:
        st.error(f"Fehler bei der Generierung: {e}")
        return ""

# --- 6. Hauptprozess ---
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

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "query" not in st.session_state:
        st.session_state.query = ""

    # --- Vektorspeicher und Dokumente laden ---
    if st.session_state.vectorstore is None:
        with st.spinner("Daten werden geladen..."):
            documents = load_koerber_data()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
            text_chunks = [{"content": chunk, "url": doc["url"]} for doc in documents for chunk in text_splitter.split_text(doc["content"])]
            st.session_state.vectorstore = get_vector_store(text_chunks)
            st.session_state.documents = text_chunks

    # --- Benutzerabfrage ---
    query_input = st.text_input("Frag Körber", value=st.session_state.query)

    if st.button("Antwort generieren") and query_input:
        st.session_state.query = query_input

    # --- Antwort generieren ---
    if st.session_state.query:
        with st.spinner("Antwort wird generiert..."):
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config)
            vectorstore = st.session_state.vectorstore
            relevant_content = vectorstore.similarity_search(st.session_state.query, k=5)

            context = "\n".join([getattr(doc, "page_content", getattr(doc, "content", "")) for doc in relevant_content])
            result = get_response(context, st.session_state.query, model, st.session_state.documents)

            st.success("Antwort:")
            st.write(result)

if __name__ == "__main__":
    main()
