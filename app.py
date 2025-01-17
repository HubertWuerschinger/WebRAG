import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datasets import load_dataset
import re
from collections import Counter

# --- Körber-Daten aus der JSON-Datei laden ---
def load_koerber_data():
    dataset = load_dataset("json", data_files={"train": "koerber_data.jsonl"})
    documents = [{
        "content": doc["completion"],
        "url": doc["meta"]["url"],
        "timestamp": doc["meta"].get("timestamp", ""),
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

# --- Erweiterte Schlagwort-Extraktion ---
def extract_keywords(text, max_keywords=3):
    # Relevante Berufs- und Fachbegriffe
    domain_terms = [
        "Ingenieur", "Maschinenbau", "Automatisierung", "Logistik", "Softwareentwicklung",
        "IT", "Projektmanagement", "Mechatronik", "Produktion", "Innovation"
    ]

    # Erweiterte Stopwords-Liste
    stopwords = [
        "der", "die", "das", "und", "in", "auf", "von", "zu", "mit", "für", "an", "bei",
        "dies", "da", "stellenangebote", "ist", "ein", "eine", "im", "sowie", "mehr", "bitte", "job", "jobs"
    ]

    # Nur Wörter mit mind. 3 Buchstaben, keine Stopwords
    words = re.findall(r'\b[A-ZÄÖÜa-zäöüß]{3,}\b', text)
    words_filtered = [word for word in words if word.lower() not in stopwords]

    # Fachbegriffe bevorzugen
    keywords = [word for word in words_filtered if word in domain_terms]

    # Wenn keine Fachbegriffe gefunden, dann häufigste Wörter nutzen
    if not keywords:
        keywords_counter = Counter(words_filtered)
        keywords = [word for word, _ in keywords_counter.most_common(max_keywords)]

    return keywords[:max_keywords]

# --- Antwort generieren ---
def get_response(context, question, model):
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
            text_chunks = [{"content": chunk, "url": doc["url"]} for doc in documents for chunk in text_splitter.split_text(doc["content"])]
            st.session_state.vectorstore = get_vector_store(text_chunks)
            st.session_state.documents = text_chunks

    # --- Benutzerabfrage ---
    query_input = st.text_input("Frag Körber", value=st.session_state.query)

    # --- Button für direkte Anfrage ---
    if st.button("Antwort generieren") and query_input:
        st.session_state.query = query_input

    # --- Generiere Antwort, wenn eine Anfrage existiert ---
    if st.session_state.query:
        with st.spinner("Antwort wird generiert..."):
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config)
            vectorstore = st.session_state.vectorstore
            relevant_content = vectorstore.similarity_search(st.session_state.query, k=5)

            context = "\n".join([doc.page_content for doc in relevant_content])
            result = get_response(context, st.session_state.query, model)

            # --- Optimierte Schlagwörter extrahieren ---
            keywords = extract_keywords(result, max_keywords=3)

            st.success("Antwort:")
            st.write(result)

            # --- Schlagwörter anzeigen ---
            st.markdown("### 📌 Relevante Themen")
            st.write(f"**Schlagwörter:** {', '.join(keywords)}")

            # --- Buttons für weitere Infos ---
            for i, keyword in enumerate(keywords):
                if st.button(f"Mehr zu: {keyword}", key=f"more_info_{i}"):
                    st.session_state.query = f"Gib mir mehr dazu zu: {keyword}"

if __name__ == "__main__":
    main()
