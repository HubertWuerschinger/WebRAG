import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datasets import load_dataset
import re
import json
import datetime

# 🔑 API-Schlüssel laden
def load_api_keys():
    """
    Lädt den Google API-Schlüssel aus der .env-Datei.

    Returns:
        str: Der geladene API-Schlüssel.

    Raises:
        Streamlit Error: Wenn der API-Schlüssel nicht gefunden wird.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("API-Schlüssel fehlt. Bitte die .env-Datei prüfen.")
        st.stop()
    return api_key

# 📂 Körber-Daten laden
def load_koerber_data():
    """
    Lädt die Körber-Daten aus einer JSONL-Datei und gibt eine Liste der Inhalte zurück.

    Returns:
        list: Eine Liste von Dictionaries mit 'content', 'url', 'timestamp' und 'title'.
    """
    dataset = load_dataset("json", data_files={"train": "koerber_data.jsonl"})
    return [{"content": doc["completion"], "url": doc["meta"].get("url", ""), 
             "timestamp": doc["meta"].get("timestamp", ""), "title": doc["meta"].get("title", "Kein Titel")} 
            for doc in dataset["train"]]

# 📦 Vektorspeicher erstellen
def get_vector_store(text_chunks):
    """
    Erstellt einen FAISS-Vektorspeicher aus Text-Chunks.

    Args:
        text_chunks (list): Liste von Textabschnitten.

    Returns:
        FAISS: Der erstellte Vektorspeicher.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        return FAISS.from_texts(texts=[chunk["content"] for chunk in text_chunks], embedding=embeddings)
    except Exception as e:
        st.error(f"Fehler beim Erstellen des Vektorspeichers: {e}")
        return None

# 🔍 Schlagwörter extrahieren
def extract_keywords_with_llm(model, query):
    """
    Extrahiert relevante Schlagwörter aus einer Anfrage mit einem LLM.

    Args:
        model (GenerativeModel): Das generative Modell von Google.
        query (str): Die Benutzeranfrage.

    Returns:
        list: Eine Liste von extrahierten Schlagwörtern.
    """
    prompt = f"Extrahiere relevante Schlagwörter aus dieser Anfrage:\n\n{query}\n\nNur Schlagwörter ohne Erklärungen."
    try:
        response = model.generate_content(prompt)
        return re.findall(r'\b\w{3,}\b', response.text)
    except Exception as e:
        st.error(f"Fehler bei der Schlagwort-Extraktion: {e}")
        return []

# 💬 Feedback speichern in JSONL
def save_feedback_jsonl(query, response, feedback_type, comment):
    """
    Speichert das Nutzerfeedback in einer JSONL-Datei und im Session State.

    Args:
        query (str): Die gestellte Frage.
        response (str): Die generierte Antwort.
        feedback_type (str): Feedback-Typ ("👍" oder "👎").
        comment (str): Optionaler Kommentar des Nutzers.
    """
    feedback_entry = {
        "query": query,
        "response": response,
        "feedback": feedback_type,
        "comment": comment,
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open("user_feedback.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(feedback_entry) + "\n")

    # Feedback im Session State speichern
    if "recent_feedback" not in st.session_state:
        st.session_state["recent_feedback"] = []
    st.session_state["recent_feedback"].append(feedback_entry)

# 📊 Letzte Feedback-Einträge anzeigen
def show_last_feedback_entries():
    """
    Zeigt die letzten 3 Feedback-Einträge direkt in der App an.
    """
    st.markdown("### 📄 **Letzte Feedback-Einträge:**")
    if "recent_feedback" in st.session_state and st.session_state["recent_feedback"]:
        for entry in st.session_state["recent_feedback"][-3:]:
            st.json(entry)
    else:
        st.info("❗️ Noch kein Feedback vorhanden.")

# 📝 Antwort generieren
def generate_response_with_feedback(vectorstore, query, model, k=5):
    """
    Generiert eine Antwort unter Berücksichtigung von Feedback und dem Vektorspeicher.

    Args:
        vectorstore (FAISS): Der Vektorspeicher.
        query (str): Die Benutzeranfrage.
        model (GenerativeModel): Das generative Modell.
        k (int): Anzahl der Top-Ergebnisse.

    Returns:
        str: Die generierte Antwort.
    """
    keywords = extract_keywords_with_llm(model, query)
    relevant_content = vectorstore.similarity_search(query, k=k)
    context = "\n".join([doc.page_content if hasattr(doc, "page_content") else doc.content for doc in relevant_content])

    prompt_template = f"""
    Kontext: {context}
    Frage: {query}

    Antworte strukturiert und präzise.
    """

    try:
        response = model.generate_content(prompt_template)
        return response.text
    except Exception as e:
        st.error(f"Fehler bei der Antwortgenerierung: {e}")
        return "Fehler bei der Antwortgenerierung."

# 🚀 Hauptprozess
def main():
    """
    Hauptfunktion zur Initialisierung der Streamlit-App.
    """
    api_key = load_api_keys()
    genai.configure(api_key=api_key)
    st.set_page_config(page_title="Körber AI Chatbot", page_icon=":factory:")
    st.header("🔍 Wie können wir dir weiterhelfen?")

    if "vectorstore" not in st.session_state:
        with st.spinner("Daten werden geladen..."):
            documents = load_koerber_data()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=600)
            text_chunks = [{"content": chunk, "url": doc["url"]} for doc in documents for chunk in text_splitter.split_text(doc["content"])]
            st.session_state.vectorstore = get_vector_store(text_chunks)

    if "recent_feedback" not in st.session_state:
        st.session_state["recent_feedback"] = []

    query_input = st.text_input("Stellen Sie hier Ihre Frage:", value="")
    generate_button = st.button("Antwort generieren")

    if generate_button and query_input:
        with st.spinner("Antwort wird generiert..."):
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
            result = generate_response_with_feedback(st.session_state.vectorstore, query_input, model)

            st.success("📝 Antwort:")
            st.write(result)

            feedback_comment = st.text_input("Kommentar zum Feedback (optional):")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Antwort war hilfreich"):
                    save_feedback_jsonl(query_input, result, "👍", feedback_comment)
                    st.success("✅ Feedback gespeichert!")
            with col2:
                if st.button("👎 Antwort verbessern"):
                    if feedback_comment.strip():
                        save_feedback_jsonl(query_input, feedback_comment, "👎", feedback_comment)
                        st.success("✅ Verbesserte Antwort gespeichert!")
                    else:
                        st.warning("⚠️ Bitte eine korrekte Antwort eingeben.")

            show_last_feedback_entries()

if __name__ == "__main__":
    main()
