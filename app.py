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
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("API-Schlüssel fehlt. Bitte die .env-Datei prüfen.")
        st.stop()
    return api_key

# 📂 Körber-Daten laden
def load_koerber_data():
    dataset = load_dataset("json", data_files={"train": "koerber_data.jsonl"})
    return [{"content": doc["completion"], "url": doc["meta"].get("url", ""), "timestamp": doc["meta"].get("timestamp", ""), "title": doc["meta"].get("title", "Kein Titel")} for doc in dataset["train"]]

# 📦 Vektorspeicher erstellen
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        return FAISS.from_texts(texts=[chunk["content"] for chunk in text_chunks], embedding=embeddings)
    except Exception as e:
        st.error(f"Fehler beim Erstellen des Vektorspeichers: {e}")
        return None

# 🔍 Schlagwörter extrahieren
def extract_keywords_with_llm(model, query):
    prompt = f"Extrahiere relevante Schlagwörter aus dieser Anfrage:\n\n{query}\n\nNur Schlagwörter ohne Erklärungen."
    try:
        response = model.generate_content(prompt)
        return re.findall(r'\b\w{3,}\b', response.text)
    except Exception as e:
        st.error(f"Fehler bei der Schlagwort-Extraktion: {e}")
        return []

# 💬 Feedback speichern in JSONL
def save_feedback_jsonl(query, response, feedback_type, comment):
    feedback_entry = {
        "query": query,
        "response": response,
        "feedback": feedback_type,
        "comment": comment,
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open("user_feedback.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(feedback_entry) + "\n")

    if check_feedback_saved(query, response):
        st.success("✅ Feedback wurde erfolgreich gespeichert!")
    else:
        st.error("❌ Fehler beim Speichern des Feedbacks!")

# ✅ Überprüfen, ob Feedback gespeichert wurde
def check_feedback_saved(query, response):
    if not os.path.exists("user_feedback.jsonl"):
        return False
    with open("user_feedback.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            entry = json.loads(line)
            if entry["query"] == query and entry["response"] == response:
                return True
    return False

# 📥 Feedback-Kommentare laden
def load_feedback_comments(query):
    comments = []
    if os.path.exists("user_feedback.jsonl"):
        with open("user_feedback.jsonl", "r", encoding="utf-8") as file:
            for line in file:
                entry = json.loads(line)
                if entry["query"].lower() == query.lower() and entry["feedback"] == "👎":
                    comments.append(entry["comment"])
    return comments

# 📝 Antwort generieren und Feedback berücksichtigen
def generate_response_with_feedback(vectorstore, query, model, k=5):
    keywords = extract_keywords_with_llm(model, query)
    relevant_content = vectorstore.similarity_search(query, k=k)
    context = "\n".join([doc.page_content if hasattr(doc, "page_content") else doc.content for doc in relevant_content])

    feedback_comments = load_feedback_comments(query)
    feedback_context = "\n".join([f"- {comment}" for comment in feedback_comments])

    prompt_template = f"""
    Kontext: {context}
    Vorheriges Feedback: {feedback_context}
    Frage: {query}

    Antworte strukturiert und präzise.
    """

    try:
        response = model.generate_content(prompt_template)
        if feedback_comments:
            st.info("🔍 Vorheriges Feedback wurde bei der Antwort berücksichtigt.")
        else:
            st.warning("⚠️ Kein vorheriges Feedback gefunden.")
        return response.text
    except Exception as e:
        st.error(f"Fehler bei der Antwortgenerierung: {e}")
        return "Fehler bei der Antwortgenerierung."

# 🚀 Hauptprozess
def main():
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
            with col2:
                if st.button("👎 Antwort verbessern"):
                    save_feedback_jsonl(query_input, result, "👎", feedback_comment)

if __name__ == "__main__":
    main()
