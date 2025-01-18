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

# 📥 Verbesserte Antworten speichern
def save_corrected_answer(query, corrected_answer):
    feedback_entry = {
        "query": query,
        "corrected_answer": corrected_answer,
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open("corrected_answers.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(feedback_entry) + "\n")
    st.success("✅ Die verbesserte Antwort wurde gespeichert!")

# 🔎 Prüfen, ob es bereits eine verbesserte Antwort gibt
def load_corrected_answer(query):
    if os.path.exists("corrected_answers.jsonl"):
        with open("corrected_answers.jsonl", "r", encoding="utf-8") as file:
            for line in file:
                entry = json.loads(line)
                if entry["query"].lower() == query.lower():
                    return entry["corrected_answer"]
    return None

# 📝 Antwort generieren und verbesserte Antworten berücksichtigen
def generate_response_with_feedback(vectorstore, query, model, k=5):
    corrected_answer = load_corrected_answer(query)
    
    if corrected_answer:
        st.info("🔄 Diese Antwort basiert auf einer vorherigen Korrektur.")
        return corrected_answer

    # Standardantwort generieren, wenn keine Korrektur vorliegt
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

            # Feedbackbereich
            feedback_expander = st.expander("🛠️ Antwort verbessern")
            with feedback_expander:
                corrected_answer_input = st.text_area("Korrigiere die Antwort hier:")
                if st.button("💾 Verbesserte Antwort speichern"):
                    if corrected_answer_input:
                        save_corrected_answer(query_input, corrected_answer_input)
                    else:
                        st.warning("⚠️ Bitte gib eine korrekte Antwort ein.")

if __name__ == "__main__":
    main()
