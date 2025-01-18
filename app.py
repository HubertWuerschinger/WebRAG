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

# 📂 Feedback-Dateipfad (lokal)
FEEDBACK_FILE_PATH = "user_feedback.jsonl"

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
    return [{"content": doc["completion"], "url": doc["meta"].get("url", ""), 
             "timestamp": doc["meta"].get("timestamp", ""), "title": doc["meta"].get("title", "Kein Titel")} 
            for doc in dataset["train"]]

# 📥 Feedback lokal laden
def load_feedback_locally():
    if not os.path.exists(FEEDBACK_FILE_PATH):
        return []
    with open(FEEDBACK_FILE_PATH, "r", encoding="utf-8") as file:
        feedback_data = file.readlines()
    return [json.loads(entry) for entry in feedback_data]

# 📦 Vektorspeicher erstellen (inkl. Feedback)
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        feedback_data = load_feedback_locally()
        feedback_chunks = [{"content": entry["completion"]} for entry in feedback_data]
        combined_chunks = text_chunks + feedback_chunks

        return FAISS.from_texts(texts=[chunk["content"] for chunk in combined_chunks], embedding=embeddings)
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

# 💬 Feedback im gewünschten Format lokal speichern
def save_feedback_locally(prompt, completion, comment=""):
    feedback_entry = {
        "prompt": prompt,
        "completion": completion,
        "meta": {
            "title": "Benutzerdefiniertes Feedback",
            "url": "N/A",
            "timestamp": datetime.datetime.now().isoformat()
        }
    }
    try:
        with open(FEEDBACK_FILE_PATH, "a", encoding="utf-8") as file:
            file.write(json.dumps(feedback_entry, ensure_ascii=False) + "\n")
        st.success("✅ Feedback wurde lokal gespeichert!")
    except Exception as e:
        st.error(f"❌ Fehler beim Speichern des Feedbacks: {e}")

# 🔄 Vektorspeicher nach Feedback aktualisieren
def update_vector_store_with_feedback(completion):
    if "vectorstore" in st.session_state and completion:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        try:
            st.session_state.vectorstore.add_texts([completion], embedding=embeddings)
            st.success("🧠 Vektorspeicher wurde mit dem neuen Feedback aktualisiert!")
        except Exception as e:
            st.error(f"❌ Fehler beim Aktualisieren des Vektorspeichers: {e}")

# 📊 Letzte Feedback-Einträge anzeigen
def show_last_feedback_entries():
    feedback_data = load_feedback_locally()[-3:]
    st.markdown("### 📄 **Letzte Feedback-Einträge:**")
    for entry in feedback_data:
        st.json(entry)

# 📝 Antwort generieren
def generate_response_with_feedback(vectorstore, query, model, k=5):
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

# 💾 Feedback-Button Callback
def feedback_callback(feedback_type):
    # Feedback als Antwort oder Kommentar speichern
    feedback_response = st.session_state.feedback_comment if feedback_type == "👎" and st.session_state.feedback_comment.strip() else st.session_state.generated_result

    # Feedback im neuen Format speichern
    save_feedback_locally(st.session_state.query_input, feedback_response)
    update_vector_store_with_feedback(feedback_response)
    st.session_state.feedback_saved = True

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

    st.session_state.query_input = st.text_input("Stellen Sie hier Ihre Frage:", value=st.session_state.get("query_input", ""))
    generate_button = st.button("Antwort generieren")

    if generate_button and st.session_state.query_input:
        with st.spinner("Antwort wird generiert..."):
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
            st.session_state.generated_result = generate_response_with_feedback(st.session_state.vectorstore, st.session_state.query_input, model)

            st.success("📝 Antwort:")
            st.write(st.session_state.generated_result)

    st.session_state.feedback_comment = st.text_input("Korrekte Antwort eingeben (optional):", value=st.session_state.get("feedback_comment", ""))

    col1, col2 = st.columns(2)

    # 👍 Feedback-Button
    col1.button("👍 Antwort war hilfreich", on_click=feedback_callback, args=("👍",))

    # 👎 Feedback-Button
    col2.button("👎 Antwort verbessern", on_click=feedback_callback, args=("👎",))

    if st.session_state.get("feedback_saved"):
        show_last_feedback_entries()
        st.session_state.feedback_saved = False

if __name__ == "__main__":
    main()
