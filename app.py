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
from github import Github

# 🔑 API-Schlüssel laden
def load_api_keys():
    """
    Lädt den Google API-Schlüssel und GitHub Token aus der .env-Datei.
    
    Returns:
        tuple: Google API-Key, GitHub Token, Repository Name
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    github_token = os.getenv("GITHUB_TOKEN")
    github_repo = os.getenv("GITHUB_REPO")
    
    if not api_key or not github_token or not github_repo:
        st.error("API- oder GitHub-Schlüssel fehlen. Bitte die .env-Datei prüfen.")
        st.stop()
    return api_key, github_token, github_repo

# 📂 Körber-Daten laden
def load_koerber_data():
    """
    Lädt die Körber-Daten aus einer JSONL-Datei und gibt eine Liste der Inhalte zurück.
    """
    dataset = load_dataset("json", data_files={"train": "koerber_data.jsonl"})
    return [{"content": doc["completion"], "url": doc["meta"].get("url", ""), 
             "timestamp": doc["meta"].get("timestamp", ""), "title": doc["meta"].get("title", "Kein Titel")} 
            for doc in dataset["train"]]

# 📦 Vektorspeicher erstellen
def get_vector_store(text_chunks):
    """
    Erstellt einen FAISS-Vektorspeicher aus Text-Chunks.
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
    """
    prompt = f"Extrahiere relevante Schlagwörter aus dieser Anfrage:\n\n{query}\n\nNur Schlagwörter ohne Erklärungen."
    try:
        response = model.generate_content(prompt)
        return re.findall(r'\b\w{3,}\b', response.text)
    except Exception as e:
        st.error(f"Fehler bei der Schlagwort-Extraktion: {e}")
        return []

# 💬 Feedback in GitHub speichern
def save_feedback_to_github(github_token, github_repo, feedback_entry):
    """
    Speichert das Feedback direkt in das GitHub-Repository.
    """
    try:
        g = Github(github_token)
        repo = g.get_repo(github_repo)
        file_path = "user_feedback.jsonl"

        contents = repo.get_contents(file_path)
        existing_content = contents.decoded_content.decode()

        updated_content = existing_content + json.dumps(feedback_entry) + "\n"

        repo.update_file(contents.path, "Feedback aktualisiert", updated_content, contents.sha)
        st.success("✅ Feedback wurde erfolgreich auf GitHub gespeichert!")
    except Exception as e:
        st.error(f"❌ Fehler beim Speichern in GitHub: {e}")

# 💬 Feedback lokal und auf GitHub speichern
def save_feedback(query, response, feedback_type, comment, github_token, github_repo):
    """
    Speichert Feedback lokal und auf GitHub.
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

    save_feedback_to_github(github_token, github_repo, feedback_entry)

# 📊 Letzte Feedback-Einträge anzeigen
def show_last_feedback_entries():
    """
    Zeigt die letzten 3 Feedback-Einträge aus GitHub an.
    """
    try:
        g = Github(os.getenv("GITHUB_TOKEN"))
        repo = g.get_repo(os.getenv("GITHUB_REPO"))
        contents = repo.get_contents("user_feedback.jsonl")
        feedback_data = contents.decoded_content.decode().splitlines()[-3:]

        st.markdown("### 📄 **Letzte Feedback-Einträge:**")
        for entry in feedback_data:
            st.json(json.loads(entry))
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der GitHub-Daten: {e}")

# 📝 Antwort generieren
def generate_response_with_feedback(vectorstore, query, model, k=5):
    """
    Generiert eine Antwort unter Berücksichtigung von Feedback und dem Vektorspeicher.
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
    api_key, github_token, github_repo = load_api_keys()
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
                    save_feedback(query_input, result, "👍", feedback_comment, github_token, github_repo)
            with col2:
                if st.button("👎 Antwort verbessern"):
                    save_feedback(query_input, feedback_comment, "👎", feedback_comment, github_token, github_repo)

            show_last_feedback_entries()

if __name__ == "__main__":
    main()
