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
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    github_token = os.getenv("GITHUB_TOKEN")
    github_repo = os.getenv("GITHUB_REPO")

    if not api_key or not github_token or not github_repo:
        st.error("API- oder GitHub-Schlüssel fehlen. Bitte die .env-Datei prüfen.")
        st.stop()
    return api_key, github_token, github_repo
    
# ✅ GitHub-Zugriffsprüfung
def check_github_access(github_token, github_repo):
    """
    Überprüft den Zugriff auf das GitHub-Repository und die Schreibrechte.
    """
    try:
        g = Github(github_token)
        repo = g.get_repo(github_repo)
        
        # Test: Repository Informationen abrufen
        st.success(f"✅ Verbindung zu {github_repo} erfolgreich!")

        # Test: Schreibrechte prüfen durch temporäre Datei
        test_file_path = "test_access.txt"
        repo.create_file(test_file_path, "Testzugriff", "Dies ist ein Test.", branch="main")
        repo.delete_file(test_file_path, "Testzugriff gelöscht", repo.get_contents(test_file_path).sha, branch="main")
        st.success("📝 Schreibrechte erfolgreich getestet!")

    except Exception as e:
        st.error(f"❌ GitHub-Zugriffsfehler: {e}")
        st.stop()

# 📂 Körber-Daten laden
def load_koerber_data():
    dataset = load_dataset("json", data_files={"train": "koerber_data.jsonl"})
    return [{"content": doc["completion"], "url": doc["meta"].get("url", ""), 
             "timestamp": doc["meta"].get("timestamp", ""), "title": doc["meta"].get("title", "Kein Titel")} 
            for doc in dataset["train"]]

# 📦 Vektorspeicher erstellen (inkl. Feedback)
def get_vector_store(text_chunks, github_token, github_repo):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        # Feedback von GitHub laden und integrieren
        feedback_data = load_feedback_from_github(github_token, github_repo)
        feedback_chunks = [{"content": entry["response"]} for entry in feedback_data]
        combined_chunks = text_chunks + feedback_chunks

        return FAISS.from_texts(texts=[chunk["content"] for chunk in combined_chunks], embedding=embeddings)
    except Exception as e:
        st.error(f"Fehler beim Erstellen des Vektorspeichers: {e}")
        return None

# 📥 Feedback von GitHub laden
def load_feedback_from_github(github_token, github_repo):
    try:
        g = Github(github_token)
        repo = g.get_repo(github_repo)
        contents = repo.get_contents("user_feedback.jsonl")
        feedback_data = contents.decoded_content.decode().splitlines()
        return [json.loads(entry) for entry in feedback_data]
    except Exception:
        return []

# 🔍 Schlagwörter extrahieren
def extract_keywords_with_llm(model, query):
    prompt = f"Extrahiere relevante Schlagwörter aus dieser Anfrage:\n\n{query}\n\nNur Schlagwörter ohne Erklärungen."
    try:
        response = model.generate_content(prompt)
        return re.findall(r'\b\w{3,}\b', response.text)
    except Exception as e:
        st.error(f"Fehler bei der Schlagwort-Extraktion: {e}")
        return []

# 💬 Feedback auf GitHub speichern
def save_feedback_to_github(github_token, github_repo, feedback_entry):
    try:
        g = Github(github_token)
        repo = g.get_repo(github_repo)
        file_path = "user_feedback.jsonl"

        try:
            contents = repo.get_contents(file_path)
            existing_content = contents.decoded_content.decode()
            updated_content = existing_content + json.dumps(feedback_entry) + "\n"
            repo.update_file(contents.path, "Feedback aktualisiert", updated_content, contents.sha)
        except Exception:
            repo.create_file(file_path, "Feedback-Datei erstellt", json.dumps(feedback_entry) + "\n")
        
        st.success("✅ Feedback wurde erfolgreich auf GitHub gespeichert!")
    except Exception as e:
        st.error(f"❌ Fehler beim Speichern in GitHub: {e}")

# 💬 Feedback speichern
def save_feedback(query, response, feedback_type, comment, github_token, github_repo):
    feedback_entry = {
        "query": query,
        "response": response,
        "feedback": feedback_type,
        "comment": comment,
        "timestamp": datetime.datetime.now().isoformat()
    }

    save_feedback_to_github(github_token, github_repo, feedback_entry)

# 📊 Letzte Feedback-Einträge anzeigen
def show_last_feedback_entries(github_token, github_repo):
    feedback_data = load_feedback_from_github(github_token, github_repo)[-3:]
    st.markdown("### 📄 **Letzte Feedback-Einträge:**")
    for entry in feedback_data:
        st.json(entry)

# 📝 Antwort generieren mit Feedback
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

# 🚀 Hauptprozess
def main():
    api_key, github_token, github_repo = load_api_keys()
    genai.configure(api_key=api_key)

    st.set_page_config(page_title="Körber AI Chatbot", page_icon=":factory:")
    st.header("🔍 Wie können wir dir weiterhelfen?")

    # 🔍 GitHub-Zugriff prüfen
    check_github_access(github_token, github_repo)

    if "vectorstore" not in st.session_state:
        with st.spinner("Daten werden geladen..."):
            documents = load_koerber_data()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=600)
            text_chunks = [{"content": chunk, "url": doc["url"]} for doc in documents for chunk in text_splitter.split_text(doc["content"])]
            st.session_state.vectorstore = get_vector_store(text_chunks, github_token, github_repo)

    query_input = st.text_input("Stellen Sie hier Ihre Frage:", value="")
    generate_button = st.button("Antwort generieren")

    if generate_button and query_input:
        with st.spinner("Antwort wird generiert..."):
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
            result = generate_response_with_feedback(st.session_state.vectorstore, query_input, model)

            st.success("📝 Antwort:")
            st.write(result)

            feedback_comment = st.text_input("Korrekte Antwort eingeben (optional):")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Antwort war hilfreich"):
                    save_feedback(query_input, result, "👍", feedback_comment, github_token, github_repo)
            with col2:
                if st.button("👎 Antwort verbessern"):
                    if feedback_comment.strip():
                        save_feedback(query_input, feedback_comment, "👎", feedback_comment, github_token, github_repo)

            show_last_feedback_entries(github_token, github_repo)

if __name__ == "__main__":
    main()
