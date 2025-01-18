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

# 1️⃣ 🔑 API-Schlüssel laden
def load_api_keys():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    github_token = os.getenv("GITHUB_TOKEN")
    github_repo = os.getenv("GITHUB_REPO")

    if not api_key or not github_token or not github_repo:
        st.error("API- oder GitHub-Schlüssel fehlen. Bitte die .env-Datei prüfen.")
        st.stop()
    return api_key, github_token, github_repo

# 2️⃣ ✅ GitHub-Zugriffsprüfung
def check_github_access(github_token, github_repo):
    try:
        g = Github(github_token)
        repo = g.get_repo(github_repo)
        st.success(f"✅ Verbindung zu {github_repo} erfolgreich!")
    except Exception as e:
        st.error(f"❌ GitHub-Zugriffsfehler: {e}")
        st.stop()

# 3️⃣ 📂 Körber-Daten laden
def load_koerber_data():
    dataset = load_dataset("json", data_files={"train": "koerber_data.jsonl"})
    return [{"content": doc["completion"], "url": doc["meta"].get("url", ""), 
             "timestamp": doc["meta"].get("timestamp", ""), "title": doc["meta"].get("title", "Kein Titel")} 
            for doc in dataset["train"]]

# 4️⃣ 📥 Feedback von GitHub laden
def load_feedback_from_github(github_token, github_repo):
    try:
        g = Github(github_token)
        repo = g.get_repo(github_repo)
        contents = repo.get_contents("user_feedback.jsonl")
        feedback_data = contents.decoded_content.decode().splitlines()
        return [json.loads(entry) for entry in feedback_data]
    except Exception:
        return []

# 5️⃣ 📦 Vektorspeicher erstellen (mit Feedback)
def get_vector_store(text_chunks, github_token, github_repo):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        feedback_data = load_feedback_from_github(github_token, github_repo)
        feedback_chunks = [{"content": entry["response"]} for entry in feedback_data]
        combined_chunks = text_chunks + feedback_chunks
        return FAISS.from_texts(texts=[chunk["content"] for chunk in combined_chunks], embedding=embeddings)
    except Exception as e:
        st.error(f"Fehler beim Erstellen des Vektorspeichers: {e}")
        return None

# 6️⃣ 🔍 Schlagwörter extrahieren
def extract_keywords_with_llm(model, query):
    prompt = f"Extrahiere relevante Schlagwörter aus dieser Anfrage:\n\n{query}\n\nNur Schlagwörter ohne Erklärungen."
    try:
        response = model.generate_content(prompt)
        return re.findall(r'\b\w{3,}\b', response.text)
    except Exception as e:
        st.error(f"Fehler bei der Schlagwort-Extraktion: {e}")
        return []

# 7️⃣ 💬 Feedback auf GitHub speichern (SHA-Schutz)
def save_feedback_to_github(github_token, github_repo, feedback_entry):
    try:
        g = Github(github_token)
        repo = g.get_repo(github_repo)
        file_path = "user_feedback.jsonl"

        contents = repo.get_contents(file_path)
        sha = contents.sha
        existing_content = contents.decoded_content.decode()
        updated_content = existing_content + json.dumps(feedback_entry, ensure_ascii=False) + "\n"

        repo.update_file(contents.path, "Feedback aktualisiert", updated_content, sha)
        st.success("✅ Feedback wurde sicher auf GitHub gespeichert!")
    except Exception as e:
        st.error(f"❌ Fehler beim Speichern in GitHub: {e}")

# 8️⃣ 📊 Letzte Feedback-Einträge anzeigen
def show_last_feedback_entries(github_token, github_repo):
    feedback_data = load_feedback_from_github(github_token, github_repo)[-3:]
    st.markdown("### 📄 **Letzte Feedback-Einträge:**")
    for entry in feedback_data:
        st.json(entry)

# 9️⃣ 📝 Antwort generieren
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


# 📂 Schreib- und Leseprüfung der Feedback-Datei
def check_feedback_file_access(github_token, github_repo, file_path="user_feedback.jsonl"):
    """
    Überprüft, ob die Feedback-Datei auf GitHub lesbar und beschreibbar ist.

    Args:
        github_token (str): GitHub-Zugriffstoken.
        github_repo (str): Repository-Name.
        file_path (str): Pfad zur Feedback-Datei im Repository.

    Raises:
        Streamlit-Fehler: Falls kein Zugriff möglich ist.
    """
    try:
        g = Github(github_token)
        repo = g.get_repo(github_repo)

        # ✅ Lesbarkeit prüfen
        contents = repo.get_contents(file_path)
        st.success(f"📖 Lesbarkeit von {file_path} bestätigt!")

        # ✍️ Schreibbarkeit testen (temporären Eintrag hinzufügen und löschen)
        test_entry = {
            "query": "Testzugriff",
            "response": "Dies ist ein Schreibtest.",
            "feedback": "✅",
            "comment": "Automatischer Testeintrag",
            "timestamp": datetime.datetime.now().isoformat()
        }

        updated_content = contents.decoded_content.decode() + json.dumps(test_entry) + "\n"
        repo.update_file(contents.path, "🔎 Testeintrag hinzugefügt", updated_content, contents.sha)
        st.success("✍️ Schreibbarkeit der Feedback-Datei bestätigt!")

        # 🔄 Testeintrag wieder entfernen
        latest_contents = repo.get_contents(file_path)
        cleaned_content = "\n".join(latest_contents.decoded_content.decode().splitlines()[:-1])
        repo.update_file(latest_contents.path, "🧹 Testeintrag entfernt", cleaned_content, latest_contents.sha)
        st.success("🧹 Testeintrag erfolgreich entfernt!")

    except Exception as e:
        st.error(f"❌ Fehler beim Zugriff auf {file_path}: {e}")
        st.stop()




# 🔟 🚀 Hauptprozess
def main():
    api_key, github_token, github_repo = load_api_keys()
    genai.configure(api_key=api_key)

    st.set_page_config(page_title="Körber AI Chatbot", page_icon=":factory:")
    st.header("🔍 Wie können wir dir weiterhelfen?")

    # ✅ GitHub-Zugriff und Feedback-Datei prüfen
    check_github_access(github_token, github_repo)
    check_feedback_file_access(github_token, github_repo)  # 👈 Schreib-/Leseprüfung für Feedback-Datei

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
                    save_feedback_to_github(github_token, github_repo, {
                        "query": query_input, "response": result, "feedback": "👍", "comment": feedback_comment, "timestamp": datetime.datetime.now().isoformat()
                    })
            with col2:
                if st.button("👎 Antwort verbessern"):
                    save_feedback_to_github(github_token, github_repo, {
                        "query": query_input, "response": feedback_comment, "feedback": "👎", "comment": feedback_comment, "timestamp": datetime.datetime.now().isoformat()
                    })

            show_last_feedback_entries(github_token, github_repo)

if __name__ == "__main__":
    main()
