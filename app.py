import requests
import base64

# 📤 Feedback in GitHub-Repo speichern
def push_feedback_to_github(file_path="user_feedback.jsonl"):
    """
    Speichert die aktualisierte Feedback-Datei ins GitHub-Repository.
    """
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")
    github_repo = os.getenv("GITHUB_REPO")
    file_name = "user_feedback.jsonl"
    github_api_url = f"https://api.github.com/repos/{github_repo}/contents/{file_name}"

    # 📂 Lade den aktuellen Inhalt von GitHub
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    response = requests.get(github_api_url, headers=headers)
    
    if response.status_code == 200:
        sha = response.json()["sha"]  # SHA für Update
    else:
        sha = None  # Datei existiert noch nicht

    # 📄 Lade lokale Datei
    with open(file_path, "rb") as file:
        content = base64.b64encode(file.read()).decode("utf-8")

    # 📤 Datei an GitHub senden
    data = {
        "message": "📥 Neues Feedback gespeichert",
        "content": content,
    }

    if sha:
        data["sha"] = sha  # Für Update notwendig

    response = requests.put(github_api_url, headers=headers, json=data)

    if response.status_code in [200, 201]:
        st.success("✅ Feedback erfolgreich auf GitHub gespeichert!")
    else:
        st.error(f"❌ Fehler beim Speichern auf GitHub: {response.json()}")

# 💬 Feedback speichern & auf GitHub hochladen
def save_feedback_jsonl(query, response, feedback_type, comment):
    """
    Speichert das Feedback lokal und pusht es anschließend zu GitHub.
    """
    feedback_entry = {
        "query": query,
        "response": response,
        "feedback": feedback_type,
        "comment": comment,
        "timestamp": datetime.datetime.now().isoformat()
    }

    # 📁 Lokales Speichern
    with open("user_feedback.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(feedback_entry) + "\n")

    # 📤 Auf GitHub pushen
    push_feedback_to_github()

    st.success("✅ Feedback gespeichert und zu GitHub hochgeladen!")
