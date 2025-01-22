import os
import json
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datasets import load_dataset

# 🔑 API-Schlüssel laden
def load_api_keys():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY fehlt. Bitte in der .env-Datei setzen.")
        st.stop()
    genai.configure(api_key=api_key)

# 🗋 JSONL-Daten laden
def load_jsonl(file_path):
    """
    Lädt die JSONL-Datei und konvertiert sie in ein Dokumentformat.
    """
    if not os.path.exists(file_path):
        st.error(f"Die Datei {file_path} wurde nicht gefunden.")
        st.stop()
    try:
        dataset = load_dataset("json", data_files={"train": file_path})
        return [
            {
                "content": doc["completion"],
                "url": doc["meta"].get("url", ""),
                "title": doc["meta"].get("title", "Unbekannter Titel")
            }
            for doc in dataset["train"]
        ]
    except Exception as e:
        st.error(f"Fehler beim Laden der JSONL-Daten: {e}")
        st.stop()

# 🧩 Chunks erstellen
def create_chunks(documents, chunk_size=6000, overlap=600):
    """
    Teilt Dokumente in Chunks, wobei größere Textabschnitte unterstützt werden,
    und zeigt maximal 3 Chunks über alle Dokumente hinweg in Streamlit an.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    displayed_chunks = 0  # Globaler Zähler für angezeigte Chunks

    st.write("Erstellte Chunks (maximal 3 werden angezeigt):")
    for idx, doc in enumerate(documents):
        splits = text_splitter.split_text(doc["content"])
        for i, chunk in enumerate(splits):
            chunk_metadata = {"title": doc["title"], "url": doc["url"]}
            chunks.append(Document(page_content=chunk, metadata=chunk_metadata))

            # Anzeige der ersten 3 Chunks
            if displayed_chunks < 3:
                st.write(f"**Dokument {idx + 1}, Chunk {i + 1}:**")
                st.write(f"Titel: {chunk_metadata['title']}")
                st.write(f"URL: {chunk_metadata['url']}")
                st.write(f"Inhalt: {chunk[:300]}...")  # Begrenzte Zeichen für Übersicht
                st.markdown("---")  # Trenner für bessere Lesbarkeit
                displayed_chunks += 1

            # Breche die Anzeige ab, sobald 3 Chunks erreicht wurden
            if displayed_chunks >= 3:
                return chunks  # Chunks vollständig zurückgeben, auch wenn Anzeige begrenzt ist

    return chunks

# 🌟 Vektorspeicher erstellen
def create_vector_store(chunks, persist_path="faiss_index"):
    """
    Erstellt einen FAISS-Vektorspeicher und speichert ihn lokal.
    """
    try:
        from vertexai.language_models import TextEmbeddingModel

        # Vertex AI Embeddings-Modell initialisieren
        model = TextEmbeddingModel.from_pretrained("text-embedding-005")

        # Embeddings für jeden Chunk erstellen
        embeddings = []
        for chunk in chunks:
            embedding = model.get_embeddings(
                texts=[chunk.page_content],  # Text aus dem Chunk
                task_type="RETRIEVAL_DOCUMENT"  # Dokument als Eingabetyp spezifizieren
            )[0].values  # Das erste Ergebnis (Einbettungsvektor) nehmen
            embeddings.append(embedding)

        # FAISS-Vektorspeicher erstellen
        vectorstore = FAISS.from_texts(
            texts=[chunk.page_content for chunk in chunks],  # Inhalte der Chunks
            embedding=embeddings,  # Generierte Embeddings
            metadatas=[chunk.metadata for chunk in chunks]  # Metadaten der Chunks
        )

        # FAISS-Vektorspeicher speichern
        vectorstore.save_local(persist_path)
        st.success(f"Vektorspeicher erfolgreich in {persist_path} erstellt.")
        return vectorstore

    except Exception as e:
        st.error(f"Fehler beim Erstellen des Vektorspeichers: {e}")
        st.stop()

# 🌟 Vektorspeicher laden
def load_vector_store(persist_path="faiss_index"):
    """
    Lädt einen vorhandenen FAISS-Vektorspeicher.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    try:
        # Aktiviert gefährliche Deserialisierung, da die Datei vertrauenswürdig ist
        return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
    except FileNotFoundError:
        st.warning("Vektorspeicher nicht gefunden. Bitte laden Sie die Daten und erstellen Sie den Speicher.")
        return None
    except Exception as e:
        st.error(f"Fehler beim Laden des Vektorspeichers: {e}")
        return None

# 🔄 RAG-Prompt erstellen
def make_rag_prompt(query, documents):
    """
    Erstellt einen RAG-Prompt basierend auf der Benutzeranfrage und den Dokumenten.
    """
    combined_content = "\n\n".join(
        [f"Titel: {doc.metadata['title']}\nURL: {doc.metadata['url']}\nInhalt: {doc.page_content}" for doc in documents]
    )
    return f"""
    Du bist ein hilfsbereiter Chatbot, der bei der Suche auf der Körberseite unterstützt. Antworte strukturiert.

    Frage: {query}
    Körber Website:
    {combined_content}

    Antwort:
    """


# 🔄 Suchanfragen- und Ergebnis-Logging
def log_search_query(query, results, log_path="search_log.json"):
    """
    Protokolliert Suchanfragen und Ergebnisse in einer JSON-Datei.
    """
    log_entry = {
        "query": query,
        "results": [
            {
                "title": doc.metadata["title"],
                "url": doc.metadata["url"],
                "content_excerpt": doc.page_content[:300]  # Begrenzte Zeichen
            }
            for doc in results
        ]
    }

    # Überprüfe, ob die Datei existiert, und lade bestehende Logs
    log_data = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                log_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            st.warning("Die Logdatei ist leer oder enthält ungültige Daten. Sie wird zurückgesetzt.")
            log_data = []

    # Füge neuen Eintrag hinzu und speichere die Datei
    log_data.append(log_entry)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=4)


# 🔄 Letzte Anfragen anzeigen
def display_recent_queries(log_path="search_log.json", max_queries=5):
    """
    Zeigt die letzten Suchanfragen an.
    """
    if not os.path.exists(log_path):
        return

    with open(log_path, "r") as f:
        log_data = json.load(f)

    recent_queries = log_data[-max_queries:]

    st.subheader("Letzte Suchanfragen:")
    for entry in recent_queries:
        st.write(f"**Anfrage:** {entry['query']}")
        for result in entry["results"]:
            st.write(f"- {result['title']} ({result['url']})")


# 🕊 Antwort generieren
def generate_answer(prompt):
    """
    Generiert eine Antwort basierend auf einem Prompt.
    """
    try:
        model = genai.GenerativeModel(model_name="gemini-pro")
        return model.generate_content(prompt).text
    except Exception as e:
        st.error(f"Fehler bei der Antwortgenerierung: {e}")
        return "Es konnte keine Antwort generiert werden."
        
    
# 🔄 Letzte Anfragen anzeigen
def display_recent_queries(log_path="search_log.json", max_queries=1):
    """
    Zeigt die letzten max_queries Suchanfragen aus der Logdatei an.
    """
    if not os.path.exists(log_path):
        st.info("Keine Suchanfragen bisher.")
        return

    with open(log_path, "r") as f:
        log_data = json.load(f)

    recent_queries = log_data[-max_queries:]  # Nur die letzten max_queries Anfragen

    st.subheader("Letzte Suchanfragen:")
    for entry in recent_queries:
        st.write(f"**Anfrage:** {entry['query']}")
        for result in entry["results"]:
            st.write(f"- {result['title']} ({result['url']})")
        st.markdown("---")  # Trenner zwischen Anfragen


    
def similarity_search_with_fallback(query, vectorstore, k=5):
    """
    Führt eine semantische Suche durch. Falls keine Schlüsselwortübereinstimmungen gefunden werden,
    wird der lose Inhalt der Chunks genutzt.
    
    Args:
        query (str): Die Suchanfrage des Benutzers.
        vectorstore (FAISS): Der Vektorspeicher mit den Chunks.
        k (int): Die Anzahl der zurückzugebenden Ergebnisse.
    
    Returns:
        list: Relevante Dokumente aus dem Vektorspeicher.
    """
    # Primäre Suche (semantisch)
    relevant_documents = vectorstore.similarity_search(query, k=k)

    if relevant_documents:
        return relevant_documents

    # Fallback-Suche: Lose Inhalte ohne genaue semantische Übereinstimmung
    st.warning("Keine exakten Ergebnisse gefunden. Suche in allgemeinen Inhalten...")
    all_documents = vectorstore.similarity_search(query="", k=k)
    return all_documents
# Letzte Anfragen anzeigen
def display_recent_queries(log_path="search_log.json", max_queries=3):
    """
    Zeigt die letzten Suchanfragen an.
    """
    if not os.path.exists(log_path):
        st.write("Keine Suchanfragen verfügbar.")
        return

    try:
        with open(log_path, "r") as f:
            log_data = json.load(f)
        recent_queries = log_data[-max_queries:]
        st.subheader("Letzte Suchanfragen:")
        for entry in recent_queries:
            st.write(f"**Anfrage:** {entry['query']}")
            for result in entry["results"]:
                st.write(f"- {result['title']} ({result['url']})")
    except json.JSONDecodeError:
        st.warning("Die Logdatei ist leer oder beschädigt.")


# 🚀 Hauptprozess
def main():
    load_api_keys()
    st.set_page_config(page_title="Frag Körber", page_icon=":robot:")
    

    # Körber-Logo anzeigen (kleiner)
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/8/88/Koerber_Logo_black.svg",
        width=200,  # Breite in Pixel
        
    )
    st.header("Frag Körber")
    # Pfad zur JSONL-Datei
    jsonl_path = st.sidebar.text_input("Pfad zur JSONL-Datei:", "cleaned_rag_data.jsonl")

    # Vektorspeicher-Pfad
    vectorstore_path = "faiss_index"

    if st.sidebar.button("Daten laden"):
        documents = load_jsonl(jsonl_path)
        st.success(f"{len(documents)} Dokumente geladen.")
        with st.spinner("Erstelle Chunks und Vektorspeicher..."):
            chunks = create_chunks(documents)
            vectorstore = create_vector_store(chunks, persist_path=vectorstore_path)
            st.success("Vektorspeicher erfolgreich erstellt und gespeichert.")
    else:
        vectorstore = load_vector_store(vectorstore_path)

    if not vectorstore:
        st.warning("Vektorspeicher ist leer. Bitte laden Sie die Daten.")
        return





    # Letzte Suchanfragen abrufen
    log_path = "search_log.json"
    previous_queries = []

    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                log_data = json.load(f)
            previous_queries = [entry["query"] for entry in log_data][-10:]  # Max. letzte 10 Anfragen
        except json.JSONDecodeError:
            st.warning("Die Logdatei ist leer oder enthält ungültige Daten. Sie wird zurückgesetzt.")
            previous_queries = []


    # Auswahl der letzten Anfrage
    selected_query = st.selectbox("Letzte Suchanfragen auswählen (optional):", [""] + previous_queries)

    # Eingabefeld mit Echo
    search_query = st.text_input("Suchanfrage eingeben:", value=selected_query)

    if st.button("Suche durchführen"):
        with st.spinner("Suche relevante Dokumente..."):
            # Verwende die Fallback-Suchfunktion
            relevant_documents = similarity_search_with_fallback(search_query, vectorstore, k=5)
            if relevant_documents:
                prompt = make_rag_prompt(search_query, relevant_documents)
                answer = generate_answer(prompt)
                st.write("Antwort:")
                st.write(answer)
                log_search_query(search_query, relevant_documents)
            else:
                st.warning("Keine relevanten Dokumente gefunden.")




if __name__ == "__main__":
    main()
