import os
import streamlit as st
import google.generativeai as genai
from datasets import load_dataset
import faiss
import numpy as np
import json

# 🔑 API-Schlüssel laden
def load_api_keys():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY fehlt. Bitte in der .env-Datei setzen.")
        st.stop()
    genai.configure(api_key=api_key)

# 🗋 JSONL-Daten laden
def load_jsonl(file_path):
    if not os.path.exists(file_path):
        st.error(f"Die Datei {file_path} wurde nicht gefunden.")
        st.stop()
    dataset = load_dataset("json", data_files={"train": file_path})
    return [
        {
            "content": doc["completion"],
            "url": doc["meta"].get("url", ""),
            "title": doc["meta"].get("title", "Unbekannter Titel")
        } for doc in dataset["train"]
    ]

# 🧩 Text in Chunks aufteilen
def create_chunks(documents, chunk_size=400, overlap=10):
    """
    Teilt Dokumente in überlappende Chunks auf.
    Args:
        documents (list): Liste von Dokumenten mit Textinhalten.
        chunk_size (int): Maximale Größe eines Chunks in Zeichen.
        overlap (int): Überlappung zwischen aufeinanderfolgenden Chunks.

    Returns:
        list: Liste von Chunks mit zugehörigen Metadaten.
    """
    chunks = []
    for doc in documents:
        content = doc["content"]
        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i:i + chunk_size]
            chunks.append({
                "content": chunk,
                "title": doc["title"],
                "url": doc["url"]
            })
    return chunks

# 🌟 Embedding erstellen
def generate_embeddings(chunks):
    model = "models/embedding-001"
    embeddings = [
        genai.embed_content(model=model, content=chunk["content"], task_type="retrieval_document")["embedding"]
        for chunk in chunks
    ]
    return np.array(embeddings, dtype="float32")

# 🔖 FAISS Index erstellen und speichern
def create_faiss_index(embeddings, chunks, faiss_path, metadata_path):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    metadata = {
        i: {"title": chunk["title"], "url": chunk["url"], "content": chunk["content"]}
        for i, chunk in enumerate(chunks)
    }

    # Speichere den FAISS-Index
    faiss.write_index(index, faiss_path)

    # Speichere die Metadaten
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    return index, metadata

# 🔖 FAISS Index und Metadaten laden
def load_faiss_index(faiss_path, metadata_path):
    if not os.path.exists(faiss_path) or not os.path.exists(metadata_path):
        return None, None

    index = faiss.read_index(faiss_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

# 🔎 Relevante Passagen abrufen
def get_relevant_passages(query, index, metadata, n_results=5):
    model = "models/embedding-001"
    query_embedding = genai.embed_content(model=model, content=query, task_type="retrieval_document")["embedding"]
    query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)

    distances, indices = index.search(query_embedding, n_results)
    return [metadata[str(idx)] for idx in indices[0]]

# 🔄 RAG-Prompt erstellen
def make_rag_prompt(query, relevant_passages):
    passages = " ".join([p["content"].replace("\n", " ") for p in relevant_passages])
    return f"""
    Du bist ein hilfsbereiter Chatbot, der bei der Jobsuche unterstützt. Zeige die Jobbeschreibungen und das Profil und den standort an

    Frage: {query}
    Beschreibung: {passages}

    Antwort:
    """

# 🕊 Antwort generieren
def generate_answer(prompt):
    model = genai.GenerativeModel(model_name="gemini-pro")
    return model.generate_content(prompt).text

# 🚀 Hauptprozess
def main():
    load_api_keys()
    st.set_page_config(page_title="RAG System mit JSONL und Gemini", page_icon=":robot:")
    st.header("RAG-System mit Google Gemini und JSONL-Daten")

    # Pfade für FAISS und Metadaten
    faiss_path = "faiss_index.bin"
    metadata_path = "faiss_metadata.json"

    # JSONL-Daten laden
    jsonl_path = st.sidebar.text_input("Pfad zur JSONL-Datei:", "rag_data.jsonl")

    # Globale Variablen für Index und Metadaten
    global index, metadata

    # FAISS und Metadaten laden oder erstellen
    index, metadata = load_faiss_index(faiss_path, metadata_path)

    if index is None or metadata is None:
        if st.sidebar.button("Daten laden"):
            documents = load_jsonl(jsonl_path)
            st.success(f"{len(documents)} Dokumente geladen.")

            # Chunks erstellen und Embeddings generieren
            with st.spinner("Erstelle Chunks und Embeddings..."):
                chunks = create_chunks(documents)
                embeddings = generate_embeddings(chunks)
                index, metadata = create_faiss_index(embeddings, chunks, faiss_path, metadata_path)
                st.success("FAISS-Index erfolgreich erstellt.")

    # Suchfunktion
    search_query = st.text_input("Suchanfrage eingeben:")

    # Überprüfen, ob Index und Metadaten geladen wurden
    if index is None or metadata is None:
        st.warning("Bitte laden Sie die Daten zuerst über die Seitenleiste.")
        return

    if st.button("Suche durchführen"):
        with st.spinner("Suche relevante Passagen..."):
            relevant_passages = get_relevant_passages(search_query, index, metadata)

            if relevant_passages:
                st.write("Relevante Passagen:")
                for passage in relevant_passages:
                    st.write(f"**{passage['title']}**")
                    st.write(f"[Zur Jobbeschreibung]({passage['url']})")
                    st.write(passage["content"])
                    st.markdown("---")

                # Antwort generieren
                prompt = make_rag_prompt(search_query, relevant_passages)
                answer = generate_answer(prompt)
                st.write("Antwort:")
                st.write(answer)
            else:
                st.warning("Keine relevanten Passagen gefunden.")

if __name__ == "__main__":
    main()
