import os
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datasets import load_dataset
import json

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

# 🧩 Chunks mit LangChain erstellen
def create_chunks(documents, chunk_size=10000, overlap=3000):
    """
    Teilt Dokumente in Chunks mit LangChain.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for doc in documents:
        splits = text_splitter.split_text(doc["content"])
        for chunk in splits:
            chunks.append(Document(page_content=chunk, metadata={"title": doc["title"], "url": doc["url"]}))
    return chunks

# 🌟 Vektorspeicher erstellen
def create_vector_store(chunks, persist_path="faiss_index"):
    """
    Erstellt einen FAISS-Vektorspeicher mit LangChain und speichert ihn.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    vectorstore.save_local(persist_path)
    return vectorstore

# 🌟 Vektorspeicher laden
# 🌟 Vektorspeicher laden
def load_vector_store(persist_path="faiss_index"):
    """
    Lädt einen vorhandenen FAISS-Vektorspeicher.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Fehler beim Laden des Vektorspeichers: {e}")
        return None


# 🔄 RAG-Prompt erstellen (ohne Passagen-Auswahl)
# 🔄 RAG-Prompt erstellen
def make_rag_prompt(query, relevant_passages, max_tokens=3000):
    """
    Erstellt ein RAG-Prompt mit den relevanten Passagen, wobei die Tokenanzahl begrenzt wird.
    """
    prompt_template = f"""
    Du bist ein hilfsbereiter Chatbot, der bei der Jobsuche unterstützt. Zeige die relevanten Informationen zu den Jobs an.

    Frage: {query}
    """
    token_count = len(prompt_template.split())
    max_tokens_allowed = max_tokens - token_count

    # Passagen hinzufügen, solange die Tokenanzahl unter dem Limit bleibt
    selected_passages = []
    for passage in relevant_passages:
        passage_text = passage.page_content.replace("\n", " ")
        if token_count + len(passage_text.split()) <= max_tokens_allowed:
            selected_passages.append(passage_text)
            token_count += len(passage_text.split())
        else:
            break

    # Füge die ausgewählten Passagen in das Prompt ein
    combined_passages = " ".join(selected_passages)
    return f"{prompt_template}\n\nBeschreibungen der Jobs: {combined_passages}\n\nAntwort:"


# 🕊 Antwort generieren
def generate_answer(prompt):
    """
    Generiert eine Antwort basierend auf einem Prompt.
    """
    model = genai.GenerativeModel(model_name="gemini-pro")
    return model.generate_content(prompt).text

# 🚀 Hauptprozess
def main():
    load_api_keys()
    st.set_page_config(page_title="RAG System mit LangChain und Gemini", page_icon=":robot:")
    st.header("RAG-System mit LangChain und Google Gemini")

    # Pfad zur JSONL-Datei
    jsonl_path = st.sidebar.text_input("Pfad zur JSONL-Datei:", "rag_data.jsonl")

    # Vektorspeicher-Pfad
    vectorstore_path = "faiss_index"

    # Vektorspeicher laden oder erstellen
    vectorstore = load_vector_store(vectorstore_path)
    if not vectorstore:
        if st.sidebar.button("Daten laden"):
            documents = load_jsonl(jsonl_path)
            st.success(f"{len(documents)} Dokumente geladen.")

            # Chunks erstellen und Vektorspeicher generieren
            with st.spinner("Erstelle Chunks und Vektorspeicher..."):
                chunks = create_chunks(documents)
                vectorstore = create_vector_store(chunks, persist_path=vectorstore_path)
                st.success("Vektorspeicher erfolgreich erstellt und gespeichert.")

    # Suchfunktion
    search_query = st.text_input("Suchanfrage eingeben:")
    if st.button("Suche durchführen"):
        if not vectorstore:
            st.warning("Bitte laden Sie die Daten zuerst.")
            return

        with st.spinner("Generiere Antwort..."):
            # Alle Dokumente aus dem Vektorspeicher abrufen
            documents = vectorstore.similarity_search(query="", k=1000)

            # Prompt erstellen
            prompt = make_rag_prompt(search_query, documents)
            
            # Antwort generieren
            answer = generate_answer(prompt)
            st.write("Antwort:")
            st.write(answer)

if __name__ == "__main__":
    main()
