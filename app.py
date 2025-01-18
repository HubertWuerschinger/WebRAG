import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datasets import load_dataset
import re

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
    return [{"content": doc["completion"], "url": doc["meta"].get("url", ""), "title": doc["meta"].get("title", "Kein Titel")} for doc in dataset["train"]]

# 📦 Vektorspeicher
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_texts(texts=[chunk["content"] for chunk in text_chunks], embedding=embeddings)

# 🔎 RAG-Suche
def rag_search(model, vectorstore, query, k=5):
    relevant_docs = vectorstore.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"Nutze diesen Kontext, um präzise zu antworten:\n\n{context}\n\nFrage:\n{query}\n\nAntworte kurz und präzise."
    response = model.generate_content(prompt)
    return response.text.strip()

# 🚀 Hauptprozess
def main():
    st.set_page_config(page_title="Körber AI Chatbot mit Feedback", page_icon="🤖")
    st.header("🔍 Körber AI Chatbot – Jetzt mit Feedback!")

    api_key = load_api_keys()
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 20,
        "max_output_tokens": 6000,
    }

    if "vectorstore" not in st.session_state:
        with st.spinner("Daten werden geladen..."):
            documents = load_koerber_data()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            text_chunks = [{"content": chunk, "url": doc["url"]} for doc in documents for chunk in text_splitter.split_text(doc["content"])]
            st.session_state.vectorstore = get_vector_store(text_chunks)

    query_input = st.text_input("Stellen Sie hier Ihre Frage:")
    generate_button = st.button("Antwort generieren")

    if generate_button and query_input:
        with st.spinner("Antwort wird generiert..."):
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config)
            response_text = rag_search(model, st.session_state.vectorstore, query_input)

            st.success("📝 Antwort:")
            st.write(response_text)

            # 📊 Feedback-Mechanismus
            st.subheader("📝 War die Antwort hilfreich?")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("👍 Ja"):
                    st.success("Danke für dein Feedback!")
                    # Optional: Ergebnis stärker gewichten
            with col2:
                if st.button("👎 Nein"):
                    feedback = st.text_area("Wie können wir die Antwort verbessern?")
                    if st.button("Feedback absenden"):
                        st.warning("Danke für dein Feedback! Wir arbeiten daran.")
                        # Optional: Feedback speichern
            with col3:
                improvement = st.text_input("Fehlende Info hinzufügen:")
                if st.button("Information speichern"):
                    st.success("Information hinzugefügt!")
                    # Optional: Info in den Vektorspeicher aufnehmen

if __name__ == "__main__":
    main()
