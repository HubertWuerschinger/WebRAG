import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datasets import load_dataset

# --- Körber-Daten aus der JSON-Datei laden ---
def load_koerber_data():
    dataset = load_dataset("json", data_files={"train": "koerber_data.jsonl"})
    documents = [{"content": doc["completion"], "url": doc["meta"]["url"]} for doc in dataset["train"]]
    return documents

# --- Vektorspeicher erstellen ---
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vectorstore = FAISS.from_texts(texts=[chunk["content"] for chunk in text_chunks], embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.warning(f"Fehler beim Erstellen des Vektorspeichers: {e}")

# --- Antwort generieren ---
def get_response(context, question, model):
    prompt_template = f"""
    Du bist ein hilfreicher Assistent, der Fragen basierend auf dem folgenden Kontext beantwortet und diese strukturiert ausgibt. Sowie ein Experte für Logistiksysteme ist:

    Kontext: {context}\n
    Frage: {question}\n
    """
    try:
        response = model.generate_content(prompt_template)
        return response.text
    except Exception as e:
        st.warning(f"Fehler bei der Generierung: {e}")
        return ""

# --- Hauptprozess ---
def main():
    load_dotenv()
    st.set_page_config(page_title="Körber AI Chatbot", page_icon=":factory:")
    st.header("🔍 Stell deine Fragen")

    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

    generation_config = {
        "temperature": 0.2,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 8000,
    }

    # --- Session State initialisieren ---
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "documents" not in st.session_state:
        st.session_state.documents = []

    # --- Vektorspeicher laden ---
    if st.session_state.vectorstore is None:
        with st.spinner("Daten werden geladen..."):
            documents = load_koerber_data()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
            text_chunks = [{"content": chunk, "url": doc["url"]} for doc in documents for chunk in text_splitter.split_text(doc["content"])]
            st.session_state.vectorstore = get_vector_store(text_chunks)
            st.session_state.documents = text_chunks

    # --- Benutzerabfrage ---
    query = st.text_input("Frag Körber")

    # --- Interaktive Buttons für "Mehr dazu" ---
    if "more_info" not in st.session_state:
        st.session_state.more_info = ""

    if st.session_state.more_info:
        query = f"Gib mir mehr dazu zu: {st.session_state.more_info}"

    if st.button("Antwort generieren") and query:
        with st.spinner("Antwort wird generiert..."):
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config)
            vectorstore = st.session_state.vectorstore
            relevant_content = vectorstore.similarity_search(query, k=5)

            context = "\n".join([doc.page_content for doc in relevant_content])
            result = get_response(context, query, model)

            st.success("Antwort:")
            st.write(result)

            # --- Interaktive Überschriften mit Button ---
            st.markdown("### 📌 Relevante Themen")
            for i, doc in enumerate(relevant_content[:3]):
                matching_doc = next((item for item in st.session_state.documents if item["content"] == doc.page_content), None)
                if matching_doc:
                    title = matching_doc["content"][:100]  # Erstes Stück Text als Vorschau
                    if st.button(f"Mehr zu: {title}", key=f"more_info_{i}"):
                        st.session_state.more_info = title
                        st.experimental_rerun()

            # --- Top 3 passende URLs anzeigen ---
            st.markdown("### 🔗 Quellen")
            shown_urls = set()  # Um doppelte Links zu vermeiden

            for doc in relevant_content[:3]:
                matching_doc = next((item for item in st.session_state.documents if item["content"] == doc.page_content), None)
                if matching_doc and matching_doc["url"] not in shown_urls:
                    shown_urls.add(matching_doc["url"])
                    st.markdown(f"[Zur Quelle]({matching_doc['url']})")

if __name__ == "__main__":
    main()
