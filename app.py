import os
import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import streamlit as st

# API-Schlüssel sicher aus Streamlit Secrets laden
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# --- Web Scraping der Körber-Website ---
def scrape_koerber_website(url="https://www.koerber.com/"):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        text_content = " ".join([p.get_text() for p in soup.find_all("p")])
        return text_content
    except Exception as e:
        st.warning(f"Fehler beim Abrufen der Website: {e}")
        return ""

# --- Vektorspeicher erstellen ---
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except:
        st.warning("Fehler beim Erstellen des Vektorspeichers.")
        return None

# --- Kontextbezogene Antwort generieren ---
def get_response(context, question, model):
    prompt_template = f"""
    Du bist ein hilfreicher Assistent, der Fragen basierend auf dem folgenden Kontext beantwortet:

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
    st.set_page_config(page_title="Körber AI Assistant", page_icon=":robot_face:")
    st.header("🔍 Frag die Körber-Website")

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    generation_config = {
        "temperature": 0.2,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 8000,
    }

    if "vectorstore" not in st.session_state:
        with st.spinner("Daten von der Körber-Website werden geladen..."):
            website_text = scrape_koerber_website()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
            text_chunks = text_splitter.split_text(website_text)
            st.session_state.vectorstore = get_vector_store(text_chunks)

    # --- Benutzerabfrage ---
    query = st.text_input("Stelle eine Frage zur Körber-Website")

    if st.button("Antwort generieren") and query:
        with st.spinner("Antwort wird generiert..."):
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config)
            vectorstore = st.session_state.vectorstore
            relevant_content = vectorstore.similarity_search(query, k=5)
            context = "\n".join([doc.page_content for doc in relevant_content])
            result = get_response(context, query, model)
            st.success("Antwort:")
            st.write(result)

if __name__ == "__main__":
    main()
