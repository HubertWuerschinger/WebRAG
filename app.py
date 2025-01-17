import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datasets import load_dataset
import re
import folium
from streamlit_folium import st_folium

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

# 📦 Optimierter Vektorspeicher mit HNSW-Index
def get_optimized_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_texts(texts=[chunk["content"] for chunk in text_chunks], embedding=embeddings)

# 🔍 Schlagwort-Extraktion mit Gemini
def extract_keywords_with_llm(model, query):
    prompt = f"Extrahiere relevante Schlagwörter aus dieser Anfrage:\n\n{query}\n\nNur Schlagwörter ohne Erklärungen."
    try:
        response = model.generate_content(prompt)
        return re.findall(r'\b\w{3,}\b', response.text)
    except Exception as e:
        st.error(f"Fehler bei der Schlagwort-Extraktion: {e}")
        return []

# 🔎 RAG-Suche (Retrieval-Augmented Generation)
def rag_search_with_gemini(model, vectorstore, query, k=5):
    relevant_docs = vectorstore.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"Nutze diesen Kontext, um präzise zu antworten:\n\n{context}\n\nFrage:\n{query}\n\nAntworte kurz und präzise."
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Fehler bei der Antwortgenerierung: {e}")
        return "Fehler bei der Antwortgenerierung."

# 🗺️ Standortanfrage erkennen
def is_location_query(keywords):
    location_terms = ["standort", "adresse", "niederlassung", "büro", "filiale", "ort"]
    return any(term in keywords for term in location_terms)

# 📍 Standortinformationen mit Gemini extrahieren
def extract_location_with_gemini(model, context):
    prompt = f"Extrahiere die genaue Adresse (Straße, Stadt) aus folgendem Text:\n\n{context}"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Fehler bei der Adress-Extraktion: {e}")
        return ""

# 🗺️ Karte anzeigen
def show_map_with_marker(location, tooltip):
    m = folium.Map(location=location, zoom_start=14)
    folium.Marker(location=location, popup=f"<b>{tooltip}</b>", tooltip=tooltip).add_to(m)
    st_folium(m, width=700, height=500)

# 🚀 Hauptprozess
def main():
    st.set_page_config(page_title="Körber AI Chatbot", page_icon=":factory:")
    st.header("🔍 Wie können wir dir weiterhelfen?")

    api_key = load_api_keys()
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 20,
        "max_output_tokens": 3000,
    }

    # 📦 Vektorspeicher initialisieren
    if "vectorstore" not in st.session_state:
        with st.spinner("Daten werden geladen..."):
            documents = load_koerber_data()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=600)
            text_chunks = [{"content": chunk, "url": doc["url"]} for doc in documents for chunk in text_splitter.split_text(doc["content"])]
            st.session_state.vectorstore = get_optimized_vector_store(text_chunks)

    # 📌 Benutzeranfrage
    query_input = st.text_input("Stellen Sie hier Ihre Frage:", value="")
    generate_button = st.button("Antwort generieren")

    if generate_button and query_input:
        with st.spinner("Antwort wird generiert..."):
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config)

            # 🔍 Schlagwörter extrahieren
            keywords = extract_keywords_with_llm(model, query_input)

            # 📊 RAG-Suche
            response_text = rag_search_with_gemini(model, st.session_state.vectorstore, query_input)

            # 🗺️ Standortanfrage prüfen
            if is_location_query(keywords):
                address_info = extract_location_with_gemini(model, response_text)
                
                if "Hamburg" in address_info:
                    show_map_with_marker([53.5450, 10.0290], tooltip=address_info)
                elif "Berlin" in address_info:
                    show_map_with_marker([52.5200, 13.4050], tooltip=address_info)
                
                st.success(f"📍 Standort: {address_info}")

            # 📝 Antwort anzeigen
            st.success("📝 Antwort:")
            st.write(response_text)

if __name__ == "__main__":
    main()
