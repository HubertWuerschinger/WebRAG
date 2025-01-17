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

# 📌 API-Schlüssel laden
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
    return [{
        "content": doc["completion"],
        "url": doc["meta"].get("url", ""),
        "timestamp": doc["meta"].get("timestamp", ""),
        "title": doc["meta"].get("title", "Kein Titel")
    } for doc in dataset["train"]]

# 📦 Vektorspeicher erstellen
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        return FAISS.from_texts(texts=[chunk["content"] for chunk in text_chunks], embedding=embeddings)
    except Exception as e:
        st.error(f"Fehler beim Erstellen des Vektorspeichers: {e}")
        return None

# 🔍 Schlagwort-Extraktion mit Gemini
def extract_keywords_with_llm(model, query):
    prompt = f"Extrahiere relevante Schlagwörter aus dieser Anfrage:\n\n{query}\n\nNur Schlagwörter ohne Erklärungen."
    try:
        response = model.generate_content(prompt)
        return re.findall(r'\b\w{3,}\b', response.text)
    except Exception as e:
        st.error(f"Fehler bei der Schlagwort-Extraktion: {e}")
        return []

# 📊 Durchsuche Vektordatenbank nach der Benutzeranfrage
def search_vectorstore(vectorstore, query, k=5):
    relevant_content = vectorstore.similarity_search(query, k=k)
    return "\n".join([doc.page_content if hasattr(doc, "page_content") else doc.content for doc in relevant_content])

# 🗺️ Extrahiere Standortinformationen aus der Antwort
def extract_address_with_llm(model, text):
    prompt = f"Extrahiere aus diesem Text die Adresse im Format 'Straße, Stadt':\n\n{text}"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Fehler bei der Adress-Extraktion: {e}")
        return ""

# 🗺️ Zeige Standort auf der Karte an
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
        "max_output_tokens": 6000,
    }

    # Initialisierung des Session State
    if "vectorstore" not in st.session_state:
        with st.spinner("Daten werden geladen..."):
            documents = load_koerber_data()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=600)
            text_chunks = [{"content": chunk, "url": doc["url"]} for doc in documents for chunk in text_splitter.split_text(doc["content"])]
            st.session_state.vectorstore = get_vector_store(text_chunks)

    # 📌 Benutzeranfrage
    query_input = st.text_input("Stellen Sie hier Ihre Frage:", value="")
    generate_button = st.button("Antwort generieren")

    # 🔍 Verarbeite Benutzeranfrage
    if generate_button and query_input:
        with st.spinner("Antwort wird generiert..."):
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config)

            # 🔎 Vektorspeicher durchsuchen
            context = search_vectorstore(st.session_state.vectorstore, query_input)

            # 🏠 Standortinformationen extrahieren
            address = extract_address_with_llm(model, context)

            # 📍 Standortanzeige, falls Adresse erkannt
            if address:
                st.success(f"📍 Gefundene Adresse: {address}")
                # Adresse geokodieren (hier Beispielkoordinaten für Hamburg)
                if "Hamburg" in address:
                    show_map_with_marker([53.5450, 10.0290], tooltip=address)
                elif "Berlin" in address:
                    show_map_with_marker([52.5200, 13.4050], tooltip=address)
            else:
                st.info("📝 Keine Standortinformationen gefunden.")

            # 📝 Antwort ausgeben
            st.success("Antwort:")
            st.write(f"**Eingabe:** {query_input}")
            st.write(context)

if __name__ == "__main__":
    main()
