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

# ⚠️ Setze die Konfiguration direkt nach den Imports
st.set_page_config(page_title="Körber AI Chatbot", page_icon=":factory:")

# 🔑 Lädt den API-Schlüssel aus der .env-Datei
def load_api_keys():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("API-Schlüssel fehlt. Bitte die .env-Datei prüfen.")
        st.stop()
    return api_key

# 📂 Lädt die JSONL-Daten für den Vektorspeicher
def load_koerber_data():
    dataset = load_dataset("json", data_files={"train": "koerber_data.jsonl"})
    return [{
        "content": doc["completion"],
        "url": doc["meta"].get("url", ""),
        "timestamp": doc["meta"].get("timestamp", ""),
        "title": doc["meta"].get("title", "Kein Titel")
    } for doc in dataset["train"]]

# 📦 Erzeugt den Vektorspeicher mit FAISS
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        return FAISS.from_texts(texts=[chunk["content"] for chunk in text_chunks], embedding=embeddings)
    except Exception as e:
        st.error(f"Fehler beim Erstellen des Vektorspeichers: {e}")
        return None

# 🔍 Schlagwörter mit Gemini extrahieren
def extract_keywords_with_llm(model, query):
    prompt = f"Extrahiere relevante Schlagwörter aus dieser Anfrage:\n\n{query}\n\nNur Schlagwörter ohne Erklärungen."
    try:
        response = model.generate_content(prompt)
        return re.findall(r'\b\w{3,}\b', response.text)
    except Exception as e:
        st.error(f"Fehler bei der Schlagwort-Extraktion: {e}")
        return []

# 📊 Vektorspeicher durchsuchen
def search_vectorstore(vectorstore, keywords, query, k=5):
    combined_query = " ".join(keywords + [query])
    relevant_content = vectorstore.similarity_search(combined_query, k=k)
    context = "\n".join([getattr(doc, "page_content", getattr(doc, "content", "")) for doc in relevant_content])
    urls = [doc.metadata.get("url", "Keine URL gefunden") for doc in relevant_content if hasattr(doc, "metadata")]
    return context, urls[:3]

# 📍 Folium-Karte anzeigen
def show_google_map_folium(location, tooltip="Körber AG"):
    # Erstelle eine Folium-Karte mit Marker
    m = folium.Map(location=location, zoom_start=15)
    folium.Marker(location, popup=tooltip, tooltip=tooltip).add_to(m)
    st_folium(m, width=700, height=500)

# 🚀 Hauptfunktion
def main():
    api_key = load_api_keys()
    genai.configure(api_key=api_key)
    st.header("🔍 Wie können wir dir weiterhelfen?")

    generation_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 20,
        "max_output_tokens": 6000,
    }

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "query" not in st.session_state:
        st.session_state.query = ""

    if st.session_state.vectorstore is None:
        with st.spinner("Daten werden geladen..."):
            documents = load_koerber_data()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=600)
            text_chunks = [{"content": chunk, "url": doc["url"]} for doc in documents for chunk in text_splitter.split_text(doc["content"])]
            st.session_state.vectorstore = get_vector_store(text_chunks)
            st.session_state.documents = text_chunks

    col1, col2 = st.columns([4, 1])

    with col1:
        query_input = st.text_input("Stellen Sie hier Ihre Frage:", value="")

    with col2:
        generate_button = st.button("Antwort generieren")

    if generate_button and query_input:
        st.session_state.query = query_input
        with st.spinner("Antwort wird generiert..."):
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config)
            keywords = extract_keywords_with_llm(model, st.session_state.query)
            context, urls = search_vectorstore(st.session_state.vectorstore, keywords, st.session_state.query)

            st.success("Antwort:")
            st.write(f"**Eingabe:** {st.session_state.query}")
            st.write(context)

            # 📍 Zeige Google Maps, wenn nach Standorten gefragt wird
            if any(keyword in ["standorte", "adresse", "büro", "niederlassung"] for keyword in keywords):
                st.markdown("### 📍 **Standort auf Google Maps:**")
                # Standort für Körber AG: Anckelmannsplatz 1, 20537 Hamburg
                show_google_map_folium([53.5450, 10.0290], tooltip="Körber AG Hamburg")

            st.markdown("### 🔗 **Relevante Links:**")
            for url in urls:
                if url:
                    st.markdown(f"- [Mehr erfahren]({url})")

            st.session_state.query = ""

    st.markdown("### 💡 Beispielanfragen:")
    st.markdown("- Wie viele Mitarbeiter hat Körber?")
    st.markdown("- Welche Produkte bietet Körber im Bereich Logistik an?")
    st.markdown("- Wo befinden sich die Standorte von Körber?")

if __name__ == "__main__":
    main()
