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
    prompt = f"Extrahiere relevante Standortinformationen aus dieser Anfrage:\n\n{query}\n\nGib nur Städte oder Adressen zurück."
    try:
        response = model.generate_content(prompt)
        return re.findall(r'[A-Za-zäöüÄÖÜß\s,.-]+', response.text)
    except Exception as e:
        st.error(f"Fehler bei der Schlagwort-Extraktion: {e}")
        return []

# 📍 Extrahiere Standorte aus der Vektordatenbank
def extract_locations_from_vectorstore(vectorstore, keywords):
    combined_query = " ".join(keywords)
    relevant_docs = vectorstore.similarity_search(combined_query, k=5)

    # Sammelt mögliche Standortinformationen
    locations = []
    for doc in relevant_docs:
        content = getattr(doc, "page_content", getattr(doc, "content", ""))
        matches = re.findall(r'\d{5}\s[A-Za-zäöüÄÖÜß\s]+', content)  # Adresse (PLZ Stadt)
        locations.extend(matches)

    return list(set(locations))  # Doppelte entfernen

# 🗺️ Dynamische Folium-Karte mit mehreren Markern
def show_map_with_markers(locations):
    m = folium.Map(location=[53.5450, 10.0290], zoom_start=5)

    # Marker für alle Standorte setzen
    for location in locations:
        # Für dieses Beispiel statisch, später dynamisch mit Geo-Coding
        tooltip = location
        folium.Marker(
            location=[53.5450, 10.0290],  # Dummy-Koordinaten ersetzen
            popup=f"<b>{tooltip}</b>",
            tooltip=tooltip
        ).add_to(m)

    # Karte in Streamlit anzeigen
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

    if "vectorstore" not in st.session_state:
        with st.spinner("Daten werden geladen..."):
            documents = load_koerber_data()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=600)
            text_chunks = [{"content": chunk, "url": doc["url"]} for doc in documents for chunk in text_splitter.split_text(doc["content"])]
            st.session_state.vectorstore = get_vector_store(text_chunks)

    # 📌 Benutzeranfrage
    col1, col2 = st.columns([4, 1])
    with col1:
        query_input = st.text_input("Stellen Sie hier Ihre Frage:", value="")
    with col2:
        generate_button = st.button("Antwort generieren")

    # 🗺️ Immer eine Standardkarte anzeigen
    show_map_with_markers(["Anckelmannsplatz 1, 20537 Hamburg"])  # Standardstandort

    # 🔍 Verarbeitung der Benutzeranfrage
    if generate_button and query_input:
        with st.spinner("Antwort wird generiert..."):
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config)
            keywords = extract_keywords_with_llm(model, query_input)

            # Standorte aus Vektorspeicher extrahieren
            locations = extract_locations_from_vectorstore(st.session_state.vectorstore, keywords)

            if locations:
                st.success("📍 Standorte wurden auf der Karte angezeigt!")
                show_map_with_markers(locations)
            else:
                st.warning("⚠️ Keine Standorte gefunden.")

            st.success("📝 Antwort:")
            st.write(f"**Eingabe:** {query_input}")

if __name__ == "__main__":
    main()
