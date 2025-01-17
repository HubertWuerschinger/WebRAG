import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datasets import load_dataset
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

# 🔍 Standortsuche mit Gemini
def search_location_with_gemini(model, query, vectorstore):
    prompt = f"""
    Suche in den folgenden Daten nach Standortinformationen für die Anfrage: {query}.
    Gib nur Adressen im Format: [Adresse, Stadt, PLZ] zurück.
    """
    relevant_content = vectorstore.similarity_search(query, k=5)
    context = "\n".join([doc.page_content if hasattr(doc, "page_content") else doc.content for doc in relevant_content])

    try:
        search_prompt = f"{prompt}\n\n{context}"
        response = model.generate_content(search_prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Fehler bei der Standortsuche: {e}")
        return ""

# 🗺️ Dynamische Folium-Karte mit Gemini-Daten
def show_dynamic_map_with_gemini(location_data):
    m = folium.Map(location=[53.55, 10.00], zoom_start=6)

    if location_data:
        addresses = location_data.split("\n")
        for address in addresses:
            folium.Marker(
                location=[53.55, 10.00],  # Dummy-Koordinaten (könnten durch Geocoding ersetzt werden)
                popup=f"<b>{address}</b>",
                tooltip=address
            ).add_to(m)

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

    # 🛠️ Initialisierung des Session State
    if "vectorstore" not in st.session_state:
        with st.spinner("Daten werden geladen..."):
            documents = load_koerber_data()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=600)
            text_chunks = [{"content": chunk, "url": doc["url"]} for doc in documents for chunk in text_splitter.split_text(doc["content"])]
            st.session_state.vectorstore = get_vector_store(text_chunks)

    if "location_data" not in st.session_state:
        st.session_state.location_data = ""

    # 📌 Benutzeranfrage
    col1, col2 = st.columns([4, 1])
    with col1:
        query_input = st.text_input("Stellen Sie hier Ihre Frage:", value="")
    with col2:
        generate_button = st.button("Antwort generieren")

    # 🗺️ Karte anzeigen, wenn Daten vorhanden sind
    if st.session_state.location_data:
        show_dynamic_map_with_gemini(st.session_state.location_data)
    else:
        # Standardkarte ohne Marker
        show_dynamic_map_with_gemini("")

    # 🔍 Anfrage bearbeiten
    if generate_button and query_input:
        with st.spinner("Antwort wird generiert..."):
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config)
            location_info = search_location_with_gemini(model, query_input, st.session_state.vectorstore)

            if location_info:
                st.session_state.location_data = location_info
                st.success("📍 Standort gefunden und Karte aktualisiert!")
            else:
                st.warning("⚠️ Kein Standort gefunden. Bitte präzisiere deine Anfrage.")

if __name__ == "__main__":
    main()
