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

# ⚠️ Streamlit-Konfiguration
st.set_page_config(page_title="Körber AI Chatbot", page_icon=":factory:")

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
    st.info("📂 Lade Körber-Daten...")
    try:
        dataset = load_dataset("json", data_files={"train": "koerber_data.jsonl"})
        st.success("✅ Körber-Daten erfolgreich geladen.")
        return [{
            "content": doc["completion"],
            "url": doc["meta"].get("url", ""),
            "timestamp": doc["meta"].get("timestamp", ""),
            "title": doc["meta"].get("title", "Kein Titel")
        } for doc in dataset["train"]]
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Daten: {e}")
        return []

# 📦 Vektorspeicher erstellen
def get_vector_store(text_chunks):
    st.info("📦 Erstelle Vektorspeicher...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vectorstore = FAISS.from_texts(texts=[chunk["content"] for chunk in text_chunks], embedding=embeddings)
        st.success("✅ Vektorspeicher erfolgreich erstellt.")
        return vectorstore
    except Exception as e:
        st.error(f"❌ Fehler beim Erstellen des Vektorspeichers: {e}")
        return None

# 🔍 Schlagwort-Extraktion mit Gemini
def extract_keywords_with_llm(model, query):
    st.info("🔍 Extrahiere Schlagwörter...")
    prompt = f"Extrahiere relevante Schlagwörter aus dieser Anfrage:\n\n{query}\n\nNur Schlagwörter ohne Erklärungen."
    try:
        response = model.generate_content(prompt)
        keywords = re.findall(r'\b\w{3,}\b', response.text)
        st.success(f"✅ Schlagwörter extrahiert: {', '.join(keywords)}")
        return keywords
    except Exception as e:
        st.error(f"❌ Fehler bei der Schlagwort-Extraktion: {e}")
        return []

# 📊 Vektorspeicher durchsuchen
def search_vectorstore(vectorstore, keywords, query, k=5):
    st.info("📊 Durchsuche Vektorspeicher...")
    combined_query = " ".join(keywords + [query])
    try:
        relevant_content = vectorstore.similarity_search(combined_query, k=k)
        context = "\n".join([getattr(doc, "page_content", getattr(doc, "content", "")) for doc in relevant_content])
        urls = [doc.metadata.get("url", "Keine URL gefunden") for doc in relevant_content if hasattr(doc, "metadata")]
        st.success("✅ Vektorspeicher erfolgreich durchsucht.")
        return context, urls[:3]
    except Exception as e:
        st.error(f"❌ Fehler bei der Vektorspeicher-Suche: {e}")
        return "", []

# 📍 Folium-Karte anzeigen (Standard oder spezifisch)
def show_map(location=None, tooltip="Körber AG"):
    st.info("📍 Erstelle Karte...")
    try:
        # Standardkarte, wenn kein Standort angegeben ist
        if location is None:
            m = folium.Map(location=[53.5511, 9.9937], zoom_start=5)  # Zentrum: Deutschland
        else:
            m = folium.Map(location=location, zoom_start=15)
            folium.Marker(location, popup=tooltip, tooltip=tooltip).add_to(m)
        
        st_folium(m, width=700, height=500)
        st.success("✅ Karte angezeigt.")
    except Exception as e:
        st.error(f"❌ Fehler bei der Kartenerstellung: {e}")

# 🚀 Hauptprozess
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

    if st.session_state.vectorstore is None:
        with st.spinner("Daten werden geladen..."):
            documents = load_koerber_data()
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=600)
                text_chunks = [{"content": chunk, "url": doc["url"]} for doc in documents for chunk in text_splitter.split_text(doc["content"])]
                st.session_state.vectorstore = get_vector_store(text_chunks)

    # Eingabe und Button nebeneinander
    col1, col2 = st.columns([4, 1])
    with col1:
        query_input = st.text_input("Stellen Sie hier Ihre Frage:", value="")
    with col2:
        generate_button = st.button("Antwort generieren")

    # 🌍 Standardkarte immer sichtbar
    show_map()

    if generate_button and query_input:
        with st.spinner("Antwort wird generiert..."):
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config)
            keywords = extract_keywords_with_llm(model, query_input)
            context, urls = search_vectorstore(st.session_state.vectorstore, keywords, query_input)

            st.success("📝 Antwort:")
            st.write(f"**Eingabe:** {query_input}")
            st.write(context)

            # 📍 Standortkarte aktualisieren, falls nach Standorten gefragt
            if any(keyword in ["standorte", "adresse", "büro", "niederlassung"] for keyword in keywords):
                show_map(location=[53.5450, 10.0290], tooltip="Körber AG Hamburg")

            st.markdown("### 🔗 **Relevante Links:**")
            for url in urls:
                if url:
                    st.markdown(f"- [Mehr erfahren]({url})")

if __name__ == "__main__":
    main()
