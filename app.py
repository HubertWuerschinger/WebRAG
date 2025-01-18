import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datasets import load_dataset
import re
import json
import datetime

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

# 🔍 Extrahiert Schlagwörter aus der Benutzeranfrage mit Gemini
def extract_keywords_with_llm(model, query):
    prompt = f"Extrahiere relevante Schlagwörter aus der folgenden Anfrage:\n\n{query}\n\nNur Schlagwörter ohne Erklärungen."
    try:
        response = model.generate_content(prompt)
        return re.findall(r'\b\w{3,}\b', response.text)
    except Exception as e:
        st.error(f"Fehler bei der Schlagwort-Extraktion: {e}")
        return []

# 🔎 RAG-Suche mit präzisem Kontext und URL-Integration
def search_vectorstore(vectorstore, keywords, query, k=5):
    combined_query = " ".join(keywords + [query])
    relevant_content = vectorstore.similarity_search(combined_query, k=k)
    context = "\n".join([doc.page_content if hasattr(doc, "page_content") else doc.content for doc in relevant_content])
    urls = [doc.metadata.get("url", "Keine URL gefunden") for doc in relevant_content if hasattr(doc, "metadata")]
    return context, urls[:3]

# 📝 Generiert strukturierte Antworten mit Gemini
def generate_response_with_gemini(vectorstore, query, model, k=5):
    keywords = extract_keywords_with_llm(model, query)
    context, urls = search_vectorstore(vectorstore, keywords, query, k)
    prompt_template = f"""
    Nutze den folgenden Kontext, um präzise und strukturierte Antworten zu liefern:

    Kontext: {context}
    Frage: {query}

    Füge relevante Links ein, wenn vorhanden.
    """
    try:
        response = model.generate_content(prompt_template)
        return response.text, urls
    except Exception as e:
        st.error(f"Fehler bei der Antwortgenerierung: {e}")
        return "Fehler bei der Antwortgenerierung.", []

# 💬 Feedback speichern
def save_feedback(query, response, feedback_type, comment):
    feedback_data = {
        "query": query,
        "response": response,
        "feedback": feedback_type,
        "comment": comment,
        "timestamp": datetime.datetime.now().isoformat()
    }
    if os.path.exists("user_feedback.json"):
        with open("user_feedback.json", "r", encoding="utf-8") as file:
            data = json.load(file)
    else:
        data = {"feedback": []}

    data["feedback"].append(feedback_data)

    with open("user_feedback.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

# 🚀 Hauptprozess zur Steuerung des Chatbots
def main():
    api_key = load_api_keys()
    genai.configure(api_key=api_key)
    st.set_page_config(page_title="Körber AI Chatbot", page_icon=":factory:")
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
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=600)
            text_chunks = [{"content": chunk, "url": doc["url"]} for doc in documents for chunk in text_splitter.split_text(doc["content"])]
            st.session_state.vectorstore = get_vector_store(text_chunks)

    query_input = st.text_input("Stellen Sie hier Ihre Frage:", value="")
    generate_button = st.button("Antwort generieren")

    if generate_button and query_input:
        with st.spinner("Antwort wird generiert..."):
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config)
            result, urls = generate_response_with_gemini(st.session_state.vectorstore, query_input, model)

            st.success("📝 Antwort:")
            st.write(result)

            st.markdown("### 🔗 **Relevante Links:**")
            for url in urls:
                if url:
                    st.markdown(f"- [Mehr erfahren]({url})")

            st.markdown("### 💬 **Feedback zur Antwort:**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Antwort war hilfreich"):
                    save_feedback(query_input, result, "👍", "")
                    st.success("Danke für dein Feedback!")
            with col2:
                if st.button("👎 Antwort verbessern"):
                    save_feedback(query_input, result, "👎", "Bitte verbessern")
                    st.warning("Danke für dein Feedback!")

if __name__ == "__main__":
    main()
