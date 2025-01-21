import json
from datetime import datetime

def convert_to_rag_format(crawled_jobs, output_file="rag_data.jsonl"):
    """
    Konvertiert gecrawlte Jobs in das JSONL-Format für ein RAG-System.

    Args:
        crawled_jobs (list): Liste der gecrawlten Stellenangebote.
        output_file (str): Dateiname für die JSONL-Ausgabe.
    """
    rag_data = []

    for job in crawled_jobs:
        # Titel und URL aus dem Job übernehmen
        title = job.get("title", "Unbekannter Titel")
        url = job.get("url", "Keine URL")
        description = job.get("description", "Keine Beschreibung")
        
        # Eintrag für RAG-System erstellen
        entry = {
            "prompt": f"Was enthält die Seite '{title}'?",
            "completion": description,
            "meta": {
                "title": title,
                "url": url,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        rag_data.append(entry)

    # Daten in JSONL-Datei speichern
    with open(output_file, "w", encoding="utf-8") as file:
        for entry in rag_data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Daten erfolgreich in {output_file} gespeichert.")


# Beispiel: Verwendung der Funktion mit gecrawlten Daten
if __name__ == "__main__":
    # Beispiel für gecrawlte Daten
    crawled_jobs = [
        {
            "title": "Cloud Engineer (AWS) Mid",
            "company": "Körber Porto, Unipessoal Lda.",
            "location": "Porto, PT",
            "url": "https://jobs.koerber.com/pharma/job/Porto-Cloud-Engineer-%28AWS%29-Mid/1113919701/",
            "description": "Your role in our team:\n- Be part of a diverse team...\n\nYour profile:\n- BS or MS in Computer Science..."
        },
        {
            "title": "Senior Cloud Engineer (Azure)",
            "company": "Körber Porto, Unipessoal Lda.",
            "location": "Porto, PT",
            "url": "https://jobs.koerber.com/pharma/job/Porto-Senior-Cloud-Engineer-%28Azure%29/1113883101/",
            "description": "Your role in our team:\n- Lead cloud-based solutions...\n\nYour profile:\n- 5+ years of experience in Azure..."
        }
    ]

    # Konvertierung aufrufen
    convert_to_rag_format(crawled_jobs, output_file="rag_data.jsonl")
