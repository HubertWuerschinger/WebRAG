import requests
from bs4 import BeautifulSoup
import json
import time
import random

# Webcrawler-Einstellungen
BASE_URL = "https://jobs.koerber.com/search/locale=de_DE?_gl=1*bgr1td*_gcl_au*MTg4NDQ3Mzc0Ny4xNzM2OTYzNTY0"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def fetch_job_details(job_url):
    retries = 5
    session = requests.Session()  # Verbindung stabilisieren
    for attempt in range(retries):
        try:
            response = session.get(job_url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # HTML in Zeilen aufteilen
            html_lines = response.text.splitlines()
            valid_keywords = {
                "Your role in our team", 
                "Ihre Rolle in unserem Team",
                "Your profile",  
                "Ihr Profil"
            }
            sections = {}
            current_section = None

            for index, line in enumerate(html_lines):
                if any(keyword in line for keyword in valid_keywords):
                    current_section = next((kw for kw in valid_keywords if kw in line), None)
                    sections[current_section] = []

                    for offset in range(1, 21):
                        next_index = index + offset
                        if next_index < len(html_lines):
                            next_line = html_lines[next_index]
                            if "<li>" in next_line:
                                clean_text = BeautifulSoup(next_line, "html.parser").get_text(strip=True)
                                sections[current_section].append(clean_text)

            if sections:
                formatted_description = "\n\n".join(
                    f"{title}:\n" + "\n".join(content) for title, content in sections.items()
                )
                return {"description": formatted_description}
            else:
                return {"description": "Keine relevanten Abschnitte gefunden."}

        except requests.exceptions.RequestException as e:
            print(f"Fehler bei Anfrage {job_url} (Versuch {attempt + 1} von {retries}): {e}")
            time.sleep(random.randint(1, 10))

    return {"description": "Details konnten nicht abgerufen werden."}


def crawl_koerber_jobs(max_pages=5, output_file="koerber_data.json"):
    failed_urls = []
    page = 0

    while page < max_pages:
        url = f"{BASE_URL}&start={page * 10}"
        print(f"Rufe Seite {page + 1} ab: {url}")
        retries = 3
        while retries > 0:
            try:
                response = requests.get(url, headers=HEADERS, timeout=10)
                response.raise_for_status()
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                print(f"Verbindungsfehler auf Seite {page + 1}: {e}. Erneuter Versuch...")
                retries -= 1
                time.sleep(random.randint(1, 10))
            except requests.exceptions.HTTPError as e:
                print(f"HTTP-Fehler auf Seite {page + 1}: {e}")
                return
        else:
            print(f"Fehler: Seite {page + 1} konnte nicht geladen werden.")
            return

        soup = BeautifulSoup(response.text, "html.parser")
        job_tiles = soup.find_all("li", class_="job-tile")
        print(f"{len(job_tiles)} Jobs auf Seite {page + 1} gefunden.")

        for job in job_tiles:
            try:
                title = job.find("a", class_="jobTitle-link").text.strip()
                company = job.find("div", id=lambda x: x and "customfield2-value" in x).text.strip()
                location = job.find("div", id=lambda x: x and "multilocation-value" in x).text.strip()

                relative_url = job.get("data-url")
                full_url = f"https://jobs.koerber.com{relative_url}"
                job_details = fetch_job_details(full_url)

                job_data = {
                    "title": title,
                    "company": company,
                    "location": location,
                    "url": full_url,
                    **job_details
                }

                # Speichere den Job direkt in die Datei
                save_job_to_file(job_data, output_file)

                time.sleep(random.randint(1, 10))
            except Exception as e:
                print(f"Fehler bei der Verarbeitung eines Jobs: {e}")
                failed_urls.append(full_url)

        page += 1

    if failed_urls:
        print(f"Fehlerhafte URLs ({len(failed_urls)}):")
        for url in failed_urls:
            print(url)


def save_job_to_file(job, filename):
    try:
        with open(filename, "a", encoding="utf-8") as file:
            file.write(json.dumps(job, ensure_ascii=False, indent=4) + ",\n")
        print(f"Job gespeichert: {job['title']}")
    except Exception as e:
        print(f"Fehler beim Speichern des Jobs: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Körber Job Webcrawler")
    parser.add_argument("--max_pages", type=int, default=5, help="Maximale Anzahl der zu crawleden Seiten")
    parser.add_argument("--output_file", type=str, default="koerber_data.json", help="Dateiname für die gespeicherten Ergebnisse")

    args = parser.parse_args()
    crawl_koerber_jobs(max_pages=args.max_pages, output_file=args.output_file)
