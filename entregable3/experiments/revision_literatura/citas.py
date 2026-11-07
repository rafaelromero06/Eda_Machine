import json, time, datetime, urllib.parse, urllib.request
from pathlib import Path

UA = {"User-Agent": "revision-literatura-curso-ML/1.0 (mailto:arafaelr@uninorte.edu.co)"}
SEMILLAS = {
    "Tutz2022": "DOI:10.1007/s00357-021-09406-4",
    "LechnerOkasa": "DOI:10.1007/s00181-024-02646-4",
    "Hornung2020": "DOI:10.1007/s00357-018-9302-x",
    "Buczak2024": "DOI:10.1007/s00357-024-09497-9",
    "Janitza2016": None,
    "ClassifierPooling2026": "ARXIV:2603.17278",
}


def get(url, intentos=6):
    for k in range(intentos):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            if k == intentos - 1:
                print("  fallo", url[:90], str(e)[:60])
                return None
            time.sleep(10 * (k + 1))


busq = get("https://api.semanticscholar.org/graph/v1/paper/search?limit=3&fields=title,externalIds,year&query="
           + urllib.parse.quote("Random forest for ordinal responses: Prediction and variable selection"))
if busq and busq.get("data"):
    SEMILLAS["Janitza2016"] = busq["data"][0]["paperId"]
    print("Janitza ->", busq["data"][0]["title"], busq["data"][0]["year"])
time.sleep(7)

salida = {"fecha": datetime.datetime.now().isoformat(timespec="seconds"), "semillas": {}}
for nombre, pid in SEMILLAS.items():
    if not pid:
        continue
    base = f"https://api.semanticscholar.org/graph/v1/paper/{pid}"
    info = get(base + "?fields=title,year,abstract,venue")
    time.sleep(7)
    refs = get(base + "/references?limit=200&fields=title,year,venue")
    time.sleep(7)
    cits = get(base + "/citations?limit=200&fields=title,year,venue")
    time.sleep(7)
    salida["semillas"][nombre] = {
        "id": pid, "info": info,
        "referencias": [r["citedPaper"] for r in ((refs or {}).get("data") or []) if r.get("citedPaper")],
        "citas": [c["citingPaper"] for c in ((cits or {}).get("data") or []) if c.get("citingPaper")],
    }
    print(nombre, (info or {}).get("title"), "| refs", len(salida["semillas"][nombre]["referencias"]),
          "| citas", len(salida["semillas"][nombre]["citas"]), flush=True)
Path(__file__).with_name(f"citas_{datetime.date.today().isoformat()}.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
