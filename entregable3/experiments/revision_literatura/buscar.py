import json, time, datetime, urllib.parse, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path

SALIDA = Path(__file__).with_name(f"busqueda_{datetime.date.today().isoformat()}.json")
UA = {"User-Agent": "revision-literatura-curso-ML/1.0 (mailto:arafaelr@uninorte.edu.co)"}

CADENAS = [
    {"id": "C1", "desc": "wine quality + random forest + ordinal",
     "openalex": '"wine quality" AND "random forest" AND ordinal',
     "s2": '"wine quality" + "random forest" + ordinal',
     "arxiv": 'all:"wine quality" AND all:"random forest" AND all:ordinal',
     "dblp": "wine quality random forest ordinal"},
    {"id": "C2", "desc": "wine quality + extremely randomized trees",
     "openalex": '"wine quality" AND ("extra trees" OR "extremely randomized trees" OR "extra-trees")',
     "s2": '"wine quality" + ("extra trees" | "extremely randomized trees")',
     "arxiv": 'all:"wine quality" AND (all:"extra trees" OR all:"extremely randomized")',
     "dblp": "wine quality extra trees"},
    {"id": "C3", "desc": "wine quality + ordinal information for a binary (dichotomized) target",
     "openalex": '"wine quality" AND ordinal AND (binary OR dichotomized OR dichotomization)',
     "s2": '"wine quality" + ordinal + (binary | dichotomized)',
     "arxiv": 'all:"wine quality" AND all:ordinal',
     "dblp": "wine quality ordinal"},
    {"id": "C4", "desc": "threshold-weighted CRPS / weighted cumulative impurity as split criterion",
     "openalex": '("threshold-weighted" AND "ranked probability") AND (tree OR forest)',
     "s2": '"threshold-weighted" + "ranked probability" + (tree | forest)',
     "arxiv": 'all:"threshold-weighted" AND (all:tree OR all:forest)',
     "dblp": "weighted ranked probability score tree"},
    {"id": "C5", "desc": "random forest on cumulative indicators of an ordinal outcome for a dichotomized binary target",
     "openalex": 'ordinal AND ("random forest" OR "extremely randomized trees") AND cumulative AND (dichotomized OR dichotomization OR "binary outcome")',
     "s2": 'ordinal + ("random forest" | "extremely randomized trees") + cumulative + (dichotomized | "binary outcome")',
     "arxiv": 'all:ordinal AND all:"random forest" AND all:cumulative',
     "dblp": "ordinal random forest cumulative"},
    {"id": "C6", "desc": "Spanish: bosque aleatorio + calidad del vino",
     "openalex": '"bosque aleatorio" AND ("calidad del vino" OR vino)',
     "s2": '"bosque aleatorio" + vino',
     "arxiv": 'all:"bosque aleatorio"',
     "dblp": "bosque aleatorio vino"},
    {"id": "C7", "desc": "ordinal forest exceedance probability / pooling adjacent thresholds",
     "openalex": 'ordinal AND forest AND ("exceedance probability" OR "adjacent thresholds" OR "neighbouring thresholds" OR "neighboring thresholds")',
     "s2": 'ordinal + forest + ("exceedance probability" | "adjacent thresholds")',
     "arxiv": 'all:ordinal AND all:forest AND all:exceedance',
     "dblp": "ordinal forest exceedance"},
    {"id": "C8", "desc": "fine-grained ordinal labels as auxiliary information for a coarse binary classifier",
     "openalex": '("ordinal labels" OR "ordinal outcome") AND ("auxiliary" OR "fine-grained") AND "binary classification" AND (tree OR forest)',
     "s2": '("ordinal labels" | "ordinal outcome") + (auxiliary | "fine-grained") + "binary classification"',
     "arxiv": 'all:ordinal AND all:auxiliary AND all:"binary classification"',
     "dblp": "ordinal labels auxiliary binary classification"},
]


def get(url, intentos=6, espera=6):
    for k in range(intentos):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return r.read()
        except Exception as e:
            if k == intentos - 1:
                print("   fallo:", str(e)[:80])
                return None
            time.sleep(espera * (k + 1))


def openalex(q):
    d = get("https://api.openalex.org/works?per_page=50&search=" + urllib.parse.quote(q))
    if not d:
        return None, []
    j = json.loads(d)
    items = []
    for w in j.get("results", []):
        ab = w.get("abstract_inverted_index") or {}
        pos = sorted((p, t) for t, ps in ab.items() for p in ps)
        items.append({"titulo": w.get("display_name"), "anio": w.get("publication_year"), "doi": w.get("doi"),
                      "fuente": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                      "citas": w.get("cited_by_count"), "resumen": " ".join(t for _, t in pos)[:700]})
    return j["meta"]["count"], items


def s2(q):
    d = get("https://api.semanticscholar.org/graph/v1/paper/search/bulk?fields=title,year,venue,externalIds,citationCount,abstract&query="
            + urllib.parse.quote(q), espera=12)
    if not d:
        return None, []
    j = json.loads(d)
    items = [{"titulo": p.get("title"), "anio": p.get("year"), "fuente": p.get("venue"),
              "doi": (p.get("externalIds") or {}).get("DOI"), "id": p.get("paperId"),
              "citas": p.get("citationCount"), "resumen": (p.get("abstract") or "")[:700]} for p in (j.get("data") or [])[:50]]
    return j.get("total"), items


def arxiv(q):
    d = get("http://export.arxiv.org/api/query?max_results=50&search_query=" + urllib.parse.quote(q))
    if not d:
        return None, []
    ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
    root = ET.fromstring(d)
    total = int(root.find("o:totalResults", ns).text)
    items = [{"titulo": " ".join(e.find("a:title", ns).text.split()), "anio": e.find("a:published", ns).text[:4],
              "id": e.find("a:id", ns).text, "resumen": " ".join(e.find("a:summary", ns).text.split())[:700]}
             for e in root.findall("a:entry", ns)]
    return total, items


def crossref(q):
    d = get("https://api.crossref.org/works?rows=50&select=title,DOI,issued,container-title,is-referenced-by-count&query.bibliographic="
            + urllib.parse.quote(q))
    if not d:
        return None, []
    j = json.loads(d)["message"]
    items = [{"titulo": (w.get("title") or [""])[0], "doi": w.get("DOI"),
              "anio": ((w.get("issued") or {}).get("date-parts") or [[None]])[0][0],
              "fuente": (w.get("container-title") or [""])[0], "citas": w.get("is-referenced-by-count")}
             for w in j.get("items", [])]
    return j.get("total-results"), items


BASES = {"OpenAlex": ("openalex", openalex), "Semantic Scholar": ("s2", s2), "arXiv": ("arxiv", arxiv), "Crossref": ("dblp", crossref)}
salida = {"fecha": datetime.datetime.now().isoformat(timespec="seconds"), "consultas": []}
for c in CADENAS:
    for base, (clave, fn) in BASES.items():
        total, items = fn(c[clave])
        salida["consultas"].append({"cadena": c["id"], "descripcion": c["desc"], "base": base, "consulta": c[clave],
                                    "total": total, "items": items})
        print(f"{c['id']} {base:17s} total={total}", flush=True)
        time.sleep(4 if base == "Semantic Scholar" else 1.5)
SALIDA.write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
