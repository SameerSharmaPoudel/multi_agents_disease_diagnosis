# knowledge/build_bridge.py
"""
UMLS Bridge — one-time build + runtime lookup
===============================================
Builds a SQLite concept_bridge.db from several offline sources,
then exposes a UMLSBridge class used at runtime by both:
  - GraphRetriever   (graph RAG)   — normalise symptom CUIs before Cypher query
  - VectorRetriever  (vector RAG)  — tag FAISS documents with CUI at index time
  - UMLSNormalizer                 — first lookup goes to this bridge, not the API

What the bridge stores
-----------------------
Every biomedical concept has many names:
    "rheumatoid arthritis"       ← DOID lowercase
    "Arthritis, Rheumatoid"      ← MeSH/NLM Title Case with inversion
    "RA"                         ← clinical abbreviation
    "M05.9"                      ← ICD-10 code
    C0003873                     ← UMLS CUI (the canonical key)
    DOID:7148                    ← Disease Ontology ID
    D001172                      ← MeSH descriptor ID

The bridge maps ALL of these to the same CUI so queries using
any form resolve to the same graph nodes and vector documents.

Data sources (build once, ship as a file)
-----------------------------------------
Tier 1 — UMLS MRCONSO.RRF   (requires free UMLS license)
    The gold standard: ~15M concept-string pairs across all sources.
    We extract only the SAB (source) columns we need:
        MSH  → MeSH
        SNOMEDCT_US → SNOMED-CT
        ICD10CM → ICD-10-CM
        NCI  → NCI Thesaurus
        HPO  → Human Phenotype Ontology
        DO   → Disease Ontology (DOID)

Tier 2 — Local TSV overrides  (from your downloaded files)
    symptoms-DO.tsv  → MeSH D-number ↔ DOID from the doid_code column
                        These are already bridged — import directly.
    human_disease_knowledge_full.tsv → protein_id ↔ DOID
                        Already keyed consistently on DOID — no bridging needed here.

Tier 3 — Abbreviation table  (hand-curated or generated)
    RA → rheumatoid arthritis (CUI C0003873)
    AMI → acute myocardial infarction (CUI C0027051)
    ... etc.

Tier 4 — Name normalisation fallback
    When no exact match is found, apply:
    1. lowercase + strip punctuation
    2. NLM inversion reversal: "Arthritis, Rheumatoid" → "rheumatoid arthritis"
    3. Fuzzy match against known preferred terms

Runtime lookup order (fast)
-----------------------------
Given input string or ID:
    1. Exact match in bridge (case-insensitive) → return CUI
    2. After NLM-inversion normalisation → retry exact match
    3. Fuzzy match (rapidfuzz, threshold 88) → best CUI
    4. UMLS REST API call (rate-limited, cached)
    5. Return None — caller falls back to raw string

Usage
------
Build (one-time, needs UMLS or local TSVs):
    python knowledge/build_bridge.py --from-tsv   # build from local files only
    python knowledge/build_bridge.py --from-umls  # also use MRCONSO.RRF

Runtime:
    from knowledge.build_bridge import UMLSBridge
    bridge = UMLSBridge()                       # loads SQLite into memory
    cui = bridge.to_cui("Arthritis, Rheumatoid")  # → C0003873
    cui = bridge.to_cui("DOID:7148")              # → C0003873
    cui = bridge.to_cui("RA")                     # → C0003873
    info = bridge.lookup(cui)                     # → full ConceptInfo object
    doid = bridge.to_doid("Arthritis, Rheumatoid")  # → DOID:7148
    mesh = bridge.to_mesh("rheumatoid arthritis")   # → D001172

Install:
    pip install rapidfuzz requests python-dotenv
"""

import os
import re
import sys
import csv
import json
import sqlite3
import logging
import argparse
import hashlib
from pathlib import Path
from typing import Optional, NamedTuple
from functools import lru_cache

from dotenv import load_dotenv

log = logging.getLogger("UMLSBridge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

load_dotenv(".env.backend")

BRIDGE_DB_PATH = Path(os.getenv("BRIDGE_DB_PATH", "./data/bridge/concept_bridge.db"))
UMLS_API_KEY   = os.getenv("UMLS_API_KEY", "")

UMLS_CACHE_PATH = Path("./data/bridge/umls_api_cache.json")

SCHEMA = """
CREATE TABLE IF NOT EXISTS concepts (
    cui     TEXT NOT NULL,
    name    TEXT NOT NULL,
    name_lc TEXT NOT NULL,
    sab     TEXT NOT NULL,
    tty     TEXT,
    PRIMARY KEY (cui, name_lc, sab)
);
CREATE INDEX IF NOT EXISTS idx_name_lc ON concepts(name_lc);
CREATE INDEX IF NOT EXISTS idx_cui     ON concepts(cui);

CREATE TABLE IF NOT EXISTS identifiers (
    cui     TEXT NOT NULL,
    id_type TEXT NOT NULL,
    id_val  TEXT NOT NULL,
    PRIMARY KEY (cui, id_type, id_val)
);
CREATE INDEX IF NOT EXISTS idx_id_val  ON identifiers(id_val);
CREATE INDEX IF NOT EXISTS idx_id_type ON identifiers(id_type);
"""


# ─────────────────────────────────────────────
# NLM inversion normaliser
# ─────────────────────────────────────────────

def normalise_nlm(term: str) -> str:
    """
    Convert NLM/MeSH inverted heading to natural order.

    "Arthritis, Rheumatoid"       → "rheumatoid arthritis"
    "Aneurysm, Abdominal Aortic"  → "abdominal aortic aneurysm"
    "Fever"                       → "fever"            (no change, no comma)

    Also strips extra whitespace and lowercases.
    """
    term = term.strip()
    if "," in term:
        parts = [p.strip() for p in term.split(",", 1)]
        term = f"{parts[1]} {parts[0]}"
    return term.lower().strip()


def normalise_name(name: str) -> str:
    """
    General normalisation for matching:
    - lowercase
    - collapse whitespace
    - remove trailing/leading punctuation
    - apply NLM inversion
    """
    name = normalise_nlm(name)
    name = re.sub(r"[''`]", "", name)          # smart quotes
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ─────────────────────────────────────────────
# Bridge builder
# ─────────────────────────────────────────────

class BridgeBuilder:
    """Builds concept_bridge.db from available sources."""

    def __init__(self, db_path: Path = BRIDGE_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()

    # ── Tier 2: local TSV files ──────────────────────────────────

    def load_symptoms_do(self, tsv_path: Path):
        """
        symptoms-DO.tsv already maps MeSH D-number ↔ DOID.

        Columns (with header):
            symptom_name  disease_name  cooccurs  tfidf_score
            disease_id    symptom_id    doid_code  doid_name

        We extract:
            disease_id (MeSH Dxxxxxx) ↔ doid_code (DOID:xxx)
            symptom_id (MeSH Dxxxxxx) with preferred symptom_name
            doid_name as an additional name string for the disease
        """
        if not tsv_path.exists():
            log.warning("symptoms-DO.tsv not found: %s", tsv_path)
            return

        log.info("Loading symptoms-DO.tsv bridge entries from %s", tsv_path)
        rows_added = 0

        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            seen_disease = {}
            seen_symptom = {}

            for row in reader:
                doid = row.get("doid_code", "").strip()
                disease_mesh = row.get("disease_id", "").strip()
                disease_name = row.get("disease_name", "").strip()
                doid_name    = row.get("doid_name", "").strip()
                symptom_mesh = row.get("symptom_id", "").strip()
                symptom_name = row.get("symptom_name", "").strip()

                # Diseases — use DOID as pseudo-CUI (we don't have real CUIs yet)
                # We'll reconcile with UMLS later; for now DOID is our canonical ID
                if doid and doid not in seen_disease:
                    seen_disease[doid] = True
                    pseudo_cui = f"DOID_CUI_{doid.replace(':', '_')}"

                    names = [disease_name, doid_name]
                    for nm in set(names):
                        if nm:
                            self._insert_concept(pseudo_cui, nm, "DO")
                            self._insert_concept(pseudo_cui, normalise_name(nm), "DO_NORM")

                    if disease_mesh:
                        self._insert_id(pseudo_cui, "DOID", doid)
                        self._insert_id(pseudo_cui, "MSH", disease_mesh)
                    rows_added += 1

                # Symptoms — use MeSH D-number as pseudo-CUI
                if symptom_mesh and symptom_mesh not in seen_symptom:
                    seen_symptom[symptom_mesh] = True
                    pseudo_cui = f"MSH_CUI_{symptom_mesh}"
                    if symptom_name:
                        self._insert_concept(pseudo_cui, symptom_name, "MSH")
                        self._insert_concept(pseudo_cui, normalise_name(symptom_name), "MSH_NORM")
                    self._insert_id(pseudo_cui, "MSH", symptom_mesh)

        self.conn.commit()
        log.info("  Added %d disease bridge entries from symptoms-DO.tsv", rows_added)

    def load_mrconso(self, mrconso_path: Path,
                     sab_filter=("MSH", "SNOMEDCT_US", "ICD10CM", "NCI", "HPO", "DO")):
        """
        Load from UMLS MRCONSO.RRF (pipe-delimited).

        MRCONSO columns (0-indexed):
            0  CUI   1 LAT   2 TS   3 LUI   4 STT   5 SUI
            6  ISPREF  7 AUI  8 SAUI  9 SCUI  10 SDUI
            11 SAB  12 TTY  13 CODE  14 STR  15 SRL  16 SUPPRESS  17 CVF

        We care about: CUI(0), SAB(11), TTY(12), CODE(13), STR(14), SUPPRESS(16)
        Filter: LAT=ENG, SUPPRESS=N, SAB in sab_filter
        """
        if not mrconso_path.exists():
            log.warning("MRCONSO.RRF not found: %s — skipping UMLS tier", mrconso_path)
            return

        log.info("Loading MRCONSO.RRF from %s (this takes a few minutes)...", mrconso_path)
        count = 0
        skipped = 0

        with open(mrconso_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 18:
                    continue

                lat      = parts[1]
                suppress = parts[16]
                sab      = parts[11]

                if lat != "ENG" or suppress != "N":
                    skipped += 1
                    continue
                if sab not in sab_filter:
                    continue

                cui  = parts[0]
                tty  = parts[12]
                code = parts[13]
                name = parts[14].strip()

                self._insert_concept(cui, name, sab, tty)
                self._insert_concept(cui, normalise_name(name), f"{sab}_NORM", tty)

                if code:
                    if sab == "MSH":
                        self._insert_id(cui, "MSH", code)
                    elif sab == "DO":
                        self._insert_id(cui, "DOID", f"DOID:{code}" if not code.startswith("DOID:") else code)
                    elif sab == "ICD10CM":
                        self._insert_id(cui, "ICD10CM", code)
                    elif sab == "SNOMEDCT_US":
                        self._insert_id(cui, "SNOMED", code)
                    elif sab == "HPO":
                        self._insert_id(cui, "HPO", code)
                    elif sab == "NCI":
                        self._insert_id(cui, "NCI", code)

                count += 1
                if count % 500_000 == 0:
                    self.conn.commit()
                    log.info("  Processed %d MRCONSO rows...", count)

        self.conn.commit()
        log.info("  Loaded %d concept-string pairs from MRCONSO.RRF", count)

    def load_abbreviations(self, abbrev_path: Optional[Path] = None):
        """
        Load common clinical abbreviations mapping to concept names.

        If no file is provided, a small built-in table covers the most common ones.
        Format (TSV, no header): abbreviation TAB full_name TAB optional_cui
        """
        BUILTIN_ABBREVS = [
            ("RA",   "rheumatoid arthritis",          "C0003873"),
            ("OA",   "osteoarthritis",                "C0029408"),
            ("AMI",  "acute myocardial infarction",   "C0027051"),
            ("MI",   "myocardial infarction",         "C0027051"),
            ("COPD", "chronic obstructive pulmonary disease", "C0024117"),
            ("DM",   "diabetes mellitus",             "C0011849"),
            ("T1DM", "type 1 diabetes mellitus",      "C0011854"),
            ("T2DM", "type 2 diabetes mellitus",      "C0011860"),
            ("HTN",  "hypertension",                  "C0020538"),
            ("IBD",  "inflammatory bowel disease",    "C0021390"),
            ("IBS",  "irritable bowel syndrome",      "C0022104"),
            ("MS",   "multiple sclerosis",            "C0026769"),
            ("SLE",  "systemic lupus erythematosus",  "C0024141"),
            ("ALS",  "amyotrophic lateral sclerosis", "C0002736"),
            ("ADHD", "attention deficit hyperactivity disorder", "C1263846"),
            ("AF",   "atrial fibrillation",           "C0004238"),
            ("CHF",  "congestive heart failure",      "C0018802"),
            ("CKD",  "chronic kidney disease",        "C1561643"),
            ("UTI",  "urinary tract infection",       "C0042029"),
            ("PE",   "pulmonary embolism",            "C0034065"),
            ("DVT",  "deep vein thrombosis",          "C0149871"),
            ("AMD",  "age-related macular degeneration", "C0242383"),
            ("PCOS", "polycystic ovary syndrome",     "C0032460"),
            ("OSA",  "obstructive sleep apnea",       "C0520679"),
            ("PD",   "Parkinson disease",             "C0030567"),
            ("AD",   "Alzheimer disease",             "C0002395"),
            ("NHL",  "non-Hodgkin lymphoma",          "C0024305"),
        ]

        entries = BUILTIN_ABBREVS
        if abbrev_path and abbrev_path.exists():
            with open(abbrev_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        abbrev = parts[0].strip()
                        name   = parts[1].strip()
                        cui    = parts[2].strip() if len(parts) > 2 else f"ABBREV_{abbrev}"
                        entries.append((abbrev, name, cui))

        log.info("Loading %d abbreviation entries", len(entries))
        for abbrev, name, cui in entries:
            self._insert_concept(cui, abbrev, "ABBREV")
            self._insert_concept(cui, name,   "ABBREV")
            self._insert_concept(cui, normalise_name(name), "ABBREV_NORM")

        self.conn.commit()

    def _insert_concept(self, cui: str, name: str, sab: str, tty: str = ""):
        if not name or not name.strip():
            return
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO concepts (cui, name, name_lc, sab, tty) VALUES (?,?,?,?,?)",
                (cui, name, name.lower(), sab, tty),
            )
        except Exception:
            pass

    def _insert_id(self, cui: str, id_type: str, id_val: str):
        if not id_val:
            return
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO identifiers (cui, id_type, id_val) VALUES (?,?,?)",
                (cui, id_type, id_val),
            )
        except Exception:
            pass


# ─────────────────────────────────────────────
# Runtime lookup
# ─────────────────────────────────────────────

class ConceptInfo(NamedTuple):
    cui:      str
    names:    list
    doid:     Optional[str]
    mesh_id:  Optional[str]
    icd10:    Optional[str]
    snomed:   Optional[str]
    hpo:      Optional[str]


class UMLSBridge:
    """
    Runtime bridge for concept normalisation.

    Used by:
        UMLSNormalizer   — first-tier lookup before calling REST API
        SymptomAnalyzerAgent — normalise symptom strings before graph query
        build_enhanced_index.py — tag vector documents with CUI at index time

    All lookups are cached in memory after first access.
    """

    def __init__(self, db_path: Path = BRIDGE_DB_PATH):
        if not db_path.exists():
            log.warning(
                "Bridge DB not found at %s. "
                "Run: python knowledge/build_bridge.py --from-tsv",
                db_path,
            )
            self._conn = None
            self._api_cache = {}
            return

        log.info("Loading UMLS bridge from %s", db_path)
        self._conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self._conn.row_factory = sqlite3.Row
        self._api_cache: dict = self._load_api_cache()

    def _load_api_cache(self) -> dict:
        if UMLS_CACHE_PATH.exists():
            try:
                return json.loads(UMLS_CACHE_PATH.read_text())
            except Exception:
                return {}
        return {}

    def _save_api_cache(self):
        UMLS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        UMLS_CACHE_PATH.write_text(json.dumps(self._api_cache, indent=2))

    # ── Public API ────────────────────────────────────────────────

    @lru_cache(maxsize=4096)
    def to_cui(self, term: str) -> Optional[str]:
        """
        Resolve any term or identifier to a UMLS CUI.

        Accepts:
            - free text:  "rheumatoid arthritis", "Arthritis, Rheumatoid", "RA"
            - DOID:       "DOID:7148"
            - MeSH ID:    "D001172"
            - ICD-10:     "M05.9"
            - CUI itself: "C0003873" (returned as-is after validation)
        """
        if not self._conn:
            return None

        # Already a CUI?
        if re.match(r"^C\d{7}$", term):
            return term

        # Identifier lookup (DOID, MeSH, ICD10, SNOMED, HPO)
        cui = self._lookup_by_id(term)
        if cui:
            return cui

        # Exact name match (case-insensitive)
        cui = self._lookup_by_name(term.lower())
        if cui:
            return cui

        # NLM inversion normalisation then retry
        normalised = normalise_name(term)
        if normalised != term.lower():
            cui = self._lookup_by_name(normalised)
            if cui:
                return cui

        # Fuzzy match (requires rapidfuzz)
        cui = self._fuzzy_lookup(normalised)
        if cui:
            return cui

        # UMLS REST API (last resort, rate-limited)
        if UMLS_API_KEY:
            cui = self._api_lookup(term)
            if cui:
                return cui

        return None

    def to_doid(self, term: str) -> Optional[str]:
        """Resolve any term to a DOID."""
        cui = self.to_cui(term)
        return self._get_id(cui, "DOID") if cui else None

    def to_mesh(self, term: str) -> Optional[str]:
        """Resolve any term to a MeSH descriptor ID."""
        cui = self.to_cui(term)
        return self._get_id(cui, "MSH") if cui else None

    def to_icd10(self, term: str) -> Optional[str]:
        """Resolve any term to an ICD-10-CM code."""
        cui = self.to_cui(term)
        return self._get_id(cui, "ICD10CM") if cui else None

    def lookup(self, cui: str) -> Optional[ConceptInfo]:
        """Return full ConceptInfo for a CUI."""
        if not self._conn or not cui:
            return None

        rows = self._conn.execute(
            "SELECT DISTINCT name FROM concepts WHERE cui = ? LIMIT 20", (cui,)
        ).fetchall()
        names = [r["name"] for r in rows]

        return ConceptInfo(
            cui=cui,
            names=names,
            doid=self._get_id(cui, "DOID"),
            mesh_id=self._get_id(cui, "MSH"),
            icd10=self._get_id(cui, "ICD10CM"),
            snomed=self._get_id(cui, "SNOMED"),
            hpo=self._get_id(cui, "HPO"),
        )

    def normalise_symptoms_dict(self, symptoms: dict) -> dict:
        """
        Given {symptom_name: value, ...}, return a new dict where
        each key is replaced with its UMLS CUI (if found),
        preserving values unchanged.

        Used by SymptomAnalyzerAgent before the graph query.
        """
        out = {}
        for name, val in symptoms.items():
            cui = self.to_cui(name)
            out[cui if cui else name] = val
        return out

    def tag_document(self, text: str, max_concepts: int = 8) -> list:
        """
        Scan text for known concept names and return their CUIs.

        Used by build_enhanced_index.py to tag vector documents
        with CUIs so graph and vector results can be merged by CUI.

        Returns a list of CUIs found in the text (up to max_concepts).
        """
        if not self._conn:
            return []

        text_lc = text.lower()
        rows = self._conn.execute(
            "SELECT DISTINCT cui, name_lc FROM concepts ORDER BY length(name_lc) DESC LIMIT 5000"
        ).fetchall()

        found = []
        for row in rows:
            if row["name_lc"] and row["name_lc"] in text_lc:
                if row["cui"] not in found:
                    found.append(row["cui"])
            if len(found) >= max_concepts:
                break

        return found

    # ── Internal helpers ─────────────────────────────────────────

    def _lookup_by_id(self, val: str) -> Optional[str]:
        """Look up a CUI by any identifier (DOID, MeSH, ICD10, SNOMED, HPO)."""
        if not self._conn:
            return None
        row = self._conn.execute(
            "SELECT cui FROM identifiers WHERE id_val = ? LIMIT 1", (val,)
        ).fetchone()
        return row["cui"] if row else None

    def _lookup_by_name(self, name_lc: str) -> Optional[str]:
        """Look up a CUI by lowercased name."""
        if not self._conn:
            return None
        row = self._conn.execute(
            "SELECT cui FROM concepts WHERE name_lc = ? LIMIT 1", (name_lc,)
        ).fetchone()
        return row["cui"] if row else None

    def _get_id(self, cui: str, id_type: str) -> Optional[str]:
        if not self._conn or not cui:
            return None
        row = self._conn.execute(
            "SELECT id_val FROM identifiers WHERE cui = ? AND id_type = ? LIMIT 1",
            (cui, id_type),
        ).fetchone()
        return row["id_val"] if row else None

    def _fuzzy_lookup(self, normalised: str) -> Optional[str]:
        """Fuzzy match against all known concept names."""
        try:
            from rapidfuzz import process, fuzz
        except ImportError:
            return None

        if not self._conn:
            return None

        rows = self._conn.execute(
            "SELECT DISTINCT name_lc, cui FROM concepts LIMIT 50000"
        ).fetchall()
        choices = {r["name_lc"]: r["cui"] for r in rows}

        result = process.extractOne(
            normalised,
            choices.keys(),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=88,
        )
        return choices[result[0]] if result else None

    def _api_lookup(self, term: str) -> Optional[str]:
        """Call UMLS REST API and cache result."""
        cache_key = hashlib.md5(term.encode()).hexdigest()
        if cache_key in self._api_cache:
            return self._api_cache[cache_key]

        try:
            import requests
            # Step 1: get TGT ticket
            r = requests.post(
                "https://utslogin.nlm.nih.gov/cas/v1/api-key",
                data={"apikey": UMLS_API_KEY},
                timeout=10,
            )
            tgt_url = r.headers.get("location", "")
            if not tgt_url:
                return None

            # Step 2: get service ticket
            st_resp = requests.post(
                tgt_url,
                data={"service": "http://umlsks.nlm.nih.gov"},
                timeout=10,
            )
            st = st_resp.text.strip()

            # Step 3: search
            search_resp = requests.get(
                "https://uts-ws.nlm.nih.gov/rest/search/current",
                params={"string": term, "ticket": st, "returnIdType": "concept"},
                timeout=10,
            )
            data = search_resp.json()
            results = data.get("result", {}).get("results", [])
            if results and results[0].get("ui") != "NONE":
                cui = results[0]["ui"]
                self._api_cache[cache_key] = cui
                self._save_api_cache()
                return cui

        except Exception as e:
            log.debug("UMLS API error for '%s': %s", term, e)

        self._api_cache[cache_key] = None
        return None


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build UMLS concept bridge database")
    parser.add_argument(
        "--from-tsv",
        action="store_true",
        help="Build from local TSV files only (no UMLS license required)",
    )
    parser.add_argument(
        "--from-umls",
        action="store_true",
        help="Also load MRCONSO.RRF (requires UMLS download)",
    )
    parser.add_argument(
        "--symptoms-do",
        default="./data/graph_raw/symptoms-DO.tsv",
        help="Path to symptoms-DO.tsv",
    )
    parser.add_argument(
        "--mrconso",
        default="./data/umls/MRCONSO.RRF",
        help="Path to UMLS MRCONSO.RRF",
    )
    parser.add_argument(
        "--abbrevs",
        default=None,
        help="Optional TSV of custom abbreviations (abbrev TAB full_name TAB cui)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="After building, run test lookups",
    )
    args = parser.parse_args()

    builder = BridgeBuilder(BRIDGE_DB_PATH)

    if args.from_tsv or args.from_umls:
        builder.load_abbreviations(Path(args.abbrevs) if args.abbrevs else None)
        builder.load_symptoms_do(Path(args.symptoms_do))

    if args.from_umls:
        builder.load_mrconso(Path(args.mrconso))

    builder.close()
    log.info("Bridge built: %s", BRIDGE_DB_PATH)

    if args.test:
        print("\n── Test lookups ──")
        bridge = UMLSBridge(BRIDGE_DB_PATH)
        test_cases = [
            ("Arthritis, Rheumatoid",    "MeSH NLM inversion"),
            ("rheumatoid arthritis",     "DOID lowercase"),
            ("RA",                       "abbreviation"),
            ("DOID:7148",               "DOID identifier"),
            ("D001172",                  "MeSH ID"),
            ("hypertension",             "common name"),
            ("Alzheimer's disease",      "possessive form"),
            ("polycystic ovary syndrome","multi-word"),
            ("PCOS",                     "abbreviation"),
        ]
        for term, desc in test_cases:
            cui  = bridge.to_cui(term)
            doid = bridge.to_doid(term) if cui else None
            mesh = bridge.to_mesh(term) if cui else None
            print(f"  [{desc}]")
            print(f"    input: {term!r}")
            print(f"    CUI:   {cui}   DOID: {doid}   MeSH: {mesh}")
