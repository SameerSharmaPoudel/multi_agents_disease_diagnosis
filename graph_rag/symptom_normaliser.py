"""
Lightweight symptom normaliser for graph RAG query-time lookup.

Solves the only real naming gap in this project:
    patient says "shortness of breath"  →  graph has "Dyspnea"
    patient says "high fever"           →  graph has "Fever"
    patient says "tired"                →  graph has "Fatigue"

No UMLS license, no REST API, no external database.
Four lookup tiers, in order:

    1. Exact match against graph symptom names (case-insensitive)
    2. Synonym table lookup (covers common patient-language variants)
    3. Fuzzy match against graph names (rapidfuzz, threshold 82)
       — catches typos and near-matches not in the synonym table
    4. Semantic Embedding match (sentence-transformers)
       — catches conceptual mappings (e.g., "ticker is pounding" -> "Palpitations")

Usage:
    from knowledge.symptom_normaliser import SymptomNormaliser

    # Initialise once (loads tsv, builds lookup tables, generates embeddings)
    norm = SymptomNormaliser("./data/graph_raw/symptoms-DO.tsv")

    # Single symptom
    result = norm.normalise("my ticker is pounding")
    # → NormResult(graph_name="Palpitations", mesh_id="D010102", score=0.87, tier="embedding")

    # Whole symptoms dict (as used by SymptomCollectorAgent)
    graph_symptoms = norm.normalise_dict({"high fever": "yes", "dry cough": "mild"})
    # → {"Fever": "yes", "Cough": "mild"}

Install:
    pip install rapidfuzz sentence-transformers torch
"""

import csv
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

log = logging.getLogger("SymptomNormaliser")

# ---------------------------------------------------------------------------
# Synonym table
# ---------------------------------------------------------------------------
SYNONYM_TABLE: dict[str, str] = {
    # Respiratory
    "shortness of breath":          "Dyspnea",
    "short of breath":              "Dyspnea",
    "breathlessness":               "Dyspnea",
    "difficulty breathing":         "Dyspnea",
    "breathing difficulty":         "Dyspnea",
    "can't breathe":                "Dyspnea",
    "trouble breathing":            "Dyspnea",
    "laboured breathing":           "Dyspnea",
    "labored breathing":            "Dyspnea",
    "dry cough":                    "Cough",
    "wet cough":                    "Cough",
    "productive cough":             "Cough",
    "persistent cough":             "Cough",
    "coughing":                     "Cough",
    "hacking cough":                "Cough",
    "wheezing":                     "Respiratory Sounds",
    "coughing blood":               "Hemoptysis",
    "blood in sputum":              "Hemoptysis",
    "spitting blood":               "Hemoptysis",

    # Fever / temperature
    "high fever":                   "Fever",
    "low fever":                    "Fever",
    "mild fever":                   "Fever",
    "slight fever":                 "Fever",
    "temperature":                  "Fever",
    "high temperature":             "Fever",
    "elevated temperature":         "Fever",
    "febrile":                      "Fever",
    "chills":                       "Chills",
    "shivering":                    "Chills",
    "rigors":                       "Chills",
    "night sweats":                 "Night Sweats",
    "sweating at night":            "Night Sweats",

    # Pain — general
    "pain":                         "Pain",
    "ache":                         "Pain",
    "aching":                       "Pain",
    "sore":                         "Pain",
    "soreness":                     "Pain",
    "discomfort":                   "Pain",
    "hurts":                        "Pain",
    "head pain":                    "Headache",
    "headaches":                    "Headache",
    "migraine":                     "Headache",
    "joint pain":                   "Arthralgia",
    "joint ache":                   "Arthralgia",
    "achy joints":                  "Arthralgia",
    "sore joints":                  "Arthralgia",
    "muscle pain":                  "Myalgia",
    "muscle ache":                  "Myalgia",
    "muscle aches":                 "Myalgia",
    "body aches":                   "Myalgia",
    "back pain":                    "Back Pain",
    "chest pain":                   "Chest Pain",
    "chest tightness":              "Chest Pain",
    "chest pressure":               "Chest Pain",
    "stomach pain":                 "Abdominal Pain",
    "belly pain":                   "Abdominal Pain",
    "tummy ache":                   "Abdominal Pain",
    "stomach ache":                 "Abdominal Pain",
    "stomach cramps":               "Abdominal Pain",
    "abdominal cramps":             "Abdominal Pain",
    "throat pain":                  "Pharyngitis",
    "sore throat":                  "Pharyngitis",
    "throat soreness":              "Pharyngitis",
    "ear pain":                     "Otalgia",
    "earache":                      "Otalgia",

    # Fatigue / energy
    "tired":                        "Fatigue",
    "tiredness":                    "Fatigue",
    "exhausted":                    "Fatigue",
    "exhaustion":                   "Fatigue",
    "weakness":                     "Muscle Weakness",
    "weak":                         "Muscle Weakness",
    "lack of energy":               "Fatigue",
    "low energy":                   "Fatigue",
    "lethargy":                     "Fatigue",
    "lethargic":                    "Fatigue",

    # GI tract
    "sick to stomach":              "Nausea",
    "feel sick":                    "Nausea",
    "queasy":                       "Nausea",
    "nauseous":                     "Nausea",
    "throwing up":                  "Vomiting",
    "vomit":                        "Vomiting",
    "puking":                       "Vomiting",
    "diarrhea":                     "Diarrhea",
    "diarrhoea":                    "Diarrhea",
    "loose stools":                 "Diarrhea",
    "runny stool":                  "Diarrhea",
    "constipated":                  "Constipation",
    "can't go to the bathroom":     "Constipation",
    "no bowel movement":            "Constipation",
    "bloated":                      "Flatulence",
    "bloating":                     "Flatulence",
    "gas":                          "Flatulence",
    "gassy":                        "Flatulence",
    "heartburn":                    "Pyrosis",
    "acid reflux":                  "Pyrosis",
    "indigestion":                  "Pyrosis",
    "reflux":                       "Pyrosis",
    "blood in stool":               "Melena",
    "rectal bleeding":              "Melena",

    # Skin
    "skin rash":                    "Exanthema",
    "rash":                         "Exanthema",
    "hives":                        "Urticaria",
    "itchy skin":                   "Pruritus",
    "itching":                      "Pruritus",
    "skin itching":                 "Pruritus",
    "skin itch":                    "Pruritus",
    "yellowing of skin":            "Jaundice",
    "yellow skin":                  "Jaundice",
    "yellowing eyes":               "Jaundice",
    "yellow eyes":                  "Jaundice",
    "bruising":                     "Ecchymosis",
    "easy bruising":                "Ecchymosis",
    "bruises easily":               "Ecchymosis",
    "hair loss":                    "Alopecia",
    "losing hair":                  "Alopecia",
    "thinning hair":                "Alopecia",

    # Neurological
    "dizzy":                        "Dizziness",
    "dizziness":                    "Dizziness",
    "light-headed":                 "Dizziness",
    "lightheaded":                  "Dizziness",
    "faint":                        "Syncope",
    "fainting":                     "Syncope",
    "passed out":                   "Syncope",
    "blackout":                     "Syncope",
    "loss of consciousness":        "Syncope",
    "confused":                     "Confusion",
    "confusion":                    "Confusion",
    "forgetful":                    "Memory Disorders",
    "memory loss":                  "Memory Disorders",
    "memory problems":              "Memory Disorders",
    "forgetting things":            "Memory Disorders",
    "tremor":                       "Tremor",
    "shaking":                      "Tremor",
    "hand tremor":                  "Tremor",
    "tingling":                     "Paresthesia",
    "numbness":                     "Paresthesia",
    "pins and needles":             "Paresthesia",
    "seizure":                      "Seizures",
    "fit":                          "Seizures",
    "convulsion":                   "Seizures",

    # Cardiovascular
    "heart pounding":               "Palpitations",
    "heart racing":                 "Palpitations",
    "heart fluttering":             "Palpitations",
    "palpitations":                 "Palpitations",
    "irregular heartbeat":          "Arrhythmias, Cardiac",
    "swollen legs":                 "Edema",
    "swollen ankles":               "Edema",
    "swelling":                     "Edema",
    "leg swelling":                 "Edema",

    # Eyes / ENT
    "blurry vision":                "Vision Disorders",
    "blurred vision":               "Vision Disorders",
    "vision problems":              "Vision Disorders",
    "can't see clearly":            "Vision Disorders",
    "runny nose":                   "Rhinitis",
    "stuffy nose":                  "Nasal Obstruction",
    "blocked nose":                 "Nasal Obstruction",
    "congestion":                   "Nasal Obstruction",
    "nasal congestion":             "Nasal Obstruction",
    "loss of smell":                "Olfaction Disorders",
    "can't smell":                  "Olfaction Disorders",
    "no sense of smell":            "Olfaction Disorders",
    "loss of taste":                "Ageusia",
    "can't taste":                  "Ageusia",
    "no taste":                     "Ageusia",

    # Urinary
    "frequent urination":           "Polyuria",
    "urinating frequently":         "Polyuria",
    "urinating a lot":              "Polyuria",
    "excessive thirst":             "Polydipsia",
    "very thirsty":                 "Polydipsia",
    "always thirsty":               "Polydipsia",
    "burning when urinating":       "Dysuria",
    "painful urination":            "Dysuria",
    "blood in urine":               "Hematuria",

    # Weight / appetite
    "losing weight":                "Weight Loss",
    "unexpected weight loss":       "Weight Loss",
    "unexplained weight loss":      "Weight Loss",
    "lost appetite":                "Anorexia",
    "no appetite":                  "Anorexia",
    "not hungry":                   "Anorexia",
    "loss of appetite":             "Anorexia",
    "weight gain":                  "Weight Gain",
    "gaining weight":               "Weight Gain",

    # Sleep
    "can't sleep":                  "Sleep Initiation and Maintenance Disorders",
    "insomnia":                     "Sleep Initiation and Maintenance Disorders",
    "trouble sleeping":             "Sleep Initiation and Maintenance Disorders",
    "sleep problems":               "Sleep Initiation and Maintenance Disorders",
    "snoring":                      "Snoring",

    # Mental health
    "feeling depressed":            "Depression",
    "sad":                          "Depression",
    "low mood":                     "Depression",
    "anxious":                      "Anxiety",
    "anxiety":                      "Anxiety",
    "panic":                        "Panic Disorder",
    "panic attack":                 "Panic Disorder",
}

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class NormResult:
    graph_name: str          # the name as it appears in symptoms-DO
    mesh_id:    Optional[str]  # MeSH D-number from symptoms-DO
    score:      float        # 1.0 = exact/synonym, 0.0-1.0 = similarity score
    tier:       str          # "exact" | "synonym" | "fuzzy" | "embedding" | "none"


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------

class SymptomNormaliser:
    """
    Maps patient-reported symptom strings to graph symptom names.
    Initialised once per process from the symptoms-DO.tsv file.
    All lookups are in-memory after init; no database calls at runtime.
    """

    FUZZY_THRESHOLD = 75       # out of 100; lower = more aggressive matching
    EMBEDDING_THRESHOLD = 0.55  # cosine similarity cut-off for Tier 4
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, symptoms_do_path: str = "./data/graph_raw/symptoms-DO.tsv"):
        self._name_to_mesh: dict[str, str] = {}   # lowercased graph name → mesh_id
        self._graph_names:  list[str] = []        # all lowercased graph names
        self._canonical:    dict[str, str] = {}   # lowercased → original-cased graph name
        
        self._fuzzy_available = self._check_rapidfuzz()
        self._load(Path(symptoms_do_path))
        
        # Tier 4 dependencies
        self._embed_model = None
        self._graph_embeddings = None
        self._embedding_available = self._init_embeddings()

    def _check_rapidfuzz(self) -> bool:
        try:
            import rapidfuzz  # noqa: F401
            return True
        except ImportError:
            log.debug("rapidfuzz not installed — tier 3 fuzzy matching disabled")
            return False

    def _init_embeddings(self) -> bool:
        if not self._graph_names:
            return False
        try:
            from sentence_transformers import SentenceTransformer
            log.info("Loading sentence-transformer model: %s...", self.MODEL_NAME)
            self._embed_model = SentenceTransformer(self.MODEL_NAME)
            
            # Pre-compute embeddings for your graph's canonical target vocabulary
            log.info("Computing embeddings for %d graph symptoms...", len(self._graph_names))
            # We encode the real display names to capture true semantic nuances
            display_names = [self._canonical[lc] for lc in self._graph_names]
            self._graph_embeddings = self._embed_model.encode(display_names, convert_to_tensor=True)
            return True
        except ImportError:
            log.warning("sentence-transformers or torch not installed — tier 4 embedding lookup disabled")
            return False
        except Exception as e:
            log.error("Failed to initialize embeddings tier: %s", e)
            return False

    def _load(self, path: Path):
        if not path.exists():
            log.warning("symptoms-DO.tsv not found at %s. Baseline rules empty.", path)
            return

        seen = set()
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                name = row["symptom_name"].strip()
                mesh = row["symptom_id"].strip()
                lc   = name.lower()
                if lc not in seen:
                    seen.add(lc)
                    self._name_to_mesh[lc] = mesh
                    self._canonical[lc]    = name
                    self._graph_names.append(lc)

        log.info("SymptomNormaliser loaded %d unique symptom names from %s", len(seen), path)

    # ── Public API ────────────────────────────────────────────────

    @lru_cache(maxsize=2048)
    def normalise(self, raw: str) -> NormResult:
        """
        Normalise a single symptom string across 4 cascading tiers.
        """
        lc = raw.strip().lower()

        # Tier 1 — exact match (case-insensitive)
        if lc in self._name_to_mesh:
            return NormResult(
                graph_name=self._canonical[lc],
                mesh_id=self._name_to_mesh[lc],
                score=1.0,
                tier="exact",
            )

        # Tier 2 — synonym table
        if lc in SYNONYM_TABLE:
            mapped = SYNONYM_TABLE[lc].lower()
            if mapped in self._name_to_mesh:
                return NormResult(
                    graph_name=self._canonical[mapped],
                    mesh_id=self._name_to_mesh[mapped],
                    score=1.0,
                    tier="synonym",
                )

        # Tier 3 — fuzzy match
        if self._fuzzy_available and self._graph_names:
            result = self._fuzzy_match(lc)
            if result:
                return result

        # Tier 4 — semantic embedding lookup
        if self._embedding_available:
            result = self._embedding_match(raw)
            if result:
                return result

        # No match
        return NormResult(graph_name=raw, mesh_id=None, score=0.0, tier="none")

    def normalise_dict(self, symptoms: dict) -> dict:
        out = {}
        for raw, val in symptoms.items():
            result = self.normalise(raw)
            out[result.graph_name] = val
            if result.tier == "none":
                log.debug("No normalisation for symptom: %r", raw)
        return out

    def coverage(self, symptoms: dict) -> dict:
        return {raw: self.normalise(raw) for raw in symptoms}

    # ── Internal ─────────────────────────────────────────────────
    def _fuzzy_match(self, lc: str) -> Optional[NormResult]:
            from rapidfuzz import process, fuzz

            # WRatio (Weighted Ratio) handles case modifications and extra trailing characters much better
            result = process.extractOne(
                lc,
                self._graph_names,
                scorer=fuzz.WRatio, 
                score_cutoff=self.FUZZY_THRESHOLD,
            )
            if result:
                matched_lc = result[0]
                score      = result[1] / 100.0
                return NormResult(
                    graph_name=self._canonical[matched_lc],
                    mesh_id=self._name_to_mesh[matched_lc],
                    score=score,
                    tier="fuzzy",
                )
            return None

    def _embedding_match(self, raw: str) -> Optional[NormResult]:
        import numpy as np

        # Ensure progress bar is silent to keep logs clean
        query_embedding = self._embed_model.encode(raw, convert_to_tensor=False, show_progress_bar=False)
        
        # Convert precomputed graph embeddings to a predictable numpy array if it isn't one already
        if hasattr(self._graph_embeddings, "cpu"):
            graph_arr = self._graph_embeddings.cpu().numpy()
        else:
            graph_arr = np.array(self._graph_embeddings)

        # Direct, bulletproof dot-product cosine similarity calculation
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        norm_graph = graph_arr / np.linalg.norm(graph_arr, axis=1, keepdims=True)
        cos_scores = np.dot(norm_graph, norm_query)

        best_idx = int(np.argmax(cos_scores))
        score = float(cos_scores[best_idx])

        if score >= self.EMBEDDING_THRESHOLD:
            matched_lc = self._graph_names[best_idx]
            return NormResult(
                graph_name=self._canonical[matched_lc],
                mesh_id=self._name_to_mesh[matched_lc],
                score=round(score, 4),
                tier="embedding",
            )
        return None


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    path = sys.argv[1] if len(sys.argv) > 1 else "../data/graph_rag/symptoms-DO.tsv"
    norm = SymptomNormaliser(path)

    test_inputs = [
        "high fever", "Fever", "feverrrr", # exact / fuzzy
        "shortness of breath", "breathlessness",
        "my ticker is pounding", "racing heartbeat", # embedding targets
        "belly cramps", "upset tummy",
        "losing sleep", "can't blink clear"
    ]

    print(f"\nSymptom Normaliser — 4-Tier Coverage Test ({len(test_inputs)} inputs)")
    print(f"{'Input':<30} {'Tier':<10} {'Score':>5}  {'Graph name'}")
    print("-" * 75)
    tiers = {"exact": 0, "synonym": 0, "fuzzy": 0, "embedding": 0, "none": 0}
    for raw in test_inputs:
        r = norm.normalise(raw)
        tiers[r.tier] += 1
        score_str = f"{r.score:.2f}" if r.score < 1.0 else "    "
        print(f"  {raw:<28} {r.tier:<10} {score_str}  {r.graph_name}")

    print()
    print(f"Tier breakdown: exact={tiers['exact']}  synonym={tiers['synonym']}"
          f"  fuzzy={tiers['fuzzy']}  embedding={tiers['embedding']}  none={tiers['none']}")
    print(f"Coverage: {100*(1 - tiers['none']/len(test_inputs)):.0f}%")