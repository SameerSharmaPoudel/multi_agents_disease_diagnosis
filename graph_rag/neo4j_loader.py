# graph_rag/neo4j_loader.py
"""
Phase 1 — Graph RAG: Neo4j Loader (v2, matches actual downloaded files)
=========================================================================

This version is adapted to the ACTUAL file formats:

1. symptoms-DO.tsv
   ------------------
   Tab-separated, WITH header:
       symptom_name  disease_name  cooccurs  tfidf_score  disease_id  symptom_id  doid_code  doid_name

   - symptom_id / disease_id are MeSH descriptor IDs (D-numbers)
   - doid_code is ALREADY bridged to Disease Ontology — no fuzzy
     name-matching needed, Disease nodes are keyed directly on DOID.
   - tfidf_score is the edge weight (cooccurs is the raw, unnormalized count)

2. human_disease_knowledge_full.tsv / human_disease_textmining_full.tsv
   ----------------------------------------------------------------------
   Tab-separated, NO header, 7 columns:
       protein_id  gene_name  doid  disease_name  col5  col6  col7

   For the "knowledge" channel:
       col5 = source database (UniProtKB-KW, MedlinePlus, AmyCo, OMIM, ...)
       col6 = evidence type (usually "CURATED")
       col7 = confidence score, integer 1-5 ("stars")

   For the "textmining" channel:
       col5 = z-score (float)
       col6 = confidence score, float 0-5  <- use this for filtering
       col7 = URL to supporting evidence

What this script builds
------------------------
Node types:
    (:Symptom {mesh_id, name})
    (:Disease {doid, name, mesh_id})   -- mesh_id set only if seen in symptoms-DO
    (:Gene    {protein_id, name})

Edge types:
    (:Symptom)-[:PRESENTS_IN {cooccurs, tfidf_score, source:'symptoms-do'}]->(:Disease)
    (:Gene)-[:IMPLICATED_IN  {score, channel, source_db}]->(:Disease)

Because both datasets key Disease on DOID directly, no heuristic
name-matching bridge is required for diseases that appear in both.

Usage
-----
    python graph_rag/neo4j_loader.py \
        --symptoms  ./data/graph_raw/symptoms-DO.tsv \
        --diseases  ./data/graph_raw/diseases/human_disease_knowledge_full.tsv \
        --diseases  ./data/graph_raw/diseases/human_disease_textmining_full.tsv

    python graph_rag/neo4j_loader.py --verify     # verify only
    python graph_rag/neo4j_loader.py --wipe ...   # wipe and reload

Environment variables (.env.backend):
    NEO4J_URI       bolt://localhost:7687
    NEO4J_USER      neo4j
    NEO4J_PASSWORD  your_password

Install:
    pip install neo4j tqdm python-dotenv
"""

import os
import csv
import sys
import logging
import argparse
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from tqdm import tqdm

log = logging.getLogger("Neo4jLoader")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

load_dotenv("../.env.backend")


# ---------------------------------------------------------------------------
# Neo4j connection
# ---------------------------------------------------------------------------

def get_driver():
    """Create a Neo4j driver from environment variables."""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        log.error("neo4j package not installed. Run: pip install neo4j")
        sys.exit(1)

    uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user     = os.getenv("NEO4J_USER",     "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    log.info("Connecting to Neo4j at %s as %s", uri, user)
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        log.info("Neo4j connection OK")
        return driver
    except Exception as e:
        log.error(
            "Cannot connect to Neo4j: %s\n"
            "  Make sure Neo4j is running and credentials are correct.\n"
            "  URI: %s  USER: %s\n"
            "  Set NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD in .env.backend",
            e, uri, user,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Schema setup
# ---------------------------------------------------------------------------

SCHEMA_QUERIES = [
    # Disease is keyed by DOID — the common key across both datasets
    "CREATE CONSTRAINT disease_doid IF NOT EXISTS FOR (d:Disease) REQUIRE d.doid IS UNIQUE",
    "CREATE CONSTRAINT symptom_mesh IF NOT EXISTS FOR (s:Symptom) REQUIRE s.mesh_id IS UNIQUE",
    "CREATE CONSTRAINT gene_protein IF NOT EXISTS FOR (g:Gene) REQUIRE g.protein_id IS UNIQUE",

    "CREATE INDEX symptom_name IF NOT EXISTS FOR (s:Symptom) ON (s.name)",
    "CREATE INDEX disease_name IF NOT EXISTS FOR (d:Disease) ON (d.name)",
    "CREATE INDEX gene_name    IF NOT EXISTS FOR (g:Gene)    ON (g.name)",
    "CREATE INDEX disease_mesh IF NOT EXISTS FOR (d:Disease) ON (d.mesh_id)",
]


def setup_schema(driver):
    log.info("Setting up Neo4j schema (indexes + constraints)...")
    with driver.session() as session:
        for query in SCHEMA_QUERIES:
            try:
                session.run(query)
            except Exception as e:
                log.debug("Schema query skipped (likely exists): %s", e)
    log.info("Schema setup complete")


def wipe_graph(driver):
    log.warning("Wiping entire graph...")
    with driver.session() as session:
        while True:
            result = session.run(
                "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS deleted"
            )
            deleted = result.single()["deleted"]
            log.info("  Deleted %d nodes", deleted)
            if deleted == 0:
                break
    log.info("Graph wiped")


# ---------------------------------------------------------------------------
# Symptom-Disease loader (symptoms-DO.tsv)
# ---------------------------------------------------------------------------

class SymptomDiseaseLoader:
    """
    Loads symptoms-DO.tsv into Neo4j.

    File has a header row:
        symptom_name  disease_name  cooccurs  tfidf_score  disease_id  symptom_id  doid_code  doid_name

    Edge weight: we use tfidf_score (normalized), not raw cooccurs count.
    A higher tfidf_score means the symptom is more *specific* to that
    disease relative to its overall frequency in the literature.

    Filtering: tfidf_score has a long tail. MIN_SCORE keeps the
    meaningfully-associated pairs without dropping too much data.
    Adjust based on verify_graph() output once loaded.
    """

    MIN_SCORE = 1.0   # tfidf_score threshold
    BATCH_SIZE = 500

    def __init__(self, min_score: float = None):
        self.min_score = min_score if min_score is not None else self.MIN_SCORE

    def load(self, driver, tsv_path: Path):
        if not tsv_path.exists():
            log.error("symptoms-DO file not found: %s", tsv_path)
            return

        log.info("Loading symptom-disease data from %s", tsv_path)
        log.info("  Min tfidf_score filter: %.2f", self.min_score)

        rows = list(self._parse_tsv(tsv_path))
        log.info("  Parsed %d rows (before score filter)", len(rows))

        filtered = [r for r in rows if r["tfidf_score"] >= self.min_score]
        log.info("  After score filter (>= %.2f): %d rows", self.min_score, len(filtered))

        total = 0
        with tqdm(total=len(filtered), desc="Symptom-Disease edges", unit="edges") as pbar:
            for i in range(0, len(filtered), self.BATCH_SIZE):
                batch = filtered[i: i + self.BATCH_SIZE]
                self._load_batch(driver, batch)
                total += len(batch)
                pbar.update(len(batch))

        log.info("Symptom-Disease load complete: %d edges", total)

    def _parse_tsv(self, tsv_path: Path) -> Iterator[dict]:
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")

            required = {
                "symptom_name", "disease_name", "cooccurs", "tfidf_score",
                "disease_id", "symptom_id", "doid_code", "doid_name",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                log.error("symptoms-DO.tsv missing expected columns: %s", missing)
                return

            for line_num, row in enumerate(reader, start=2):
                try:
                    doid = row["doid_code"].strip()
                    if not doid:
                        continue  # skip rows without a DOID bridge

                    yield {
                        "symptom_mesh":   row["symptom_id"].strip(),
                        "symptom_name":   row["symptom_name"].strip(),
                        "disease_mesh":   row["disease_id"].strip(),
                        "disease_name":   row["disease_name"].strip(),
                        "doid":           doid,
                        "doid_name":      row["doid_name"].strip(),
                        "cooccurs":       int(row["cooccurs"]),
                        "tfidf_score":    float(row["tfidf_score"]),
                    }
                except (ValueError, KeyError) as e:
                    log.debug("Parse error on row %d: %s", line_num, e)
                    continue

    def _load_batch(self, driver, batch: list):
        """
        MERGE Symptom on mesh_id, Disease on doid (the cross-dataset key),
        and also stamp mesh_id onto the Disease node so the UMLS bridge
        (Phase 0d) has both identifiers to work with later.
        """
        query = """
        UNWIND $rows AS row

        MERGE (s:Symptom {mesh_id: row.symptom_mesh})
        ON CREATE SET
            s.name   = row.symptom_name,
            s.source = 'symptoms-do'
        ON MATCH SET
            s.name = row.symptom_name

        MERGE (d:Disease {doid: row.doid})
        ON CREATE SET
            d.name    = row.doid_name,
            d.mesh_id = row.disease_mesh,
            d.source  = 'symptoms-do'
        ON MATCH SET
            d.mesh_id = CASE WHEN d.mesh_id IS NULL THEN row.disease_mesh ELSE d.mesh_id END,
            d.name    = CASE WHEN d.name IS NULL THEN row.doid_name ELSE d.name END

        MERGE (s)-[r:PRESENTS_IN]->(d)
        ON CREATE SET
            r.cooccurs    = row.cooccurs,
            r.tfidf_score = row.tfidf_score,
            r.source      = 'symptoms-do'
        ON MATCH SET
            r.cooccurs    = row.cooccurs,
            r.tfidf_score = row.tfidf_score
        """
        with driver.session() as session:
            session.run(query, rows=batch)


# ---------------------------------------------------------------------------
# DISEASES (Jensen Lab) loader — knowledge + textmining channels
# ---------------------------------------------------------------------------

class DISEASESLoader:
    """
    Loads a Jensen Lab DISEASES TSV (7 columns, no header).

    knowledge channel:
        protein_id  gene_name  doid  disease_name  source_db  evidence_type  score(1-5 int)

    textmining channel:
        protein_id  gene_name  doid  disease_name  z_score    confidence(0-5 float)  url

    The threshold column differs by channel:
        knowledge  -> column 7 (1-5 stars)
        textmining -> column 6 (0-5 confidence)
    """

    MIN_SCORE_KNOWLEDGE  = 2.0
    MIN_SCORE_TEXTMINING = 3.0

    BATCH_SIZE = 500

    def load(self, driver, tsv_path: Path, channel: str):
        if not tsv_path.exists():
            log.error("DISEASES file not found: %s", tsv_path)
            return

        min_score = (
            self.MIN_SCORE_KNOWLEDGE if channel == "knowledge" else self.MIN_SCORE_TEXTMINING
        )

        log.info("Loading DISEASES (%s channel) from %s", channel, tsv_path)
        log.info("  Min score filter: %.1f", min_score)

        rows = list(self._parse_tsv(tsv_path, channel))
        log.info("  Parsed %d rows (before score filter)", len(rows))

        filtered = [r for r in rows if r["score"] >= min_score]
        log.info("  After score filter (>= %.1f): %d rows", min_score, len(filtered))

        total = 0
        with tqdm(total=len(filtered), desc=f"DISEASES ({channel})", unit="edges") as pbar:
            for i in range(0, len(filtered), self.BATCH_SIZE):
                batch = filtered[i: i + self.BATCH_SIZE]
                self._load_batch(driver, batch, channel)
                total += len(batch)
                pbar.update(len(batch))

        log.info("DISEASES (%s) load complete: %d gene-disease edges", channel, total)

    def _parse_tsv(self, tsv_path: Path, channel: str) -> Iterator[dict]:
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for line_num, row in enumerate(reader, start=1):
                if len(row) < 7:
                    log.debug("Skipping short row %d: %s", line_num, row)
                    continue
                try:
                    protein_id   = row[0].strip()
                    gene_name    = row[1].strip()
                    doid         = row[2].strip()
                    disease_name = row[3].strip()

                    if channel == "knowledge":
                        source_db = row[4].strip()      # e.g. UniProtKB-KW
                        score     = float(row[6].strip())  # 1-5 stars
                    else:  # textmining
                        source_db = "PubMed textmining"
                        score     = float(row[5].strip())  # 0-5 confidence

                    if not doid.startswith("DOID:"):
                        continue  # skip rows keyed to ICD10 etc., not DOID

                    yield {
                        "protein_id":   protein_id,
                        "gene_name":    gene_name,
                        "doid":         doid,
                        "disease_name": disease_name,
                        "source_db":    source_db,
                        "score":        score,
                        "channel":      channel,
                    }
                except (ValueError, IndexError) as e:
                    log.debug("Parse error on row %d: %s", line_num, e)
                    continue

    def _load_batch(self, driver, batch: list, channel: str):
        """
        MERGE Disease on doid — same key as SymptomDiseaseLoader uses,
        so genes attach directly to existing Disease nodes when the
        DOID matches (no separate bridging step needed).
        """
        query = """
        UNWIND $rows AS row

        MERGE (g:Gene {protein_id: row.protein_id})
        ON CREATE SET
            g.name   = row.gene_name,
            g.source = 'diseases'
        ON MATCH SET
            g.name = row.gene_name

        MERGE (d:Disease {doid: row.doid})
        ON CREATE SET
            d.name   = row.disease_name,
            d.source = 'diseases'
        ON MATCH SET
            d.name = CASE WHEN d.name IS NULL THEN row.disease_name ELSE d.name END

        MERGE (g)-[r:IMPLICATED_IN {channel: row.channel, source_db: row.source_db}]->(d)
        ON CREATE SET
            r.score = row.score
        ON MATCH SET
            r.score = row.score
        """
        with driver.session() as session:
            session.run(query, rows=batch)


# ---------------------------------------------------------------------------
# Verification queries
# ---------------------------------------------------------------------------

VERIFY_QUERIES = [
    ("Symptom nodes",                    "MATCH (s:Symptom) RETURN count(s) AS n"),
    ("Disease nodes",                    "MATCH (d:Disease) RETURN count(d) AS n"),
    ("Gene nodes",                       "MATCH (g:Gene) RETURN count(g) AS n"),
    ("PRESENTS_IN edges (symptom→disease)",  "MATCH ()-[r:PRESENTS_IN]->() RETURN count(r) AS n"),
    ("IMPLICATED_IN edges (gene→disease)",   "MATCH ()-[r:IMPLICATED_IN]->() RETURN count(r) AS n"),
    ("Diseases with BOTH symptoms AND genes", """
        MATCH (d:Disease)
        WHERE EXISTS { (d)<-[:PRESENTS_IN]-() } AND EXISTS { (d)<-[:IMPLICATED_IN]-() }
        RETURN count(d) AS n
    """),
]

SAMPLE_QUERIES = [
    (
        "Top 5 diseases by symptom count",
        """
        MATCH (s:Symptom)-[:PRESENTS_IN]->(d:Disease)
        RETURN d.name AS disease, count(s) AS symptom_count
        ORDER BY symptom_count DESC LIMIT 5
        """,
    ),
    (
        "Top 5 symptoms by disease count",
        """
        MATCH (s:Symptom)-[:PRESENTS_IN]->(d:Disease)
        RETURN s.name AS symptom, count(d) AS disease_count
        ORDER BY disease_count DESC LIMIT 5
        """,
    ),
    (
        "Diseases for symptoms matching 'Fever' or 'Cough'",
        """
        MATCH (s:Symptom)-[r:PRESENTS_IN]->(d:Disease)
        WHERE toLower(s.name) CONTAINS 'fever' OR toLower(s.name) CONTAINS 'cough'
        WITH d, collect(s.name) AS symptoms, avg(r.tfidf_score) AS avg_score
        RETURN d.name AS disease, symptoms, round(avg_score * 100) / 100 AS avg_score
        ORDER BY avg_score DESC LIMIT 5
        """,
    ),
    (
        "Genes implicated in diseases that also have symptom data",
        """
        MATCH (s:Symptom)-[:PRESENTS_IN]->(d:Disease)<-[:IMPLICATED_IN]-(g:Gene)
        RETURN d.name AS disease, collect(DISTINCT g.name)[..5] AS genes
        LIMIT 5
        """,
    ),
]


def verify_graph(driver):
    print("\n" + "=" * 60)
    print("Graph Verification Report")
    print("=" * 60)

    print("\nNode and edge counts:")
    with driver.session() as session:
        for label, query in VERIFY_QUERIES:
            result = session.run(query)
            record = result.single()
            n = record["n"] if record else 0
            status = "OK" if n > 0 else "!!"
            print(f"  [{status}] {label:<45} {n:>10,}")

    print("\nSample queries:")
    with driver.session() as session:
        for label, query in SAMPLE_QUERIES:
            print(f"\n  {label}:")
            try:
                result = session.run(query)
                records = result.data()
                if records:
                    for row in records[:5]:
                        print(f"    {row}")
                else:
                    print("    (no results)")
            except Exception as e:
                print(f"    ERROR: {e}")

    print("\n" + "=" * 60)
    print("Next step: python graph_rag/graph_retriever.py --test")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load symptoms-DO.tsv and DISEASES (Jensen Lab) data into Neo4j"
    )
    parser.add_argument(
        "--symptoms",
        type=str,
        default="../data/graph_rag/symptoms-DO.tsv",
        help="Path to symptoms-DO.tsv",
    )
    parser.add_argument(
        "--diseases",
        type=str,
        action="append",
        default=None,
        help=(
            "Path to a DISEASES TSV file. Use twice:\n"
            "  --diseases ../data/graph_rag/human_disease_knowledge_full.tsv\n"
            "  --diseases ../data/graph_rag/human_disease_textmining_full.tsv\n"
            "Channel (knowledge/textmining) is auto-detected from filename."
        ),
    )
    parser.add_argument(
        "--symptom-min-score",
        type=float,
        default=1.0,
        help="Minimum tfidf_score for symptom-disease edges (default: 1.0)",
    )
    parser.add_argument("--wipe", action="store_true", help="Wipe graph before loading")
    parser.add_argument("--verify", action="store_true", help="Only run verification queries")

    args = parser.parse_args()

    driver = get_driver()

    if args.verify:
        verify_graph(driver)
        driver.close()
        sys.exit(0)

    if args.wipe:
        confirm = input("This will DELETE all graph data. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)
        wipe_graph(driver)

    setup_schema(driver)

    # Load symptom-disease data
    symptoms_path = Path(args.symptoms)
    if symptoms_path.exists():
        SymptomDiseaseLoader(min_score=args.symptom_min_score).load(driver, symptoms_path)
    else:
        log.warning("symptoms-DO.tsv not found at %s — skipping.", symptoms_path)

    # Load DISEASES files
    diseases_paths = args.diseases or []
    if not diseases_paths:
        defaults = [
            "../data/graph_rag/human_disease_knowledge_full.tsv",
            "./data/graph_rag/human_disease_textmining_full.tsv",
        ]
        diseases_paths = [p for p in defaults if Path(p).exists()]
        if not diseases_paths:
            log.warning(
                "No DISEASES files found. Skipping.\n"
                "  Download from: https://download.jensenlab.org/\n"
                "    human_disease_knowledge_full.tsv\n"
                "    human_disease_textmining_full.tsv"
            )

    loader = DISEASESLoader()
    for path_str in diseases_paths:
        path = Path(path_str)
        if not path.exists():
            log.warning("DISEASES file not found: %s — skipping", path)
            continue
        channel = "textmining" if "textmining" in path.name.lower() else "knowledge"
        loader.load(driver, path, channel=channel)

    verify_graph(driver)

    driver.close()
    log.info("Done.")
