# graph_rag/graph_retriever.py
"""
Phase 1 — Graph RAG: Knowledge Graph Retriever
===================================================
This module acts as the bridging retriever for multi-agent RAG pipelines.
It takes unstructured symptom inputs, searches the structural Neo4j database 
using optimized multi-hop Cypher queries, calculates rank based on weights 
(tfidf_score and confidence scores), and formats pristine contexts for LLMs.

Usage (Testing directly):
    python graph_rag/graph_retriever.py --test

Usage (Importing into an AI Agent script):
    from graph_rag.graph_retriever import GraphRetriever
    retriever = GraphRetriever()
    results = retriever.retrieve_context_by_symptoms(["Fever", "Cough"])
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# Set up logging format
log = logging.getLogger("GraphRetriever")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Look for configuration environment settings 
load_dotenv("../.env.backend")


class GraphRetriever:
    """Manages secure access, filtering math, and structural traversal rules inside Neo4j."""

    def __init__(self):
        self.driver = self._get_driver()

    def _get_driver(self):
        """Initializes connection to the running Neo4j database."""
        try:
            from neo4j import GraphDatabase
        except ImportError:
            log.error("neo4j package not found. Run: pip install neo4j")
            sys.exit(1)

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")

        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
            return driver
        except Exception as e:
            log.error("Retriever failed to connect to Neo4j database at %s: %s", uri, e)
            sys.exit(1)

    def close(self):
        """Gracefully closes active driver session pools."""
        if self.driver:
            self.driver.close()

    def retrieve_context_by_symptoms(
        self, 
        symptoms: List[str], 
        min_tfidf: float = 1.0, 
        min_gene_score: float = 2.0, 
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Executes an advanced multi-hop Cypher traversal query.
        
        Symptom -> [PRESENTS_IN] -> Disease <- [IMPLICATED_IN] - Gene
        
        Aggregates match weights, isolates overlapping candidate conditions,
        and dynamically extracts matching underlying biomarkers.
        """
        # Formulate query using precise string properties on edges
        query = """
        MATCH (s:Symptom)-[r1:PRESENTS_IN]->(d:Disease)
        WHERE any(input_sym IN $symptoms WHERE toLower(s.name) = toLower(input_sym))
          AND r1.tfidf_score >= $min_tfidf
        
        WITH d, collect(DISTINCT s.name) AS matched_symptoms, avg(r1.tfidf_score) AS avg_symptom_weight
        
        OPTIONAL MATCH (g:Gene)-[r2:IMPLICATED_IN]->(d)
        WHERE r2.score >= $min_gene_score
        
        WITH d, matched_symptoms, avg_symptom_weight, 
             collect(DISTINCT {name: g.name, score: r2.score, source: r2.source_db})[..10] AS associated_genes
        
        RETURN d.name AS disease,
               d.doid AS doid,
               matched_symptoms,
               round(avg_symptom_weight * 100) / 100 AS dynamic_rank,
               associated_genes
        ORDER BY size(matched_symptoms) DESC, dynamic_rank DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                symptoms=symptoms,
                min_tfidf=min_tfidf,
                min_gene_score=min_gene_score,
                limit=limit
            )
            records = result.data()
            
        # Formulate an optimized raw string format out of the records for direct insertion into LLM contexts
        llm_context_blocks = []
        for rank, rec in enumerate(records, start=1):
            gene_strings = [f"{g['name']} (Score: {g['score']})" for g in rec['associated_genes'] if g['name']]
            genes_text = ", ".join(gene_strings) if gene_strings else "No highly validated genes extracted"
            
            block = (
                f"Candidate Rank #{rank}: {rec['disease'].upper()} ({rec['doid']})\n"
                f" - Intersected Symptoms: {', '.join(rec['matched_symptoms'])}\n"
                f" - Mean Specificity Score (TF-IDF): {rec['dynamic_rank']}\n"
                f" - Implicated Bio-Markers: {genes_text}\n"
            )
            llm_context_blocks.append(block)
            
        raw_llm_context = "\n".join(llm_context_blocks) if llm_context_blocks else "No matching diagnostic pathways found."
        
        return {
            "structured_data": records,
            "llm_context_prompt": raw_llm_context
        }


# ---------------------------------------------------------------------------
# Direct Execution Mode (Testing Harness)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query and test the Knowledge Graph RAG retriever module")
    parser.add_argument("--test", action="store_true", help="Runs diagnostic symptom queries to verify graph access")
    args = parser.parse_args()

    if not args.test:
        parser.print_help()
        sys.exit(0)

    print("\n" + "=" * 60)
    print("Graph RAG Retriever Test Runner")
    print("=" * 60)

    retriever = GraphRetriever()
    
    # Test cases representing standard user queries
    test_scenarios = [
        ["Fever", "Cough"],
        ["Nausea", "Weight Loss", "Pain"],
        ["Fatigue", "Edema"]
    ]
    
    try:
        for index, scenario in enumerate(test_scenarios, start=1):
            print(f"\n[Test Case #{index}] Querying for User Symptoms: {scenario}")
            response = retriever.retrieve_context_by_symptoms(symptoms=scenario, limit=3)
            
            print("\n>>> Parsed Context Sent to LLM Prompt Engine:")
            print("-" * 60)
            print(response["llm_context_prompt"])
            print("-" * 60)
            
    finally:
        retriever.close()
        print("Test runner complete. Driver pool disconnected safely.\n")