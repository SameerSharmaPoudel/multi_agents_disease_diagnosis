"""
Comprehensive parser testing.

Tests

✓ Parsing
✓ Validation
✓ Statistics
✓ Metadata
✓ Manual inspection
✓ Duplicate IDs
✓ Section statistics


"""

from collections import Counter
import random

from ingestion.parsers.medlineplus_parser import MedlinePlusParser
from ingestion.parsers.medquad_parser import MedQuADParser
from ingestion.parsers.statpearls_parser import StatPearlsParser

from ingestion.validators.document_validator import (
    DocumentValidator,
    ValidationError,
)


# ============================================================
# Validation
# ============================================================

def validate_documents(documents):

    passed = 0
    failed = 0

    for doc in documents:

        try:

            DocumentValidator.validate(doc)

            passed += 1

        except ValidationError as e:

            failed += 1

            print(e)

    print(f"\nValidation passed : {passed}")
    print(f"Validation failed : {failed}")

    return passed, failed


# ============================================================
# Statistics
# ============================================================

def parser_statistics(documents):

    print("\n========== Statistics ==========\n")

    print(f"Documents : {len(documents)}")

    total_sections = sum(len(d.sections) for d in documents)

    print(f"Sections  : {total_sections}")

    avg_sections = total_sections / len(documents)

    print(f"Average sections/document : {avg_sections:.2f}")

    diseases = len(set(d.disease for d in documents))

    print(f"Unique diseases : {diseases}")

    total_chars = sum(
        len(sec.text)
        for d in documents
        for sec in d.sections
    )

    avg_chars = total_chars / total_sections

    print(f"Average section length : {avg_chars:.0f} characters")


# ============================================================
# Section statistics
# ============================================================

def section_statistics(documents):

    counter = Counter()

    for doc in documents:

        for sec in doc.sections:

            counter[sec.title] += 1

    print("\n========== Section Distribution ==========\n")

    for title, count in counter.most_common():

        print(f"{title:<35} {count}")


# ============================================================
# Metadata inspection
# ============================================================

def metadata_statistics(documents):

    print("\n========== Metadata ==========\n")

    keys = Counter()

    for doc in documents:

        for key in doc.metadata:

            keys[key] += 1

    for key, count in keys.items():

        print(f"{key:<25} {count}")


# ============================================================
# Duplicate IDs
# ============================================================

def duplicate_check(documents):

    ids = Counter(doc.doc_id for doc in documents)

    duplicates = [

        doc_id

        for doc_id, count in ids.items()

        if count > 1

    ]

    print("\n========== Duplicate IDs ==========\n")

    print(len(duplicates))

    if duplicates:

        print(duplicates[:20])


# ============================================================
# Manual inspection
# ============================================================

def inspect_document(document):

    print("\n")
    print("=" * 80)

    print(document.doc_id)

    print("=" * 80)

    print("Disease :", document.disease)

    print("Source  :", document.source)

    print("Title   :", document.title)

    print("Disease ID :", document.disease_id)

    print()

    print("Metadata")

    print("-" * 80)

    for k, v in document.metadata.items():

        print(f"{k}: {v}")

    print()

    print("Sections")

    print("-" * 80)

    for sec in document.sections:

        print()

        print(sec.title)

        print("-" * len(sec.title))

        print(sec.text[:300])

        print()


# ============================================================
# Random inspection
# ============================================================

def inspect_random_documents(documents, n=3):

    print("\n========== Random Documents ==========\n")

    docs = random.sample(

        documents,

        min(n, len(documents))

    )

    for doc in docs:

        inspect_document(doc)


# ============================================================
# Longest document
# ============================================================

def longest_document(documents):

    longest = max(

        documents,

        key=lambda d: sum(len(s.text) for s in d.sections)

    )

    print("\n========== Longest Document ==========\n")

    print(longest.doc_id)

    print(longest.disease)

    print(len(longest.sections))

    print(sum(len(s.text) for s in longest.sections))


# ============================================================
# Test helper
# ============================================================

def run_test(name, parser):

    print("\n")
    print("=" * 90)
    print(name)
    print("=" * 90)

    documents = parser.parse()

    print(f"\nParsed {len(documents)} documents")

    validate_documents(documents)

    parser_statistics(documents)

    section_statistics(documents)

    metadata_statistics(documents)

    duplicate_check(documents)

    longest_document(documents)

    inspect_document(documents[0])

    inspect_random_documents(documents)

    return documents


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    from pathlib import Path

    # Find the project root directory dynamically (up two levels from ingestion/parsers/)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    # Build the absolute path to your data file
    medlineplus_file_path = PROJECT_ROOT / "data" / "vector_rag" / "medlineplus" / "medlineplus_topics.xml"
    medquad_file_path = PROJECT_ROOT / "data" / "vector_rag" / "medquad" 
    statpearls_file_path = PROJECT_ROOT / "data" / "vector_rag" / "statpearls" 

    medline_docs = run_test(

        "MedlinePlus",

        MedlinePlusParser(

            input_path=str(medlineplus_file_path)

        ),

    )

    # medquad_docs = run_test(

    #     "MedQuAD",

    #     MedQuADParser(

    #         input_path=str(medquad_file_path)

    #     ),

    # )

    # statpearls_docs = run_test(

    #     "StatPearls",

    #     StatPearlsParser(

    #         input_path=str(statpearls_file_path)

    #     ),

    # )

    
    
    # uv run python -m ingestion.parsers.test_parsers