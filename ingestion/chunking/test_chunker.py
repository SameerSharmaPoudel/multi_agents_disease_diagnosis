from __future__ import annotations

import random
from collections import Counter

import tiktoken

from ingestion.parsers.medlineplus_parser import MedlinePlusParser
from ingestion.parsers.medquad_parser import MedQuADParser
from ingestion.parsers.statpearls_parser import StatPearlsParser

from ingestion.chunking.medical_chunker import MedicalChunker

from ingestion.validators.document_validator import (
    DocumentValidator,
)

from ingestion.validators.chunk_validator import (
    ChunkValidator,
    ChunkValidationError,
)

def print_chunk(chunk):

    print("=" * 80)

    print(chunk.chunk_id)

    print("=" * 80)

    print("Disease :", chunk.disease)
    print("Source  :", chunk.source)
    print("Section :", chunk.section)

    print("\nMetadata")

    print("-" * 80)

    for k, v in chunk.metadata.items():
        print(f"{k}: {v}")

    print("\nText")

    print("-" * 80)

    print(chunk.text[:1000])

    print()

def chunk_statistics(chunks, chunker):

    print()

    print("=" * 15, "Statistics", "=" * 15)

    print()

    print("Chunks :", len(chunks))

    # lengths = [len(c.text) for c in chunks]
    token_lengths = [
    len(chunker.tokenizer.encode(c.text))
    for c in chunks
]

    print(
        "Average tokens :",
        round(sum(token_lengths) / len(token_lengths), 1),
    )

    print(
        "Maximum tokens :",
        max(token_lengths),
    )

    print(
        "Minimum tokens :",
        min(token_lengths),
    )

    print()

    print("=" * 15, "Sections", "=" * 15)

    counter = Counter()

    for c in chunks:
        counter[c.section] += 1

    for name, count in counter.most_common():
        print(f"{name:<35} {count}")

    print()

    print("=" * 15, "Duplicate Chunk IDs", "=" * 15)

    ids = [c.chunk_id for c in chunks]

    duplicates = len(ids) - len(set(ids))

    print(duplicates)

    print("=" * 15, "Tokens exceeding the limit", "=" * 15)
    for chunk in chunks:

        tokens = len(
            chunker.tokenizer.encode(chunk.text)
        )

        if tokens > 512:

            print(chunk.chunk_id)

            print(tokens)

            print(chunk.section)

            print(chunk.text[:300])


def run_chunk_test(name, parser):

    print()

    print("=" * 90)
    print(name)
    print("=" * 90)

    documents = parser.parse()

    chunker = MedicalChunker()

    chunks = []

    passed = 0
    failed = 0

    for doc in documents:

        try:

            DocumentValidator.validate(doc)

            doc_chunks = chunker.chunk(doc)

            for chunk in doc_chunks:

                ChunkValidator.validate(chunk)

            chunks.extend(doc_chunks)

            passed += 1

        except Exception as e:

            failed += 1

            print(e)

    print()

    print("Documents :", len(documents))
    print("Chunks    :", len(chunks))

    print()

    print("Validation passed :", passed)
    print("Validation failed :", failed)

    chunk_statistics(chunks, chunker)

    print()

    print("=" * 20, "Random Chunks", "=" * 20)

    for chunk in random.sample(
        chunks,
        min(5, len(chunks)),
    ):

        print_chunk(chunk)

    return chunks 


if __name__ == "__main__":

    # run_chunk_test(

    #     "MedlinePlus",

    #     MedlinePlusParser(
    #         "data/vector_rag/medlineplus/medlineplus_topics.xml"
    #     ),

    # )

    # run_chunk_test(

    #     "MedQuAD",

    #     MedQuADParser(
    #         "data/vector_rag/MedQuAD"
    #     ),

    # )

    run_chunk_test(

        "StatPearls",

        StatPearlsParser(
            "data/vector_rag/statpearls"
        ),

    )


    # uv run python -m ingestion.chunking.test_chunker