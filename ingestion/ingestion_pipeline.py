"""
Main ingestion pipeline.

Pipeline

Parser
    ↓
Validator
    ↓
Keyword Extractor
    ↓
Chunker
    ↓
Embedding Generator
    ↓
Vector Database
"""

from __future__ import annotations

from typing import List

from core.schemas import MedicalDocument

from ingestion.document_validator import (
    DocumentValidator,
)


class IngestionPipeline:

    def __init__(
        self,
        parser,
        validator=None,
        keyword_extractor=None,
        chunker=None,
        embedding_generator=None,
        vector_store=None,
    ):

        self.parser = parser

        self.validator = validator or DocumentValidator()

        self.keyword_extractor = keyword_extractor

        self.chunker = chunker

        self.embedding_generator = embedding_generator

        self.vector_store = vector_store

    # ---------------------------------------------------------

    def run(self):

        print(f"Running {self.parser.__class__.__name__}")

        documents = self.parser.parse()

        print(f"Parsed {len(documents)} documents")

        documents = self.validate(documents)

        documents = self.extract_keywords(documents)

        chunks = self.chunk(documents)

        chunks = self.embed(chunks)

        self.store(chunks)

        return chunks

    # ---------------------------------------------------------

    def validate(
        self,
        documents: List[MedicalDocument],
    ):

        valid_documents = []

        for document in documents:

            try:

                self.validator.validate(document)

                valid_documents.append(document)

            except Exception as e:

                print(f"Validation failed: {e}")

        print(f"Validated {len(valid_documents)} documents")

        return valid_documents

    # ---------------------------------------------------------

    def extract_keywords(self, documents):

        if self.keyword_extractor is None:

            return documents

        return self.keyword_extractor.process(documents)

    # ---------------------------------------------------------

    def chunk(self, documents):

        if self.chunker is None:

            return documents

        return self.chunker.process(documents)

    # ---------------------------------------------------------

    def embed(self, chunks):

        if self.embedding_generator is None:

            return chunks

        return self.embedding_generator.process(chunks)

    # ---------------------------------------------------------

    def store(self, chunks):

        if self.vector_store is None:

            return

        self.vector_store.add(chunks)