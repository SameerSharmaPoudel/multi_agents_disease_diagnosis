from __future__ import annotations

from typing import List

from core.schemas import (
    MedicalDocument,
    MedicalChunk,
)
from ingestion.chunking.base_chunker import BaseChunker


class MedicalChunker(BaseChunker):
    """
    Default chunker used throughout the ingestion pipeline.

    Characteristics
    ---------------
    - Preserves section boundaries strictly
    - Splits text on sentence boundaries (keeps clinical facts atomic)
    - Regulates chunk scale natively using sub-word LLM tokens
    - Carries extensive document and section metadata schemas
    """

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def chunk(
        self,
        document: MedicalDocument,
    ) -> List[MedicalChunk]:

        chunks: List[MedicalChunk] = []
        chunk_index = 0

        # -----------------------------------------------------
        # Process each document section independently
        # -----------------------------------------------------
        for section in document.sections:

            # Skip empty sections
            if not section.text.strip():
                continue

            # Split section into clean sentence-based token bounded strings
            split_chunks = self.split_text(
                section.text
            )

            # Map raw chunks into structured MedicalChunk models
            for chunk_text in split_chunks:

                chunk = self.create_chunk(
                    document=document,
                    section=section,
                    chunk_text=chunk_text,
                    chunk_index=chunk_index,
                )

                chunks.append(chunk)
                chunk_index += 1

        return chunks

    # ---------------------------------------------------------
    # Batch helper
    # ---------------------------------------------------------

    def chunk_documents(
        self,
        documents: List[MedicalDocument],
    ) -> List[MedicalChunk]:
        """
        Chunk an entire collection of documents sequentially.
        """
        all_chunks: List[MedicalChunk] = []

        for document in documents:
            all_chunks.extend(
                self.chunk(document)
            )

        return all_chunks