"""
Validates MedicalChunk objects.

The validator NEVER modifies a chunk.

It only verifies that chunking produced valid retrieval units.
"""

from __future__ import annotations

from core.schemas import MedicalChunk


class ChunkValidationError(Exception):
    pass


class ChunkValidator:

    # ---------------------------------------------------------

    @classmethod
    def validate(
        cls,
        chunk: MedicalChunk,
    ) -> bool:

        cls._validate_identity(chunk)

        cls._validate_content(chunk)

        cls._validate_metadata(chunk)

        return True

    # ---------------------------------------------------------

    @staticmethod
    def _validate_identity(chunk):

        if not chunk.chunk_id:
            raise ChunkValidationError(
                "Missing chunk_id"
            )

        if not chunk.document_id:
            raise ChunkValidationError(
                f"{chunk.chunk_id}: missing document_id"
            )

        if chunk.chunk_index < 0:
            raise ChunkValidationError(
                f"{chunk.chunk_id}: invalid chunk_index"
            )

    # ---------------------------------------------------------

    @staticmethod
    def _validate_content(chunk):

        if not chunk.source:
            raise ChunkValidationError(
                f"{chunk.chunk_id}: missing source"
            )

        if not chunk.disease:
            raise ChunkValidationError(
                f"{chunk.chunk_id}: missing disease"
            )

        if not chunk.section:
            raise ChunkValidationError(
                f"{chunk.chunk_id}: missing section"
            )

        if not chunk.text.strip():
            raise ChunkValidationError(
                f"{chunk.chunk_id}: empty text"
            )

    # ---------------------------------------------------------

    @staticmethod
    def _validate_metadata(chunk):

        required = [
            "section_id",
            "section_title",
            "section_type",
        ]

        for key in required:

            if key not in chunk.metadata:

                raise ChunkValidationError(
                    f"{chunk.chunk_id}: metadata missing '{key}'"
                )