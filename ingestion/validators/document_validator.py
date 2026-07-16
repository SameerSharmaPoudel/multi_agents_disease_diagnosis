"""
Validates MedicalDocument objects produced by parsers.

The validator NEVER modifies a document.

Responsibilities
----------------
✓ Validate required document fields
✓ Validate sections
✓ Validate metadata types
✓ Detect duplicate section IDs
✓ Detect duplicate section titles
✓ Detect empty documents

Raises
------
ValidationError
"""

from __future__ import annotations

from typing import Set

from core.schemas import (
    MedicalDocument,
    DocumentSection,
)


class ValidationError(Exception):
    """Raised when a MedicalDocument fails validation."""
    pass


class DocumentValidator:

    # ---------------------------------------------------------

    @classmethod
    def validate(
        cls,
        document: MedicalDocument,
    ) -> bool:

        cls._validate_document(document)

        cls._validate_sections(document)

        cls._validate_metadata(document)

        return True

    # ---------------------------------------------------------
    # Document validation
    # ---------------------------------------------------------

    @staticmethod
    def _validate_document(document: MedicalDocument):

        if not isinstance(document, MedicalDocument):
            raise ValidationError("Object is not a MedicalDocument.")

        if not document.doc_id:
            raise ValidationError("Missing document ID.")

        if not document.source:
            raise ValidationError(
                f"{document.doc_id}: missing source."
            )

        if not document.disease:
            raise ValidationError(
                f"{document.doc_id}: missing disease."
            )

        if not document.title:
            raise ValidationError(
                f"{document.doc_id}: missing title."
            )

        if not isinstance(document.sections, list):
            raise ValidationError(
                f"{document.doc_id}: sections must be a list."
            )

        if len(document.sections) == 0:
            raise ValidationError(
                f"{document.doc_id}: document contains no sections."
            )

    # ---------------------------------------------------------
    # Section validation
    # ---------------------------------------------------------

    @staticmethod
    def _validate_sections(document: MedicalDocument):

        section_ids: Set[str] = set()
        section_titles: Set[str] = set()

        for section in document.sections:

            if not isinstance(section, DocumentSection):
                raise ValidationError(
                    f"{document.doc_id}: invalid section object."
                )

            if not section.section_id:
                raise ValidationError(
                    f"{document.doc_id}: section without ID."
                )

            if section.section_id in section_ids:
                raise ValidationError(
                    f"{document.doc_id}: duplicate section ID "
                    f"{section.section_id}"
                )

            section_ids.add(section.section_id)

            if not section.title.strip():
                raise ValidationError(
                    f"{document.doc_id}: empty section title."
                )

            if not section.text.strip():
                raise ValidationError(
                    f"{document.doc_id}: section '{section.title}' is empty."
                )

            if section.title in section_titles:
                print(
                    f"Warning: duplicate section title "
                    f"'{section.title}' "
                    f"in {document.doc_id}"
                )

            section_titles.add(section.title)

    # ---------------------------------------------------------
    # Metadata validation
    # ---------------------------------------------------------

    @staticmethod
    def _validate_metadata(document: MedicalDocument):

        if not isinstance(document.metadata, dict):
            raise ValidationError(
                f"{document.doc_id}: metadata must be a dictionary."
            )

        for section in document.sections:

            if not isinstance(section.metadata, dict):
                raise ValidationError(
                    f"{document.doc_id}: "
                    f"section metadata must be dictionary."
                )