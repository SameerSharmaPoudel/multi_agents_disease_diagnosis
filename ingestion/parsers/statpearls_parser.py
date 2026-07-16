"""
Parser for StatPearls NXML articles.

Output
------
List[MedicalDocument]

Each article becomes one MedicalDocument.

Major article sections become DocumentSections.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from ingestion.parsers.base_parser import BaseParser

from core.schemas import (
    MedicalDocument,
)


class StatPearlsParser(BaseParser):

    SOURCE = "StatPearls"

    # ---------------------------------------------------------

    def parse(self):

        documents = []

        for file_path in self.iter_files(".nxml"):

            try:

                document = self._parse_article(file_path)

                if document is not None:
                    documents.append(document)

            except Exception as e:

                print(
                    f"[WARNING] Failed to parse "
                    f"{file_path}: {e}"
                )

        return documents

    # ---------------------------------------------------------

    def _parse_article(self, file_path):

        tree = ET.parse(file_path)

        root = tree.getroot()

        # =====================================================
        # Article title
        # =====================================================

        title_node = root.find(".//article-title")

        title = self.safe_text(title_node)

        if not title:
            return None

        # =====================================================
        # ID
        # =====================================================

        article_id_node = root.find(".//article-id")

        raw_id = self.safe_text(article_id_node)

        # Fall back to the unique filename
        if not raw_id:

            raw_id = file_path.stem        # article-33977

        # Absolute last resort
        if not raw_id:

            raw_id = title

        doc_id = self.generate_doc_id(
            self.SOURCE,
            raw_id,
        )
        # =====================================================
        # Disease
        #
        # For StatPearls we usually use title
        # =====================================================

        disease = title

        # =====================================================
        # Sections
        # =====================================================

        sections = []

        sec_index = 0

        for sec in root.findall(".//sec"):

            sec_title = self.safe_text(
                sec.find("title")
            )

            paragraphs = []

            for p in sec.findall(".//p"):

                text = self.safe_text(p)

                if text:
                    paragraphs.append(text)

            section_text = "\n".join(paragraphs)

            section_type = (
                            sec_title.lower()
                            .replace("&", "and")
                            .replace("/", "_")
                            .replace(" ", "_")
                        )

            section = self.create_section(

                document_id=doc_id,

                index=sec_index,

                title=sec_title
                if sec_title
                else f"Section {sec_index}",

                section_type=section_type,

                text=section_text,
            )

            if section:

                sections.append(section)

                sec_index += 1

        # =====================================================
        # Abstract
        # =====================================================

        abstract_paragraphs = []

        for p in root.findall(".//abstract//p"):

            text = self.safe_text(p)

            if text:
                abstract_paragraphs.append(text)

        abstract_text = "\n".join(
            abstract_paragraphs
        )

        abstract_section = self.create_section(

            document_id=doc_id,

            index=999,

            title="Abstract",

            section_type="abstract",

            text=abstract_text,
        )

        if abstract_section:

            sections.insert(
                0,
                abstract_section,
            )

        # =====================================================
        # Keywords
        # =====================================================

        keywords = []

        for kwd in root.findall(".//kwd"):

            text = self.safe_text(kwd)

            if text:
                keywords.append(text)

        # =====================================================
        # Authors
        # =====================================================

        authors = []

        for contrib in root.findall(
            ".//contrib[@contrib-type='author']"
        ):

            name = []

            surname = self.safe_text(
                contrib.find(".//surname")
            )

            given = self.safe_text(
                contrib.find(".//given-names")
            )

            if given:
                name.append(given)

            if surname:
                name.append(surname)

            full_name = " ".join(name)

            if full_name:
                authors.append(full_name)

        # =====================================================
        # Metadata
        # =====================================================

        article_id_node = root.find(".//article-id")

        article_id = self.safe_text(article_id_node)

        metadata = {

            "source_file": str(file_path),

            "article_id": article_id,

            "authors": authors,

            "keywords": keywords,

            "num_sections": len(sections),

        }
        
        metadata = self.remove_empty_metadata(
            metadata
        )

        # =====================================================
        # Document
        # =====================================================

        return MedicalDocument(

            doc_id=doc_id,

            source=self.SOURCE,

            disease=disease,

            disease_id=None,

            title=title,

            sections=sections,

            keywords=keywords,

            metadata=metadata,
        )