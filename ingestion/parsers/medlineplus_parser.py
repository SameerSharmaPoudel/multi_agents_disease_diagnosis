"""
Production-grade parser for the MedlinePlus XML dataset.

Output
------
List[MedicalDocument]

Each MedicalDocument contains

    • multiple DocumentSections
    • disease metadata
    • MeSH descriptor
    • synonyms
    • related topics
    • linked resources

Chunking, keyword extraction and embeddings happen later.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from ingestion.parsers.base_parser import BaseParser

from core.schemas import (
    MedicalDocument,
)


class MedlinePlusParser(BaseParser):

    SOURCE = "MedlinePlus"

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def parse(self):

        tree = ET.parse(self.input_path)

        root = tree.getroot()

        documents = []

        for topic in root.findall("health-topic"):

            document = self._parse_topic(topic)

            if document is not None:
                documents.append(document)

        return documents

    # ---------------------------------------------------------
    # Parse one health topic
    # ---------------------------------------------------------

    def _parse_topic(self, topic):

        disease = self.clean_text(
            topic.attrib.get("title")
        )

        if not disease:
            return None

        raw_id = topic.attrib.get("id")

        doc_id = self.generate_doc_id(
            self.SOURCE,
            raw_id,
        )

        # =====================================================
        # Sections
        # =====================================================

        sections = []

        # -----------------------------------------------------
        # Summary
        # -----------------------------------------------------
        summary_node = topic.find("full-summary")

        summary = self.create_section(
            document_id=doc_id,
            index=len(sections),
            title="Summary",
            section_type="summary",
            # text=topic.findtext("full-summary"),
            text=self.element_text(summary_node),
        )

        if summary:
            sections.append(summary)

        # -----------------------------------------------------
        # Content groups
        #
        # Some MedlinePlus releases contain textual sections
        # inside <group>.
        #
        # Others only contain navigation links.
        #
        # We keep textual groups as sections.
        # -----------------------------------------------------

        linked_resources = []

        for group in topic.findall("group"):

            title = self.clean_text(
                group.attrib.get("title")
            )

            body = self.element_text(group)

            if body and body != title:

                section = self.create_section(
                    document_id=doc_id,
                    index=len(sections),
                    title=title or "Section",
                    section_type="content",
                    text=body,
                )

                if section:
                    sections.append(section)

            # collect linked articles

            articles = []

            for article in group.findall("article"):

                article_title = self.clean_text(
                    article.text
                )

                articles.append(
                    {
                        "title": article_title,
                        "url": article.attrib.get("url"),
                    }
                )

            if articles:

                linked_resources.append(
                    {
                        "group": title,
                        "articles": articles,
                    }
                )

        # =====================================================
        # Synonyms
        # =====================================================

        synonyms = self.get_children_text(
            topic,
            "also-called",
        )

        # =====================================================
        # Related topics
        # =====================================================

        related_topics = self.get_children_text(
            topic,
            "related-topic",
        )

        # =====================================================
        # MeSH descriptor
        # =====================================================

        descriptor = topic.find(
            "./mesh-heading/descriptor"
        )

        disease_id = self.safe_text(descriptor)

        # =====================================================
        # Metadata
        # =====================================================

        metadata = self.build_metadata(

            topic_id=topic.attrib.get("id"),

            url=topic.attrib.get("url"),

            language=topic.attrib.get("language"),

            date_created=topic.attrib.get("date-created"),

            mesh_descriptor=disease_id,

            synonyms=synonyms,

            related_topics=related_topics,

            linked_resources=linked_resources,

            num_sections=len(sections),

        )

        # =====================================================
        # MedicalDocument
        # =====================================================

        return MedicalDocument(

            doc_id=doc_id,

            source=self.SOURCE,

            disease=disease,

            disease_id=disease_id,

            title=disease,

            sections=sections,

            keywords=[],

            metadata=metadata,
        )