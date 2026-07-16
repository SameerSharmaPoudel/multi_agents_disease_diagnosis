"""
Parser for the MedQuAD dataset.

Directory structure

MedQuAD/
    CancerGov/
        *.xml
    GHR/
        *.xml
    NIH/
        *.xml
    ...

Each XML file becomes ONE MedicalDocument.

Each Question/Answer pair becomes ONE DocumentSection.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from ingestion.parsers.base_parser import BaseParser

from core.schemas import (
    MedicalDocument,
)


class MedQuADParser(BaseParser):

    SOURCE = "MedQuAD"

    # ---------------------------------------------------------

    def parse(self):

        documents = []

        for xml_file in self.iter_files(".xml"):

            try:

                tree = ET.parse(xml_file)

                root = tree.getroot()

                document = self._parse_document(
                    root,
                    xml_file.parent.name,     # dataset/source
                )

                if document is not None:
                    documents.append(document)

            except Exception as e:

                print(f"Failed parsing {xml_file}: {e}")

        return documents

    # ---------------------------------------------------------

    def _parse_document(self, root, dataset_name):

        disease = self.clean_text(
            root.findtext("Focus")
        )

        if not disease:
            return None

        raw_id = root.attrib.get("id", disease)

        # doc_id = self.generate_doc_id(
        #     self.SOURCE,
        #     raw_id,
        # )

        raw_unique = f"{dataset_name}_{raw_id}"

        doc_id = self.generate_doc_id(
            self.SOURCE,
            raw_unique,
        )

        # =====================================================
        # UMLS
        # =====================================================

        disease_id = None

        cui = root.find("./FocusAnnotations/UMLS/CUIs/CUI")

        if cui is not None:

            disease_id = self.clean_text(cui.text)

        semantic_types = []

        for st in root.findall(
            "./FocusAnnotations/UMLS/SemanticTypes/SemanticType"
        ):

            text = self.clean_text(st.text)

            if text:
                semantic_types.append(text)

        semantic_group = self.clean_text(

            root.findtext(
                "./FocusAnnotations/UMLS/SemanticGroup"
            )

        )

        # =====================================================
        # Sections
        # =====================================================

        sections = []

        qapairs = root.find("QAPairs")

        if qapairs is not None:

            index = 0

            for pair in qapairs.findall("QAPair"):

                question = pair.find("Question")

                answer = pair.find("Answer")

                if answer is None:
                    continue

                # question_text = self.clean_text(question.text)
                question_text = self.element_text(question)

                # answer_text = self.clean_text(answer.text)
                answer_text = self.element_text(answer)

                qtype = ""

                qid = ""

                if question is not None:

                    qtype = question.attrib.get("qtype", "")

                    qid = question.attrib.get("qid", "")

                title = qtype.title()

                if not title:
                    title = question_text

                section = self.create_section(

                    document_id=doc_id,

                    index=index,

                    title=title,

                    section_type="qa",

                    text=answer_text,

                    metadata={

                        "question": question_text,

                        "question_type": qtype,

                        "question_id": qid,

                        "pair_id": pair.attrib.get("pid"),

                    },

                )

                if section:

                    sections.append(section)

                    index += 1

        # =====================================================
        # Metadata
        # =====================================================

        metadata = {

            "dataset": dataset_name,

            "document_id": raw_id,

            "semantic_group": semantic_group,

            "semantic_types": semantic_types,

            "num_sections": len(sections),

        }

        metadata = self.remove_empty_metadata(metadata)

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