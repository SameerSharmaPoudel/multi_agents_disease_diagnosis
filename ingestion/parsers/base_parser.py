from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Generator
from bs4 import BeautifulSoup

import hashlib
import html
import re

from core.schemas import (
    MedicalDocument,
    DocumentSection,
)


class BaseParser(ABC):

    SOURCE = "Unknown"

    def __init__(self, input_path: str):

        self.input_path = Path(input_path)

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Input path not found:\n{self.input_path}"
            )

    # ---------------------------------------------------------
    # Required API
    # ---------------------------------------------------------

    @abstractmethod
    def parse(self) -> List[MedicalDocument]:
        pass

    # ---------------------------------------------------------
    # File iteration
    # ---------------------------------------------------------

    def iter_files(
        self,
        suffixes: Optional[List[str]] = None,
    ) -> Generator[Path, None, None]:

        if self.input_path.is_file():
            yield self.input_path
            return

        suffixes = suffixes or []

        for file in self.input_path.rglob("*"):

            if not file.is_file():
                continue

            if suffixes and file.suffix.lower() not in suffixes:
                continue

            yield file

    # ---------------------------------------------------------
    # Cleaning
    # ---------------------------------------------------------

    # @staticmethod
    # def clean_text(text: Optional[str]) -> str:

    #     if text is None:
    #         return ""

    #     text = html.unescape(text)

    #     text = text.replace("\r", " ")
    #     text = text.replace("\n", " ")

    #     text = re.sub(r"\s+", " ", text)

    #     return text.strip()
    
    @staticmethod
    def clean_text(text):

        if text is None:
            return ""

        text = html.unescape(text)

        text = BeautifulSoup(text, "html.parser").get_text(
            separator=" ",
            strip=True,
        )

        text = re.sub(r"\s+", " ", text)

        return text

    @classmethod
    def safe_text(cls, element) -> str:

        if element is None:
            return ""

        return cls.clean_text(element.text)

    @classmethod
    def element_text(cls, element) -> str:
        """
        Returns all nested XML text.

        Useful for StatPearls .nxml files.
        """

        if element is None:
            return ""

        return cls.clean_text(
            "".join(element.itertext())
        )

    @staticmethod
    def safe_attrib(element, key: str):

        if element is None:
            return None

        return element.attrib.get(key)

    # ---------------------------------------------------------
    # IDs
    # ---------------------------------------------------------

    @classmethod
    def generate_doc_id(
        cls,
        source: str,
        raw_id: Optional[str],
    ):

        if raw_id:

            return f"{source.lower()}_{raw_id}"

        digest = hashlib.md5(
            source.encode()
        ).hexdigest()[:12]

        return f"{source.lower()}_{digest}"

    @staticmethod
    def generate_section_id(
        document_id: str,
        index: int,
    ):

        return f"{document_id}_sec_{index:03d}"

    # ---------------------------------------------------------
    # Sections
    # ---------------------------------------------------------

    @classmethod
    def create_section(
        cls,
        document_id: str,
        index: int,
        title: str,
        text: str,
        section_type: str = "content",
        metadata: Optional[dict] = None,
    ):

        text = cls.clean_text(text)

        if not text:
            return None

        return DocumentSection(

            section_id=cls.generate_section_id(
                document_id,
                index,
            ),

            title=title.strip(),

            section_type=section_type,

            text=text,

            metadata=metadata or {},
        )

    # ---------------------------------------------------------
    # XML helpers
    # ---------------------------------------------------------

    @classmethod
    def get_children_text(
        cls,
        parent,
        tag,
    ):

        values = []

        for child in parent.findall(tag):

            text = cls.safe_text(child)

            if text:
                values.append(text)

        return values

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    @staticmethod
    def build_metadata(**kwargs):

        return {
            k: v
            for k, v in kwargs.items()
            if v not in (
                None,
                "",
                [],
                {},
            )
        }

    @staticmethod
    def remove_empty_metadata(metadata):

        return {
            k: v
            for k, v in metadata.items()
            if v not in (
                None,
                "",
                [],
                {},
            )
        }