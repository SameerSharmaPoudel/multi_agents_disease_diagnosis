from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import tiktoken  # Tokenizer factory

from core.schemas import (
    MedicalDocument,
    DocumentSection,
    MedicalChunk,
)


class BaseChunker(ABC):

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        encoding_name: str = "cl100k_base",  # Default OpenAI tokenizer (gpt-3.5/gpt-4)
    ):
        if overlap_tokens >= max_tokens:
            raise ValueError(
                "overlap_tokens must be smaller than max_tokens."
            )

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
        # Initialize the native tokenizer engine
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    # ---------------------------------------------------------
    # Required API
    # ---------------------------------------------------------

    @abstractmethod
    def chunk(
        self,
        document: MedicalDocument,
    ) -> List[MedicalChunk]:
        """
        Convert one MedicalDocument into MedicalChunks.
        """
        pass

    # ---------------------------------------------------------
    # ID generation
    # ---------------------------------------------------------

    @staticmethod
    def generate_chunk_id(
        document_id: str,
        chunk_index: int,
    ) -> str:
        return f"{document_id}_chunk_{chunk_index:04d}"

    # ---------------------------------------------------------
    # Token-Bounded Sentence Splitter
    # ---------------------------------------------------------

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Helper to split text into sentences.
        Uses a regular expression that respects basic medical abbreviations.
        """
        # Split on punctuation followed by space/capital letter, avoiding common medical abbreviations
        sentence_end = re.compile(r'(?<!\bdr)(?<!\bmg)(?<!\bvs)(?<!\biv)(?<=[.!?])\s+')
        sentences = sentence_end.split(text.strip())
        return [s for s in sentences if s.strip()]

    def split_text(
        self,
        text: str,
    ) -> List[str]:
        """
        Split text into chunks of whole sentences bounded by token length constraints.
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = len(self.tokenizer.encode(sentence))

            # Edge case: If a single sentence exceeds max_tokens on its own, 
            # we are forced to break it down forcibly by sub-tokens.
            if sentence_tokens > self.max_tokens:
                if current_chunk_sentences:
                    chunks.append(" ".join(current_chunk_sentences))
                    current_chunk_sentences = []
                    current_chunk_tokens = 0
                
                # Forcibly truncate the single long sentence by token limits
                encoded_sentence = self.tokenizer.encode(sentence)
                for start_tok in range(0, len(encoded_sentence), self.max_tokens - self.overlap_tokens):
                    end_tok = start_tok + self.max_tokens
                    sub_str = self.tokenizer.decode(encoded_sentence[start_tok:end_tok])
                    chunks.append(sub_str)
                i += 1
                continue

            # If adding this sentence exceeds the maximum token allowance
            if current_chunk_tokens + sentence_tokens > self.max_tokens:
                chunks.append(" ".join(current_chunk_sentences))
                
                # Rewind back to create the mathematical overlap
                # Accumulate sentences backwards until we hit the desired overlap_tokens limit
                overlap_sentences = []
                overlap_token_count = 0
                
                for prev_sentence in reversed(current_chunk_sentences):
                    prev_tokens = len(self.tokenizer.encode(prev_sentence))
                    if overlap_token_count + prev_tokens > self.overlap_tokens:
                        break
                    overlap_sentences.insert(0, prev_sentence)
                    overlap_token_count += prev_tokens
                
                current_chunk_sentences = overlap_sentences
                current_chunk_tokens = overlap_token_count

            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sentence_tokens
            i += 1

        # Append residual trailing sentences
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks

    # ---------------------------------------------------------
    # Metadata helper
    # ---------------------------------------------------------

    @staticmethod
    def merge_metadata(
        document_metadata: Dict,
        section_metadata: Optional[Dict] = None,
    ) -> Dict:
        metadata = dict(document_metadata)
        if section_metadata:
            metadata.update(section_metadata)
        return metadata

    # ---------------------------------------------------------
    # Chunk builder
    # ---------------------------------------------------------

    def create_chunk(
        self,
        document: MedicalDocument,
        section: DocumentSection,
        chunk_text: str,
        chunk_index: int,
    ) -> MedicalChunk:

        metadata = self.merge_metadata(
            document.metadata,
            section.metadata,
        )
        metadata["section_id"] = section.section_id
        metadata["section_title"] = section.title
        metadata["section_type"] = section.section_type

        return MedicalChunk(
            chunk_id=self.generate_chunk_id(
                document.doc_id,
                chunk_index,
            ),
            document_id=document.doc_id,
            chunk_index=chunk_index,
            source=document.source,
            disease=document.disease,
            disease_id=document.disease_id,
            title=document.title,
            section=section.title,
            text=chunk_text,
            keywords=document.keywords.copy(),
            metadata=metadata,
            embedding=None,
        )