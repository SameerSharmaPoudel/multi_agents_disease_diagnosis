"""

Shared schemas used throughout the disease diagnosis system.

These schemas are intentionally source-independent and form the
contract between:

    Parsers
        ↓
    Validators
        ↓
    Enrichers
        ↓
    Chunkers
        ↓
    Retrievers
        ↓
    Diagnosis Agents

Try not to modify these once the ingestion pipeline is implemented.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================
# KNOWLEDGE BASE
# ============================================================

@dataclass
class DocumentSection:
    """
    Represents one logical section inside a medical document.

    Examples
    --------
    MedlinePlus:
        Summary
        Symptoms
        Diagnosis
        Treatment

    StatPearls:
        Introduction
        Etiology
        History and Physical
        Evaluation
        Treatment
        Differential Diagnosis

    MedQuAD:
        Question
        Answer
    """

    section_id: str

    title: str

    section_type: str 

    text: str

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MedicalDocument:
    """
    Unified representation of an entire medical document.

    Produced by
    -----------
    - MedlinePlus parser
    - MedQuAD parser
    - StatPearls parser

    Consumed by
    -----------
    - Validator
    - KeywordExtractor
    - Chunker
    """

    # ---------- Identity ----------

    doc_id: str

    source: str

    # ---------- Medical ----------

    disease: str

    disease_id: Optional[str] = None

    title: str = ""

    # ---------- Content ----------

    sections: List[DocumentSection] = field(default_factory=list)

    # ---------- Retrieval ----------

    keywords: List[str] = field(default_factory=list)

    # ---------- Extra ----------

    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# CHUNKS
# ============================================================

@dataclass
class MedicalChunk:
    """
    Smallest retrieval unit.

    Produced by
    -----------
    Chunker

    Consumed by
    -----------
    Dense Retrieval
    BM25
    Metadata Filtering
    Reranker
    """

    # ---------- Identity ----------

    chunk_id: str

    document_id: str

    chunk_index: int

    # ---------- Provenance ----------

    source: str

    disease: str

    disease_id: Optional[str] = None

    title: str = ""

    section: str = ""

    # ---------- Content ----------

    text: str = ""

    # ---------- Retrieval ----------

    keywords: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    embedding: Optional[List[float]] = None


# ============================================================
# PATIENT UNDERSTANDING
# ============================================================

@dataclass
class Symptom:
    """
    Canonical patient symptom.
    """

    raw_name: str

    canonical_name: Optional[str] = None

    mesh_id: Optional[str] = None

    value: Optional[str] = None

    severity: Optional[str] = None

    duration: Optional[str] = None

    onset: Optional[str] = None

    negated: bool = False

    confidence: float = 1.0


@dataclass
class PatientState:
    """
    Structured understanding of the patient.

    Produced by
    -----------
    Patient Understanding Agent

    Consumed by
    -----------
    Retrieval Planner
    Diagnosis Agent
    Reflection Agent
    """

    symptoms: List[Symptom] = field(default_factory=list)

    age: Optional[int] = None

    sex: Optional[str] = None

    medications: List[str] = field(default_factory=list)

    medical_history: List[str] = field(default_factory=list)

    family_history: List[str] = field(default_factory=list)

    allergies: List[str] = field(default_factory=list)

    lifestyle: List[str] = field(default_factory=list)

    risk_factors: List[str] = field(default_factory=list)

    free_text: str = ""


# ============================================================
# RETRIEVAL
# ============================================================

@dataclass
class RetrievalPlan:
    """
    Output of the Retrieval Planner.

    Every retriever receives exactly the same plan.
    """

    dense_query: str

    bm25_query: List[str]

    graph_query: List[str]

    metadata_filters: Dict[str, Any] = field(default_factory=dict)

    top_k: int = 10


@dataclass
class RetrievalResult:
    """
    Returned by every retrieval method.
    """

    chunk_id: Optional[str]

    document_id: Optional[str]

    disease: str

    disease_id: Optional[str]

    source: str

    section: Optional[str]

    text: Optional[str]

    retrieval_method: str

    retrieval_score: float

    rerank_score: Optional[float] = None

    final_score: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# FUSION
# ============================================================

@dataclass
class EvidenceBundle:
    """
    Aggregated retrieval evidence.

    Produced by
    -----------
    Hybrid Retrieval Layer

    Consumed by
    -----------
    Diagnosis Agent
    Reflection Agent
    """

    dense_results: List[RetrievalResult] = field(default_factory=list)

    bm25_results: List[RetrievalResult] = field(default_factory=list)

    graph_results: List[RetrievalResult] = field(default_factory=list)

    fused_results: List[RetrievalResult] = field(default_factory=list)


# ============================================================
# DIAGNOSIS
# ============================================================

@dataclass
class DiagnosisCandidate:
    """
    One possible diagnosis.
    """

    disease: str

    disease_id: Optional[str]

    confidence: float

    supporting_evidence: List[str] = field(default_factory=list)

    contradicting_evidence: List[str] = field(default_factory=list)

    rationale: str = ""


@dataclass
class DiagnosisResult:
    """
    Final output of the Diagnosis Agent.
    """

    candidates: List[DiagnosisCandidate] = field(default_factory=list)

    recommended_questions: List[str] = field(default_factory=list)

    recommended_tests: List[str] = field(default_factory=list)

    confidence: float = 0.0

    diagnosis_complete: bool = False