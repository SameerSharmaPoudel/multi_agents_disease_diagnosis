# rag/build_symptom_index.py
import os
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


KNOWN_LABELS = {"prognosis", "disease", "diagnosis", "label", "target"}


def detect_label_column(df: pd.DataFrame, user_label: Optional[str]) -> str:
    if user_label:
        if user_label not in df.columns:
            raise ValueError(f"Label column '{user_label}' not found. Available: {list(df.columns)[:20]}...")
        return user_label
    for c in df.columns:
        if c.strip().lower() in KNOWN_LABELS:
            return c
    # fallback: assume last column is the label
    return df.columns[-1]


def coerce_binary(x) -> int:
    try:
        return 1 if float(x) > 0 else 0
    except Exception:
        # booleans/strings
        s = str(x).strip().lower()
        return 1 if s in {"1", "true", "yes", "y"} else 0


def build_documents(df: pd.DataFrame, label_col: str) -> Tuple[List[Document], List[Dict]]:
    symptom_cols = [c for c in df.columns if c != label_col]
    docs, metadatas = [], []

    for idx, row in df.iterrows():
        bin_map = {c: coerce_binary(row[c]) for c in symptom_cols}
        positive = [c for c, v in bin_map.items() if v == 1]

        disease = str(row[label_col]).strip()
        # Create a compact text representation for embedding
        text = (
            f"Symptoms: {', '.join(positive) if positive else 'none'}\n"
            f"Disease: {disease}"
        )
        meta = {
            "row_id": int(idx),
            "disease": disease,
            "positive_symptoms": positive,
            "label_col": label_col,
        }
        docs.append(Document(page_content=text, metadata=meta))
        metadatas.append(meta)

    return docs, metadatas


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from symptom one-hot CSV")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV (one-hot symptoms + label).")
    parser.add_argument("--out_dir", type=str, default="./indices/symptoms_faiss", help="Output directory for the index.")
    parser.add_argument("--label_col", type=str, default=None, help="Optional: explicit label column name.")
    parser.add_argument("--hf_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HuggingFace embedding model.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    label_col = detect_label_column(df, args.label_col)
    print(f"[build_symptom_index] Using label column: {label_col}")

    docs, metadatas = build_documents(df, label_col)

    embeddings = HuggingFaceEmbeddings(model_name=args.hf_model)
    vs = FAISS.from_documents(docs, embeddings)

    # Save FAISS
    vs.save_local(args.out_dir)
    # Save a JSON metadata snapshot (not required, but handy)
    with open(Path(args.out_dir) / "metadata_preview.json", "w", encoding="utf-8") as f:
        json.dump(metadatas[:20], f, indent=2)

    print(f"[build_symptom_index] Saved FAISS index to {args.out_dir} (docs: {len(docs)})")


if __name__ == "__main__":
    main()
