from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def build_rag_chain(vectorstore, llm, system_prompt: str = None):
    """
    Build a simple RetrievalQA chain that uses a retriever (FAISS, Chroma, etc.)
    and a given LLM for reasoning over retrieved disease information.

    :param vectorstore: a FAISS or Chroma vector store instance
    :param llm: loaded language model instance (e.g., from ModelLoader)
    :param system_prompt: optional system prompt string
    :return: RetrievalQA chain
    """

    # Basic prompt for disease reasoning
    template = (
        "{system_prompt}\n\n"
        "You are a clinical assistant reasoning about possible diseases based on patient symptoms.\n"
        "Use the following context about diseases to support your reasoning.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Provide your answer as structured JSON with ranked diseases and reasoning."
    )

    prompt = PromptTemplate(
        input_variables=["context", "question", "system_prompt"],
        template=template
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt}
    )

    return rag_chain