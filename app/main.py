from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from diagnosis import router as diagnosis_router

app = FastAPI(
    title="Multi-Agent Disease Diagnosis API",
    version="1.0.0",
    description="LangGraph-based multi-agent clinical reasoning system",
)

# CORS (safe defaults for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(diagnosis_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}