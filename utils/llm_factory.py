from config import settings
from utils.model_loader import ModelLoader
from tests.fakes.fake_llm import FakeLLM


def get_llm():
    """
    Environment-based LLM selection with optional fallback.
    """

    # -------------------------
    # Test environment → FakeLLM
    # -------------------------
    if settings.app_env == "test":
        return FakeLLM()

    # -------------------------
    # Try primary model
    # -------------------------
    try:
        loader = ModelLoader(
            provider=settings.llm_provider,
            model=settings.llm_model,
        )
        llm = loader.load_llm()
        print(f"[LLM] Using {settings.llm_provider}:{settings.llm_model}")
        return llm

    except Exception as primary_error:
        print(f"[LLM] Primary model failed: {primary_error}")

    # -------------------------
    # Try fallback model (if defined)
    # -------------------------
    if settings.llm_fallback_provider and settings.llm_fallback_model:
        try:
            loader = ModelLoader(
                provider=settings.llm_fallback_provider,
                model=settings.llm_fallback_model,
            )
            llm = loader.load_llm()
            print(
                f"[LLM] Fallback to "
                f"{settings.llm_fallback_provider}:{settings.llm_fallback_model}"
            )
            return llm

        except Exception as fallback_error:
            print(f"[LLM] Fallback model failed: {fallback_error}")

    # -------------------------
    # Total failure
    # -------------------------
    raise RuntimeError("No usable LLM available")