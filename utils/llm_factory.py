from utils.config_backend import backend_settings
from utils.model_loader import ModelLoader
from tests.fakes.fake_llm import FakeLLM


def get_llm():
    """
    Environment-based LLM selection with optional fallback.
    """

    # -------------------------
    # Test environment → FakeLLM
    # -------------------------
    if backend_settings.app_env == "test":
        return FakeLLM()

    # -------------------------
    # Try primary model
    # -------------------------
    try:
        loader = ModelLoader(
            provider=backend_settings.llm_provider,
            model=backend_settings.llm_model,
        )
        llm = loader.load_llm()
        print(f"[LLM] Using {backend_settings.llm_provider}:{backend_settings.llm_model}")
        return llm

    except Exception as primary_error:
        print(f"[LLM] Primary model failed: {primary_error}")

    # -------------------------
    # Try fallback model (if defined)
    # -------------------------
    if backend_settings.llm_fallback_provider and backend_settings.llm_fallback_model:
        try:
            loader = ModelLoader(
                provider=backend_settings.llm_fallback_provider,
                model=backend_settings.llm_fallback_model,
            )
            llm = loader.load_llm()
            print(
                f"[LLM] Fallback to "
                f"{backend_settings.llm_fallback_provider}:{backend_settings.llm_fallback_model}"
            )
            return llm

        except Exception as fallback_error:
            print(f"[LLM] Fallback model failed: {fallback_error}")

    # -------------------------
    # Total failure
    # -------------------------
    raise RuntimeError("No usable LLM available")