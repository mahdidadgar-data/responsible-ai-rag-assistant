from pathlib import Path
import os
from dotenv import load_dotenv


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
REPORTS_DIR = PROJECT_ROOT / "reports"


# Load local environment variables
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)


# Model and vector-store settings
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CHROMA_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    "responsible_ai_rag_multisource_v1",
)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")


# Reproducible default:
# Keep OpenAI generation disabled unless explicitly enabled.
ENABLE_OPENAI_GENERATION = os.getenv(
    "ENABLE_OPENAI_GENERATION",
    "false",
).lower() == "true"


OPENAI_API_KEY_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))