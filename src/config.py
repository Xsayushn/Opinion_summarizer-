"""
Configuration settings for Opinion Summarizer Framework.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEEDBACK_DIR = DATA_DIR / "feedback"
CHROMA_DIR = DATA_DIR / "chroma_db"
SQLITE_DB_PATH = DATA_DIR / "opinions.db"

# Create necessary directories
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, FEEDBACK_DIR, CHROMA_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Default Canonical Aspect Taxonomies
HOTEL_ASPECT_TAXONOMY = {
    "Cleanliness": ["clean", "dirty", "hygiene", "smell", "dust", "tidy", "stain", "sheets", "bathroom"],
    "Service": ["staff", "reception", "desk", "check-in", "checkout", "helpful", "rude", "hospitality", "attendant"],
    "Location": ["location", "subway", "distance", "metro", "downtown", "airport", "walk", "central", "neighborhood"],
    "Rooms": ["room", "bed", "mattress", "shower", "view", "quiet", "noise", "space", "spacious", "cramped", "balcony"],
    "Amenities": ["pool", "gym", "wifi", "internet", "parking", "breakfast", "elevator", "ac", "air condition", "spa"],
    "Value": ["price", "cost", "expensive", "cheap", "worth", "deal", "value", "overpriced", "affordable"]
}

ELECTRONICS_ASPECT_TAXONOMY = {
    "Battery": ["battery", "backup", "drain", "charge", "charging", "charger", "mah", "sot", "battery life", "screen-on-time", "screen on time"],
    "Display": ["screen", "display", "amoled", "oled", "brightness", "refresh rate", "resolution", "colors", "panel"],
    "Performance": ["speed", "lag", "fast", "processor", "chipset", "ram", "multitasking", "gaming", "smooth", "heating", "warm"],
    "Camera": ["camera", "photo", "lens", "video", "sensor", "low light", "night mode", "portrait", "zoom", "megapixels"],
    "Build & Design": ["build", "quality", "design", "durability", "feel", "premium", "plastic", "metal", "weight", "ergonomics"],
    "Value for Money": ["price", "cost", "value", "expensive", "affordable", "cheap", "worth"]
}

# Ingestion / Preprocessing
MINHASH_PERMUTATIONS = 128
DEDUP_JACCARD_THRESHOLD = 0.85

# Retrieval Hyperparameters
DEFAULT_CANDIDATE_POOL_K = 100
DEFAULT_CONTEXT_EXEMPLAR_K = 40
MINORITY_QUOTA_RATIO = 0.15  # Reserve at least 15% slots for significant minority polarity

# Verification / NLI Thresholds
NLI_ENTAILMENT_THRESHOLD = 0.70
NLI_CONTRADICTION_THRESHOLD = 0.30

# Quantifier Threshold Boundaries
QUANTIFIER_BOUNDARIES = {
    "universal": {"terms": ["all", "universally", "everyone", "every"], "min": 0.90, "max": 1.00},
    "majority": {"terms": ["most", "majority", "largely", "mainly", "mostly"], "min": 0.50, "max": 0.90},
    "moderate": {"terms": ["many", "several", "a significant number"], "min": 0.30, "max": 0.60},
    "minority": {"terms": ["some", "minority", "a portion"], "min": 0.10, "max": 0.50},
    "rare": {"terms": ["few", "rarely", "isolated", "hardly any", "scant"], "min": 0.00, "max": 0.15}
}

# Model Settings
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_NLI_MODEL = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
DEFAULT_ABSA_MODEL = os.getenv("ABSA_MODEL", "yangheng/deberta-v3-base-absa-v1.1")

# LLM Configurations
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # 'gemini', 'groq', or 'ollama'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
