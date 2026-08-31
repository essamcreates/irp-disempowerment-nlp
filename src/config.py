"""Central configuration for reproducible experiments."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"
ERROR_ANALYSIS_DIR = OUTPUTS_DIR / "error_analysis"
LOGS_DIR = PROJECT_ROOT / "logs"

DATASET_ID = "lmsys/lmsys-chat-1m"
DATASET_SPLIT = "train"

# Pin the dataset to the exact Hugging Face revision used for the experiments.
# This prevents later changes to the repository from altering the input data.
DATASET_REVISION = "200748d9d3cddcc9d782887541057aca0b18c5da"

RANDOM_SEED = 42

# Frozen relevance-filter version.
# Increment this if the filtering rules are changed in future.
FILTER_VERSION = "relevance_v1"

# Filtering and pilot sampling
PILOT_SCAN_LIMIT = 5000
PILOT_SAMPLE_SIZE = 100

# Basic text quality thresholds.
MIN_USER_CHARS = 20
MIN_ASSISTANT_CHARS = 20
MAX_USER_CHARS = 5000
MAX_ASSISTANT_CHARS = 10000


# Keep the first inspection intentionally small.
INSPECTION_SAMPLE_SIZE = 25
PREVIEW_CHARS = 300

for directory in [
    RAW_DIR,
    PROCESSED_DIR,
    SAMPLES_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    ERROR_ANALYSIS_DIR,
    LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
