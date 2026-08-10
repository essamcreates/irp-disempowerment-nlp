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

# Use "main" for initial exploration.
# Once the pipeline is stable, pin this to a specific dataset revision/commit.
DATASET_REVISION = "main"

RANDOM_SEED = 42

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
