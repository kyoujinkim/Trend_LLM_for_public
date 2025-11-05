"""
Configuration file for FACTOR-LLM
"""
import os
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "news_data_by_date"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Data Settings
CSVS = [
    "data_1",
    "data_2",
    "data_4"
]

# Text Preprocessing Settings
BATCH_SIZE=8

# Text Processing Settings
MIN_KEYWORD_LENGTH = 2
MAX_KEYWORD_LENGTH = 20
TOP_N_KEYWORDS = 50

# Time Series Settings
WINDOW_SIZE = 7  # days for moving average
THRESHOLD_EMERGENCE = 0.7  # threshold for emerging trends
THRESHOLD_DECLINE = 0.3  # threshold for declining trends

# Prediction Settings
PREDICTION_HORIZON_DAYS = 365  # 1 year prediction
CONFIDENCE_LEVEL = 0.95

# LLM Settings
LLM_PROVIDER = "anthropic"  # or "openai"
LLM_MODEL = "claude-3-5-sonnet-20241022"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 4000

# API Keys (load from environment)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Report Settings
REPORT_FORMAT = "markdown"  # or "html", "pdf"
INCLUDE_VISUALIZATIONS = True

# Memory Efficiency Settings
USE_MEMORY_EFFICIENT_MODE = True  # Enable chunked processing
CHUNK_SIZE = 1000  # Number of rows per chunk (auto-configured if None)
AUTO_CONFIGURE_MEMORY = True  # Automatically configure based on system memory
MEMORY_THRESHOLD_MB = 1000  # Memory threshold for warnings (MB)
OPTIMIZE_DATAFRAMES = True  # Optimize DataFrame memory usage

# Processing Mode
PROCESSING_MODE = "chunked"  # "full" (load all), "chunked", or "sample"
SAMPLE_SIZE = 10000  # Number of rows to sample if PROCESSING_MODE = "sample"

# Garbage Collection
AGGRESSIVE_GC = True  # Force garbage collection between operations
