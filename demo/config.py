from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
RAW_DIR = DEMO_DIR / "raw_data"
OUTPUT_DIR = DEMO_DIR / "outputs"
DB_PATH = DEMO_DIR / "agentic_demo.sqlite"
VIEWERS_CSV = RAW_DIR / "viewers.csv"
COMPUTE_NODES_CSV = RAW_DIR / "compute_nodes.csv"
POLICY_TXT = DEMO_DIR / "policy.txt"
DEFAULT_INTERVAL_SECONDS = 30
