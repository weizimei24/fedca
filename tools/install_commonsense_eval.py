"""Install the pinned, isolated harness runtime without upgrading PyTorch."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--index-url", "https://pypi.org/simple",
                           "--disable-pip-version-check", "--target", str(ROOT / ".runtime/lm_eval"),
                           "--no-deps", "-r", str(ROOT / 'requirements' / 'requirements-commonsense-eval.txt')])
    sys.path.insert(0, str(ROOT))
    from utils.commonsense_eval import ensure_runtime
    ensure_runtime()
    print("Commonsense lm-evaluation-harness runtime ready; existing torch was not replaced.")
