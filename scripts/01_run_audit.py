"""
01_run_audit.py
Run the leakage audit only (no manifests). Thin wrapper around
`02_prepare_splits.py --audit-only` so the audit logic lives in one place.

  python scripts/01_run_audit.py                    # both datasets
  python scripts/01_run_audit.py --dataset semdetect
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("prepare_splits", HERE / "02_prepare_splits.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.exit(mod.main(["--audit-only"] + sys.argv[1:]))
