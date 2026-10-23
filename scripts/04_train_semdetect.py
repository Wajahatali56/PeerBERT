"""
04_train_semdetect.py
Train PeerBERT (ModernBERT-base) on the Sem-Detect manifests. Training is blocked unless
outputs/audit/<ds>/audit_status.json exists and is not FAIL (scripts/02_prepare_splits.py).

  python scripts/04_train_semdetect.py --variant auto        # all seeds, variant selected by the ablation
  python scripts/04_train_semdetect.py --seeds 42            # one seed, base configuration
  python scripts/04_train_semdetect.py --smoke               # tiny whole-paper subset, 1 epoch (pipeline check)
Restart-safe: rerun the same command to resume (last_state.pt) or skip finished seeds (done.json).
A finished or interrupted run is reused only if its configuration and data are unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.train_cli import main  # noqa: E402

if __name__ == "__main__":
    main("semdetect", __doc__)
