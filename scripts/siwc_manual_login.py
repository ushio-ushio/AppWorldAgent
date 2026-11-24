"""Start SIWC login without launching a browser; paste the URL into Edge."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentlab.models.siwc_auth import login  # noqa: E402


if __name__ == "__main__":
    result = login(timeout_seconds=600, open_browser=False)
    print("SIWC sign-in completed.")
    print("Granted scopes:", ", ".join(result["granted_scopes"]))
