"""Read summary counts from an AppWorld evaluator report."""
from __future__ import annotations

import re


_PASSED = re.compile(r"Num Passed Tests\s*:\s*(\d+)")
_TOTAL = re.compile(r"Num Total\s+Tests\s*:\s*(\d+)")


def parse_report_counts(report: str) -> tuple[int | None, int | None]:
    """Return (passed, total), or None for either count if the report is incomplete."""
    passed = _PASSED.search(report)
    total = _TOTAL.search(report)
    return (int(passed.group(1)) if passed else None,
            int(total.group(1)) if total else None)
