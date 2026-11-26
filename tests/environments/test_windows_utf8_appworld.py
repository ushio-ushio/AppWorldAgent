"""Opt-in local AppWorld regression for Windows log encoding; no model calls."""
from __future__ import annotations

import os
import sys
from uuid import uuid4

import pytest


@pytest.mark.skipif(os.name != "nt" or os.getenv("RUN_APPWORLD_UTF8_INTEGRATION") != "1",
                    reason="Enable explicitly on Windows with PYTHONUTF8=1")
def test_non_gbk_output_does_not_poison_later_api_calls():
    assert sys.flags.utf8_mode == 1, "PYTHONUTF8=1 must be set before Python starts"
    from appworld import AppWorld
    from models.siwc_responses import _wall_seconds

    started = _wall_seconds()
    with AppWorld(task_id="82e2fac_1", experiment_name=f"agentlab_utf8_regression_{uuid4().hex[:8]}") as world:
        first = world.execute("print('🏀')")
        second = world.execute("print(apis.api_docs.show_api_descriptions(app_name='spotify'))")
        third = world.execute("print(apis.api_docs.show_api_descriptions(app_name='spotify'))")
    assert "🏀" in first
    assert not second.startswith("Execution failed.")
    assert not third.startswith("Execution failed.")
    assert _wall_seconds() > started
