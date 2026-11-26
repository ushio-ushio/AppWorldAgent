import json
import threading
import time
from unittest.mock import patch

from scripts.run_pipeline import PipelineRunner


def _report(passed=1, total=1):
    return f"Num Passed Tests : {passed}\nNum Total Tests : {total}\n"


def _write_inference_artifacts(task_dir, *, steps=3, tokens=1200, py_error=False):
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "summary.json").write_text(json.dumps({
        "steps": steps, "stop_reason": "appworld_task_completed",
        "total_usage": {"input_tokens": tokens, "output_tokens": 50,
                        "input_cache_hit_tokens": 100},
        "async_extraction": {"input_tokens": 200, "output_tokens": 10},
    }), encoding="utf-8")
    event = {"event": "tool_call", "tool_name": "execute_python",
             "tool_result": {"ok": not py_error,
                             **({"execution_error": "mock error",
                                 "error_envelope": {"error_type": "mock_error"}}
                                if py_error else {})}}
    (task_dir / "trace.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")


def _fake_infer(calls, *, fail_task=None, second_started=None):
    def infer(task_id, task_dir):
        calls.append(task_id)
        if task_id == fail_task:
            raise RuntimeError("mock inference failure")
        if task_id == "task-2" and second_started:
            second_started.set()
        _write_inference_artifacts(task_dir, py_error=(task_id == "task-3"))
        return {"task_id": task_id, "task_dir": task_dir,
                "exit_code": 0, "inference_seconds": 2.0}
    return infer


def test_inference_overlaps_background_evaluation_and_persists_results(tmp_path):
    run_root = tmp_path / "pipeline"
    infer_calls = []
    evaluation_started = threading.Event()
    allow_evaluation = threading.Event()
    second_inference_started = threading.Event()

    def evaluate(task_id, _experiment):
        if task_id == "task-1":
            evaluation_started.set()
            assert allow_evaluation.wait(5)
        return _report()

    runner = PipelineRunner(
        run_root, "exp", {}, inference_workers=1, eval_workers=1,
        infer_fn=_fake_infer(infer_calls, second_started=second_inference_started),
        evaluate_fn=evaluate,
    )
    outcome = {}
    thread = threading.Thread(target=lambda: outcome.setdefault(
        "summary", runner.run(["task-1", "task-2", "task-3"])))
    thread.start()
    assert evaluation_started.wait(5)
    assert second_inference_started.wait(5), "inference should continue while evaluation is blocked"
    allow_evaluation.set()
    thread.join(10)

    assert not thread.is_alive()
    assert outcome["summary"]["status"] == "completed"
    assert infer_calls == ["task-1", "task-2", "task-3"]
    rows = [json.loads(line) for line in (run_root / "pipeline_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert (run_root / "pipeline_summary.json").exists()
    assert (run_root / "pipeline_summary.md").exists()
    assert outcome["summary"]["input_tokens_total"] == 4200


def test_resume_skips_tasks_with_complete_official_scores(tmp_path):
    run_root = tmp_path / "resume"
    for task_id in ("task-1", "task-2"):
        completed = run_root / task_id / "attempt_001"
        _write_inference_artifacts(completed)
        (completed / "evaluation_report.md").write_text(_report(), encoding="utf-8")
    calls, evaluated = [], []
    runner = PipelineRunner(
        run_root, "exp", {}, inference_workers=1,
        infer_fn=_fake_infer(calls),
        evaluate_fn=lambda task, _exp: (evaluated.append(task) or _report()),
    )
    summary = runner.run(["task-1", "task-2", "task-3"], resume=True)

    assert calls == ["task-3"]
    assert evaluated == ["task-3"]
    assert summary["scored_tasks"] == 3


def test_inference_error_isolated_and_later_tasks_finish(tmp_path):
    calls, evaluated = [], []
    runner = PipelineRunner(
        tmp_path / "errors", "exp", {}, inference_workers=1,
        infer_fn=_fake_infer(calls, fail_task="task-2"),
        evaluate_fn=lambda task, _exp: (evaluated.append(task) or _report()),
    )
    summary = runner.run(["task-1", "task-2", "task-3"])
    rows = [json.loads(line) for line in (tmp_path / "errors" / "pipeline_results.jsonl").read_text(encoding="utf-8").splitlines()]

    assert calls == ["task-1", "task-2", "task-3"]
    assert evaluated == ["task-1", "task-3"]
    assert next(row for row in rows if row["task_id"] == "task-2")["status"] == "error"
    assert summary["status"] == "completed_with_errors"


def test_compare_with_baseline_records_and_renders_deltas(tmp_path):
    baseline = tmp_path / "baseline"
    old = baseline / "task-1"
    old.mkdir(parents=True)
    (old / "summary.json").write_text(json.dumps({
        "steps": 8, "total_usage": {"input_tokens": 3000},
    }), encoding="utf-8")
    (old / "evaluation_report.md").write_text(_report(1, 2), encoding="utf-8")
    run_root = tmp_path / "current"

    def infer(task, task_dir):
        _write_inference_artifacts(task_dir, steps=5, tokens=2000)
        return {"exit_code": 0, "inference_seconds": 1.0}

    runner = PipelineRunner(run_root, "current-exp", {}, compare_with=baseline,
                            infer_fn=infer, evaluate_fn=lambda *_: _report(2, 2))
    summary = runner.run(["task-1"])
    record = summary["tasks"][0]
    assert record["comparison"] == {
        "baseline_score": "1/2", "score_delta_passed": 1,
        "baseline_steps": 8, "steps_delta": -3,
        "baseline_input_tokens": 3000, "input_tokens_delta": -800,
    }
    markdown = (run_root / "pipeline_summary.md").read_text(encoding="utf-8")
    assert "Δscore +1, steps -3, tok -800" in markdown


def test_subprocess_worker_reuses_dispatchers_fresh_attempt_directory(tmp_path):
    run_root = tmp_path / "run"
    runner = PipelineRunner(run_root, "exp", {
        "context_mode": "cache_first", "max_steps": 100,
        "connect_timeout": 15, "read_timeout": 120,
        "max_no_progress_steps": 8, "assurance_mode": "reflection_v1",
    }, infer_fn=lambda *_: {})
    task_dir = runner._create_attempt("task-1")

    class Process:
        def wait(self, timeout=None):
            return 0

    with patch("scripts.run_pipeline.subprocess.Popen", return_value=Process()) as popen:
        result = runner._infer_subprocess("task-1", task_dir)

    assert result["exit_code"] == 0
    assert task_dir.is_dir()
    assert (task_dir / "worker.log").exists()
    assert popen.called
