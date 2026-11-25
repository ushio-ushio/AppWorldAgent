"""Run the frozen Train-15 once in bounded SIWC waves."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentlab.evaluation.report import parse_report_counts


CONFIG = json.loads(
    (ROOT / "configs" / "v3_shadow_train15.json").read_text(encoding="utf-8")
)
TASKS = tuple(CONFIG["tasks"])


def _task_command(task_id: str, experiment: str, task_dir: Path) -> list[str]:
    command = [
        sys.executable, "-u", str(ROOT / "scripts" / "run_siwc_agent.py"),
        "--task-id", task_id, "--model", CONFIG["model"],
        "--context-mode", CONFIG["context_mode"],
    ]
    if CONFIG["optimized_unified"]:
        command.append("--optimized-unified")
    command.append("--schema-aware-retrieval" if CONFIG["schema_aware_retrieval"]
                   else "--no-schema-aware-retrieval")
    command.append("--programmatic-execution" if CONFIG["programmatic_execution"]
                   else "--no-programmatic-execution")
    if CONFIG["architecture"] == "plan_guided_react":
        command.append("--plan-guided-react")
    command.extend([
        "--assurance-mode", CONFIG["execution_assurance_mode"],
        "--max-steps", str(CONFIG["max_executor_steps"]),
        "--siwc-connect-timeout", str(CONFIG["connect_timeout_seconds"]),
        "--siwc-read-timeout", str(CONFIG["read_timeout_seconds"]),
        "--experiment", experiment,
        "--task-dir", str(task_dir),
    ])
    return command


def _transport_diagnostics(trace_path: Path, worker_log: Path) -> list[dict]:
    diagnostics = []
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") == "agent_error" and event.get("adapter_diagnostic"):
                diagnostics.append(event["adapter_diagnostic"])
    if worker_log.exists():
        content = worker_log.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"SIWC transport failed:\s*([^\r\n(]+)\s*\(([^)]+)\)", content)
        if match:
            diagnostics.append({"phase": match.group(1).strip(),
                                "error_class": match.group(2).strip(),
                                "source": "worker_log"})
    return diagnostics


def _task_result(task_id: str, wave: int, exit_code: int, task_dir: Path) -> dict:
    summary_path = task_dir / "summary.json"
    report_path = task_dir / "evaluation_report.md"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    passed, total = parse_report_counts(report)
    assurance = summary.get("execution_assurance") or {}
    goals = assurance.get("goals") or {}
    diagnostics = _transport_diagnostics(task_dir / "trace.jsonl", task_dir / "worker.log")
    worker_error = None
    if exit_code != 0:
        worker_error = ("SIWC transport failed: " + diagnostics[-1]["phase"] +
                        " (" + diagnostics[-1]["error_class"] + ")"
                        if diagnostics else "worker exited before a complete evaluation")
    return {
        "task_id": task_id,
        "wave": wave,
        "exit_code": exit_code,
        "passed": passed,
        "total": total,
        "steps": summary.get("steps"),
        "llm_requests": summary.get("llm_request_count"),
        "stop_reason": summary.get("stop_reason"),
        "worker_error": worker_error,
        "usage": summary.get("total_usage"),
        "api_boundary_coverage": summary.get("api_boundary_coverage"),
        "assurance_events": len(assurance.get("events") or []),
        "goal_status_counts": {
            status: sum(goal.get("status") == status for goal in goals.values())
            for status in ("PENDING", "VERIFIED", "FAILED", "UNKNOWN")
        },
        "transport_diagnostics": diagnostics,
        "valid_full_run": (passed is not None and total is not None
                           and summary.get("stop_reason") == "appworld_task_completed"
                           and not diagnostics and exit_code == 0),
    }


def main() -> None:
    from appworld import load_task_ids
    from agentlab.models.siwc_responses import ChatGPTResponsesAdapter

    if len(TASKS) != 15 or not set(TASKS) <= set(load_task_ids("train")):
        raise RuntimeError("Invalid frozen Train-15 task list")

    # Verify the configured model before creating task workers.
    adapter = ChatGPTResponsesAdapter(CONFIG["model"])
    adapter.close()

    experiment = CONFIG["experiment_prefix"] + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = ROOT / "runs" / experiment
    run_root.mkdir(parents=True)
    waves = [TASKS[i:i + CONFIG["max_parallel"]]
             for i in range(0, len(TASKS), CONFIG["max_parallel"])]
    effective_config = {
        **CONFIG,
        "tasks": TASKS,
        "execution_assurance": {
            "mode": CONFIG["execution_assurance_mode"],
            "enabled": True,
            "api_boundary": "AppWorld in-process Requester.request",
        },
        "waves": waves,
        "per_task_attempts": 1,
    }
    (run_root / "effective_config.json").write_text(
        json.dumps(effective_config, indent=2), encoding="utf-8"
    )

    environment = {
        **os.environ,
        "PYTHONUTF8": "1",
        "APPWORLD_ROOT": str(ROOT / "vendor" / "appworld"),
        "PYTHONPATH": os.pathsep.join((
        str(ROOT / "src"), str(ROOT / "vendor" / "appworld" / "src")
        )),
    }
    results: list[dict] = []
    transport_failures = 0

    for wave_number, wave in enumerate(waves, start=1):
        active = {}
        print("WAVE", wave_number, "START", ",".join(wave), flush=True)
        for task_id in wave:
            task_dir = run_root / task_id
            task_dir.mkdir()
            log = (task_dir / "worker.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                _task_command(task_id, experiment, task_dir),
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            )
            active[task_id] = (process, log, time.monotonic())

        while active:
            for task_id, (process, log, started) in list(active.items()):
                exit_code = process.poll()
                if exit_code is None and time.monotonic() - started > 7200:
                    process.terminate()
                    exit_code = process.wait()
                if exit_code is None:
                    continue

                log.close()
                del active[task_id]
                task_dir = run_root / task_id
                result = _task_result(task_id, wave_number, exit_code, task_dir)
                results.append(result)
                transport_failures += bool(result["transport_diagnostics"])
                (task_dir / "result.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
                (run_root / "progress.json").write_text(
                    json.dumps(results, indent=2), encoding="utf-8"
                )
                score = f"{result['passed']}/{result['total']}"
                print("DONE", task_id, "exit", exit_code,
                      "score", score, "steps", result["steps"], flush=True)
            if active:
                time.sleep(2)

        if transport_failures >= 2:
            print("STOPPED after repeated SIWC transport failures", flush=True)
            break

    attempted = {row["task_id"] for row in results}
    valid = {row["task_id"] for row in results if row["valid_full_run"]}
    if valid == set(TASKS):
        run_status = "completed_all_tasks"
    elif attempted == set(TASKS):
        run_status = "completed_with_worker_failures"
    elif transport_failures >= 2:
        run_status = "stopped_on_transport_failures"
    else:
        run_status = "incomplete"
    status = {
        "status": run_status,
        "attempted": sorted(attempted),
        "unstarted": [task_id for task_id in TASKS if task_id not in attempted],
        "valid_full_runs": sorted(valid),
        "not_scored": sorted(row["task_id"] for row in results if row["total"] is None),
        "invalid_or_interrupted": sorted(row["task_id"] for row in results
                                          if row["total"] is not None and not row["valid_full_run"]),
    }
    (run_root / "batch_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print("RUN_ROOT", run_root, "STATUS", status["status"], flush=True)


if __name__ == "__main__":
    main()
