"""Run AppWorld tasks through separate inference and official-evaluation queues."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("APPWORLD_ROOT", str(ROOT / "vendor" / "appworld"))
os.environ.setdefault("APPWORLD_CACHE", str(Path(os.environ["APPWORLD_ROOT"]) / ".tmp" / "agent-cache"))

from evaluation.report import parse_report_counts  # noqa: E402

EXPLORATION_TOOLS = {"search_api", "get_api_schema", "read_observation", "search_history"}


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _attempt_directory(run_root: Path, task_id: str) -> Path:
    task_root = run_root / task_id
    task_root.mkdir(parents=True, exist_ok=True)
    numbers = [int(path.name.removeprefix("attempt_"))
               for path in task_root.glob("attempt_*") if path.name.removeprefix("attempt_").isdigit()]
    return task_root / f"attempt_{max(numbers, default=0) + 1:03d}"


def _scored_artifacts(run_root: Path, task_id: str) -> tuple[Path, str, dict[str, Any]] | None:
    candidates = []
    for report_path in run_root.rglob("evaluation_report.md"):
        if task_id not in report_path.parts:
            continue
        report = report_path.read_text(encoding="utf-8", errors="replace")
        passed, total = parse_report_counts(report)
        if passed is None or total is None:
            continue
        candidates.append((report_path.parent, report, _read_json(report_path.parent / "summary.json", {})))
    return max(candidates, key=lambda row: str(row[0])) if candidates else None


def _trace_metrics(trace_path: Path) -> dict[str, Any]:
    tools: Counter[str] = Counter()
    python_errors = 0
    python_calls = 0
    execution_summaries = 0
    error_envelopes: Counter[str] = Counter()
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") != "tool_call":
                continue
            name = str(event.get("tool_name") or "unknown")
            tools[name] += 1
            result = event.get("tool_result") or {}
            if name == "execute_python":
                python_calls += 1
                python_errors += bool(event.get("execution_error") or result.get("execution_error")
                                      or (result.get("ok") is False and result.get("validation") == "accepted"))
                execution_summaries += isinstance(result.get("execution_summary"), dict)
                envelope = result.get("error_envelope") or {}
                if envelope.get("error_type"):
                    error_envelopes[envelope["error_type"]] += 1
    return {"tool_calls": dict(sorted(tools.items())),
            "total_tool_calls": sum(tools.values()),
            "exploration_calls": sum(tools[name] for name in EXPLORATION_TOOLS),
            "python_calls": python_calls, "python_errors": python_errors,
            "execution_summary_hits": execution_summaries,
            "error_envelope_hits": dict(sorted(error_envelopes.items()))}


def collect_task_metrics(task_id: str, task_dir: Path, report: str,
                         inference_seconds: float = 0.0,
                         evaluation_seconds: float = 0.0) -> dict[str, Any]:
    summary = _read_json(task_dir / "summary.json", {}) or {}
    passed, total = parse_report_counts(report)
    usage = summary.get("total_usage") or {}
    clone = summary.get("async_extraction") or {}
    trace = _trace_metrics(task_dir / "trace.jsonl")
    input_main = int(usage.get("input_tokens", 0) or 0)
    input_clone = int(clone.get("input_tokens", 0) or 0)
    steps = summary.get("steps")
    return {
        "task_id": task_id, "status": "completed", "passed": passed, "total": total,
        "steps": steps, "stop_reason": summary.get("stop_reason"),
        "input_tokens_main": input_main, "input_tokens_clone": input_clone,
        "input_tokens_total": input_main + input_clone,
        "output_tokens": int(usage.get("output_tokens", 0) or 0)
                         + int(clone.get("output_tokens", 0) or 0),
        "cached_tokens_main": int(usage.get("input_cache_hit_tokens", usage.get("cached_tokens", 0)) or 0),
        "python_error_rate": trace["python_errors"] / trace["python_calls"] if trace["python_calls"] else 0.0,
        "exploration_share": trace["exploration_calls"] / trace["total_tool_calls"]
                             if trace["total_tool_calls"] else 0.0,
        "inference_seconds": round(inference_seconds, 3),
        "evaluation_seconds": round(evaluation_seconds, 3),
        "elapsed_seconds": round(inference_seconds + evaluation_seconds, 3),
        **trace,
        "task_dir": str(task_dir),
    }


def _score(record: dict[str, Any]) -> str:
    passed, total = record.get("passed"), record.get("total")
    return f"{passed}/{total}" if passed is not None and total is not None else "ERR"


def _format_tokens(tokens: Any) -> str:
    try:
        return f"{int(tokens) / 1000:.0f}K"
    except (TypeError, ValueError):
        return "n/a"


def _comparison_text(comparison: dict[str, Any] | None) -> str:
    if not comparison:
        return ""
    score, steps, tokens = (comparison.get("score_delta_passed"),
                            comparison.get("steps_delta"),
                            comparison.get("input_tokens_delta"))
    score_text = f"Δscore {score:+d}" if isinstance(score, int) else "Δscore n/a"
    steps_text = f"steps {steps:+d}" if isinstance(steps, int) else "steps n/a"
    tokens_text = f"tok {tokens:+,}" if isinstance(tokens, int) else "tok n/a"
    return f"{score_text}, {steps_text}, {tokens_text}"


def _baseline_task(baseline_dir: Path, task_id: str) -> dict[str, Any] | None:
    artifacts = _scored_artifacts(baseline_dir, task_id)
    if artifacts is None:
        return None
    directory, report, summary = artifacts
    result = collect_task_metrics(task_id, directory, report)
    if summary:
        usage = summary.get("total_usage") or {}
        clone = summary.get("async_extraction") or {}
        result.update(steps=summary.get("steps"),
                      input_tokens_total=int(usage.get("input_tokens", 0) or 0)
                                        + int(clone.get("input_tokens", 0) or 0))
    return result


def _compare(record: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any] | None:
    if baseline is None:
        return None
    current_score = record.get("passed")
    baseline_score = baseline.get("passed")
    return {
        "baseline_score": _score(baseline),
        "score_delta_passed": (current_score - baseline_score
                               if isinstance(current_score, int) and isinstance(baseline_score, int) else None),
        "baseline_steps": baseline.get("steps"),
        "steps_delta": (record["steps"] - baseline["steps"]
                        if isinstance(record.get("steps"), int) and isinstance(baseline.get("steps"), int) else None),
        "baseline_input_tokens": baseline.get("input_tokens_total", 0),
        "input_tokens_delta": record.get("input_tokens_total", 0) - baseline.get("input_tokens_total", 0),
    }


class PipelineRunner:
    """Subprocess inference workers feed a separate official-evaluation queue."""

    def __init__(self, run_root: Path, experiment: str, config: dict[str, Any], *,
                 inference_workers: int = 1, eval_workers: int = 1,
                 compare_with: Path | None = None,
                 infer_fn: Callable[[str, Path], dict[str, Any]] | None = None,
                 evaluate_fn: Callable[[str, str], str] | None = None,
                 preflight: Callable[[], None] | None = None):
        if inference_workers < 1 or eval_workers < 1:
            raise ValueError("Worker counts must be positive.")
        self.run_root, self.experiment, self.config = run_root, experiment, config
        self.inference_workers, self.eval_workers = inference_workers, eval_workers
        self.compare_with = compare_with
        self.infer_fn = infer_fn or self._infer_subprocess
        self.evaluate_fn = evaluate_fn or evaluate_official
        self.preflight = preflight
        self._process_lock = threading.Lock()
        self._active_processes: dict[str, subprocess.Popen] = {}
        self._record_lock = threading.Lock()
        self._evaluation_lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}
        self._current_task_ids: set[str] = set()
        self._current_done: set[str] = set()
        self._ledger_path = run_root / "pipeline_results.jsonl"
        self._environment = self._worker_environment()

    @staticmethod
    def _worker_environment() -> dict[str, str]:
        environment = {**os.environ, "PYTHONUTF8": "1",
                       "APPWORLD_ROOT": str(ROOT / "vendor" / "appworld")}
        environment.setdefault("APPWORLD_CACHE", str(Path(environment["APPWORLD_ROOT"]) / ".tmp" / "agent-cache"))
        source_paths = [str(ROOT / "src"), str(ROOT / "vendor" / "appworld" / "src")]
        if environment.get("PYTHONPATH"):
            source_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(source_paths)
        return environment

    def _command(self, task_id: str, task_dir: Path) -> list[str]:
        config = self.config
        command = [sys.executable, "-u", str(ROOT / "scripts" / "run_siwc_agent.py"),
                   "--task-id", task_id, "--dataset-name", config.get("dataset_name", "train"),
                   "--context-mode", config["context_mode"],
                   "--max-steps", str(config["max_steps"]),
                   "--siwc-connect-timeout", str(config["connect_timeout"]),
                   "--siwc-read-timeout", str(config["read_timeout"]),
                   "--max-no-progress-steps", str(config["max_no_progress_steps"]),
                   "--assurance-mode", config["assurance_mode"],
                   "--experiment", self.experiment, "--task-dir", str(task_dir),
                   "--defer-evaluation"]
        if config.get("model"):
            command.extend(("--model", config["model"]))
        if config.get("optimized_unified"):
            command.append("--optimized-unified")
        for name, option in (("schema_aware_retrieval", "schema-aware-retrieval"),
                             ("programmatic_execution", "programmatic-execution")):
            value = config.get(name)
            if value is not None:
                command.append(f"--{'no-' if not value else ''}{option}")
        if config.get("plan_guided_react"):
            command.append("--plan-guided-react")
        if config.get("prompt_cache_key"):
            command.append("--prompt-cache-key")
        return command

    def _infer_subprocess(self, task_id: str, task_dir: Path) -> dict[str, Any]:
        # The dispatcher creates a fresh attempt directory before submitting
        # the worker. Reuse it here; asking for exist_ok=False would reject
        # every correctly dispatched task on Windows.
        task_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        log_path = task_dir / "worker.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(self._command(task_id, task_dir), cwd=ROOT,
                                       env=self._environment, stdout=log, stderr=subprocess.STDOUT)
            with self._process_lock:
                self._active_processes[task_id] = process
            try:
                try:
                    exit_code = process.wait(timeout=self.config.get("task_timeout_seconds", 7200))
                    timed_out = False
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        exit_code = process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        exit_code = process.wait()
                    timed_out = True
            finally:
                with self._process_lock:
                    self._active_processes.pop(task_id, None)
        return {"task_id": task_id, "task_dir": task_dir, "exit_code": exit_code,
                "inference_seconds": time.monotonic() - started,
                "started_monotonic": started,
                "error": ("inference_worker_timeout" if timed_out else
                          None if exit_code == 0 else f"inference_worker_exit_{exit_code}")}

    def _persist(self, record: dict[str, Any], *, emit: bool = True) -> None:
        record["timestamp"] = datetime.now().astimezone().isoformat()
        with self._record_lock:
            with self._ledger_path.open("a", encoding="utf-8") as ledger:
                ledger.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
            self._records[record["task_id"]] = record
            if record["task_id"] in self._current_task_ids:
                self._current_done.add(record["task_id"])
            done = len(self._current_done)
            if emit:
                status = "PASS" if record.get("passed") == record.get("total") and record.get("total") else _score(record)
                total_seconds = record.get("elapsed_seconds", 0)
                print(f"[{done}/{self._expected}] {record['task_id']} | {status} | "
                      f"{record.get('steps', '—')} steps | {_format_tokens(record.get('input_tokens_total'))} tok | "
                      f"py_err {record.get('python_errors', 0)}/{record.get('python_calls', 0)} | "
                      f"elapsed {total_seconds:.0f}s" +
                      (f" | {_comparison_text(record['comparison'])}" if record.get("comparison") else ""), flush=True)

    def _evaluation_worker(self, jobs: queue.Queue) -> None:
        while True:
            job = jobs.get()
            try:
                if job is None:
                    return
                started = time.monotonic()
                try:
                    # AppWorld's evaluator changes process-wide simulated time;
                    # serialize it even if a caller requests multiple workers.
                    with self._evaluation_lock:
                        report = self.evaluate_fn(job["task_id"], self.experiment)
                    passed, total = parse_report_counts(report)
                    if passed is None or total is None:
                        raise ValueError("Official evaluation report did not contain complete score counts")
                    passed, total = parse_report_counts(report)
                    if passed is None or total is None:
                        raise ValueError("Official evaluation report did not contain complete score counts")
                    task_dir = Path(job["task_dir"])
                    (task_dir / "evaluation_report.md").write_text(report, encoding="utf-8")
                    elapsed = time.monotonic() - started
                    record = collect_task_metrics(job["task_id"], task_dir, report,
                                                  job["inference_seconds"], elapsed)
                    record["elapsed_seconds"] = round(
                        time.monotonic() - job.get("started_monotonic", time.monotonic()), 3)
                    baseline = (_baseline_task(self.compare_with, job["task_id"])
                                if self.compare_with else None)
                    record["comparison"] = _compare(record, baseline)
                    self._persist(record)
                except Exception as exc:
                    self._persist({"task_id": job["task_id"], "status": "error",
                                   "stage": "evaluation", "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                                   "task_dir": str(job["task_dir"]),
                                   "inference_seconds": round(job["inference_seconds"], 3)})
            finally:
                jobs.task_done()

    def _load_resume(self, task_ids: list[str]) -> list[str]:
        if self._ledger_path.exists():
            for line in self._ledger_path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    record = json.loads(line)
                    self._records[record["task_id"]] = record
                except (ValueError, KeyError):
                    continue
        pending = []
        for task_id in task_ids:
            artifacts = _scored_artifacts(self.run_root, task_id)
            if artifacts:
                self._current_done.add(task_id)
                task_dir, report, summary = artifacts
                existing = self._records.get(task_id, {})
                if existing.get("status") != "completed" or existing.get("task_dir") != str(task_dir) or (
                        self.compare_with is not None and not existing.get("comparison")):
                    record = collect_task_metrics(task_id, task_dir, report)
                    baseline = (_baseline_task(self.compare_with, task_id)
                                if self.compare_with else None)
                    record["comparison"] = _compare(record, baseline)
                    self._persist(record, emit=False)
                continue
            pending.append(task_id)
        return pending

    def _create_attempt(self, task_id: str) -> Path:
        attempt = _attempt_directory(self.run_root, task_id)
        attempt.mkdir(parents=True, exist_ok=False)
        return attempt

    def run(self, task_ids: list[str], *, resume: bool = False) -> dict[str, Any]:
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Task IDs must be unique.")
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._expected = len(task_ids)
        self._current_task_ids = set(task_ids)
        self._current_done = set()
        self._ledger_path.touch(exist_ok=True)
        pending = self._load_resume(task_ids) if resume else list(task_ids)
        if pending and self.preflight:
            try:
                self.preflight()
            except Exception as exc:
                for task_id in pending:
                    self._persist({"task_id": task_id, "status": "error", "stage": "preflight",
                                   "error": f"SIWC preflight failed ({type(exc).__name__})"})
                return self._write_summary(task_ids, interrupted=False)

        jobs: queue.Queue = queue.Queue()
        evaluation_pool = ThreadPoolExecutor(max_workers=self.eval_workers, thread_name_prefix="agentlab-eval")
        eval_consumers = [evaluation_pool.submit(self._evaluation_worker, jobs)
                          for _ in range(self.eval_workers)]
        inference_pool = ThreadPoolExecutor(max_workers=self.inference_workers,
                                            thread_name_prefix="agentlab-infer")
        iterator = iter(pending)
        futures: dict[Any, tuple[str, Path]] = {}
        interrupted = False

        def submit_next() -> bool:
            while True:
                try:
                    task_id = next(iterator)
                except StopIteration:
                    return False
                try:
                    task_dir = self._create_attempt(task_id)
                    future = inference_pool.submit(self.infer_fn, task_id, task_dir)
                    futures[future] = (task_id, task_dir)
                    return True
                except Exception as exc:
                    self._persist({"task_id": task_id, "status": "error", "stage": "dispatch",
                                   "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                                   "task_dir": str(self.run_root / task_id)})

        try:
            for _ in range(min(self.inference_workers, len(pending))):
                submit_next()
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    task_id, task_dir = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"exit_code": -1, "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                                  "inference_seconds": 0.0}
                    if result.get("exit_code") == 0 and (task_dir / "summary.json").exists():
                        jobs.put({"task_id": task_id, "task_dir": str(task_dir),
                                  "inference_seconds": float(result.get("inference_seconds", 0.0)),
                                  "started_monotonic": result.get("started_monotonic", time.monotonic())})
                    else:
                        self._persist({"task_id": task_id, "status": "error", "stage": "inference",
                                       "error": result.get("error") or "missing_summary_after_inference",
                                       "exit_code": result.get("exit_code"),
                                       "task_dir": str(task_dir),
                                       "inference_seconds": round(float(result.get("inference_seconds", 0)), 3)})
                    submit_next()
        except KeyboardInterrupt:
            interrupted = True
            print("\nCtrl+C received: stopping new inference dispatch and terminating active workers.", flush=True)
            with self._process_lock:
                processes = list(self._active_processes.items())
            for _task_id, process in processes:
                if process.poll() is None:
                    process.terminate()
            inference_pool.shutdown(wait=True, cancel_futures=True)
            for future, (task_id, task_dir) in list(futures.items()):
                if future.cancelled():
                    self._persist({"task_id": task_id, "status": "error", "stage": "inference",
                                   "error": "interrupted_before_start", "task_dir": str(task_dir)})
                else:
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"exit_code": -1, "error": type(exc).__name__, "inference_seconds": 0}
                    if result.get("exit_code") == 0 and (task_dir / "summary.json").exists():
                        jobs.put({"task_id": task_id, "task_dir": str(task_dir),
                                  "inference_seconds": float(result.get("inference_seconds", 0)),
                                  "started_monotonic": result.get("started_monotonic", time.monotonic())})
                    else:
                        self._persist({"task_id": task_id, "status": "error", "stage": "inference",
                                       "error": "interrupted" if result.get("exit_code") else "inference_interrupted",
                                       "exit_code": result.get("exit_code"), "task_dir": str(task_dir),
                                       "inference_seconds": round(float(result.get("inference_seconds", 0)), 3)})
            futures.clear()
        else:
            inference_pool.shutdown(wait=True)
        finally:
            for _ in range(self.eval_workers):
                jobs.put(None)
            try:
                jobs.join()
            except KeyboardInterrupt:
                interrupted = True
                print("\nCtrl+C received during evaluation drain; waiting for queued scores to flush.", flush=True)
                jobs.join()
            evaluation_pool.shutdown(wait=True)

        summary = self._write_summary(task_ids, interrupted)
        return summary

    def _write_summary(self, task_ids: list[str], interrupted: bool) -> dict[str, Any]:
        records = [self._records[task_id] for task_id in task_ids if task_id in self._records]
        scored = [row for row in records if row.get("passed") is not None and row.get("total") is not None]
        passed_tasks = sum(row["total"] > 0 and row["passed"] == row["total"] for row in scored)
        passed_assertions = sum(row["passed"] for row in scored)
        total_assertions = sum(row["total"] for row in scored)
        tools: Counter[str] = Counter()
        error_types: Counter[str] = Counter()
        py_errors = py_calls = exploration_calls = total_calls = 0
        for row in scored:
            tools.update(row.get("tool_calls", {}))
            py_errors += row.get("python_errors", 0)
            py_calls += row.get("python_calls", 0)
            exploration_calls += row.get("exploration_calls", 0)
            total_calls += row.get("total_tool_calls", 0)
            error_types.update(row.get("error_envelope_hits", {}))
        summary = {
            "status": ("interrupted" if interrupted else "completed" if len(scored) == len(task_ids)
                       and all(row.get("status") == "completed" for row in records) else "completed_with_errors"),
            "task_count": len(task_ids), "scored_tasks": len(scored),
            "task_goal_completion": {"passed": passed_tasks, "total": len(task_ids)},
            "assertions": {"passed": passed_assertions, "total": total_assertions,
                           "rate": passed_assertions / total_assertions if total_assertions else None},
            "average_steps": (sum(row["steps"] for row in scored if isinstance(row.get("steps"), int)) /
                              max(1, sum(isinstance(row.get("steps"), int) for row in scored))),
            "input_tokens_total": sum(row.get("input_tokens_total", 0) for row in scored),
            "tool_calls": dict(sorted(tools.items())),
            "exploration_calls": exploration_calls,
            "exploration_share": exploration_calls / total_calls if total_calls else 0.0,
            "python_errors": {"count": py_errors, "calls": py_calls,
                              "rate": py_errors / py_calls if py_calls else 0.0},
            "execution_summary_hits": sum(row.get("execution_summary_hits", 0) for row in scored),
            "error_envelope_hits": dict(sorted(error_types.items())),
            "tasks": records,
        }
        (self.run_root / "pipeline_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = ["# Pipeline Summary", "", f"- Status: `{summary['status']}`",
                 f"- TGC: **{passed_tasks}/{len(task_ids)}**",
                 f"- Assertions: **{passed_assertions}/{total_assertions}** "
                 f"({summary['assertions']['rate']:.1%})" if total_assertions else "- Assertions: n/a",
                 f"- Average steps: {summary['average_steps']:.1f}",
                 f"- Total input tokens (main + Clone): {summary['input_tokens_total']:,}", "",
                 "| Task | Score | Steps | Input tokens | Python errors | Baseline delta | Stop reason / status |",
                 "|---|---:|---:|---:|---:|---|---|"]
        for row in records:
            lines.append(f"| {row['task_id']} | {_score(row)} | {row.get('steps', '—')} | "
                         f"{row.get('input_tokens_total', 0):,} | "
                         f"{row.get('python_errors', 0)}/{row.get('python_calls', 0)} | "
                         f"{_comparison_text(row.get('comparison')) or '—'} | "
                         f"{row.get('stop_reason') or row.get('error') or row.get('status', 'pending')} |")
        lines.extend(["", "## Tool friction", "",
                      f"- Tool calls: `{json.dumps(dict(sorted(tools.items())), ensure_ascii=False)}`",
                      f"- Exploration calls: {exploration_calls}/{total_calls} ({summary['exploration_share']:.1%})",
                      f"- Python errors: {py_errors}/{py_calls} ({summary['python_errors']['rate']:.1%})",
                      f"- `execution_summary` hits: {summary['execution_summary_hits']}",
                      f"- `error_envelope` hits: `{json.dumps(dict(sorted(error_types.items())), ensure_ascii=False)}`"])
        (self.run_root / "pipeline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"SUMMARY {passed_tasks}/{len(task_ids)} full; assertions {passed_assertions}/{total_assertions}; "
              f"avg_steps {summary['average_steps']:.1f}; input_tokens {summary['input_tokens_total']:,}", flush=True)
        return summary


def evaluate_official(task_id: str, experiment: str) -> str:
    """Run AppWorld's official evaluator against the persisted experiment DB."""
    from appworld.evaluator import evaluate_task
    tracker = evaluate_task(task_id=task_id, experiment_name=experiment)
    return tracker.report(print_it=False, colorize=False)


def _credential_preflight(model_name: str | None) -> None:
    from models.siwc_auth import access_token
    from models.siwc_responses import ChatGPTResponsesAdapter
    access_token()
    adapter = ChatGPTResponsesAdapter(model_name)
    adapter.close()


def _task_ids(args: argparse.Namespace, resume_config: dict[str, Any] | None) -> list[str]:
    if resume_config is not None and args.task_ids is None and args.task_file is None:
        return list(resume_config["tasks"])
    if args.task_ids is not None:
        ids = args.task_ids
    elif args.task_file:
        ids = [line.strip() for line in args.task_file.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    else:
        raise ValueError("Provide --task-ids or --task-file; on resume, the saved task list may be reused.")
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Task list must be non-empty and contain no duplicates.")
    return ids


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--task-ids", nargs="+", help="Task IDs from --dataset-name (default: train)")
    source.add_argument("--task-file", type=Path, help="UTF-8 text file with one task ID per line")
    parser.add_argument("--experiment", help="New run name; defaults to a timestamped pipeline name")
    parser.add_argument("--resume", type=Path, help="Resume an existing Pipeline Runner directory")
    parser.add_argument("--compare-with", type=Path, help="Existing baseline run directory")
    parser.add_argument("--inference-workers", type=int)
    parser.add_argument("--eval-workers", type=int)
    parser.add_argument("--model")
    parser.add_argument("--dataset-name", choices=("train", "dev", "test_normal", "test_challenge"))
    parser.add_argument("--context-mode", choices=("full", "dynamic", "unified", "cache_first"), default="full")
    parser.add_argument("--optimized-unified", action="store_true")
    parser.add_argument("--prompt-cache-key", action="store_true")
    parser.add_argument("--schema-aware-retrieval", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--programmatic-execution", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--task-timeout-seconds", type=int, default=7200)
    parser.add_argument("--siwc-read-timeout", type=int, default=90)
    parser.add_argument("--siwc-connect-timeout", type=int, default=20)
    parser.add_argument("--plan-guided-react", action="store_true")
    parser.add_argument("--assurance-mode", choices=("shadow", "enforce", "reflection_v1", "reflection_v2"), default="shadow")
    parser.add_argument("--max-no-progress-steps", type=int, default=18)
    args = parser.parse_args(argv)

    run_root = args.resume.resolve() if args.resume else None
    saved = _read_json(run_root / "effective_config.json") if run_root else None
    if run_root and not saved:
        parser.error("--resume directory must contain effective_config.json")
    try:
        task_ids = _task_ids(args, saved)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    dataset_name = (saved["agent_config"].get("dataset_name", "train") if saved
                    else args.dataset_name or "train")
    from appworld import load_task_ids
    if not set(task_ids) <= set(load_task_ids(dataset_name)):
        parser.error(f"Task IDs must belong to the selected {dataset_name} split.")
    if saved:
        if (args.task_ids is not None or args.task_file is not None) and task_ids != saved.get("tasks"):
            parser.error("Resume task IDs must match the saved task list.")
        config = saved["agent_config"]
        if args.dataset_name and args.dataset_name != config.get("dataset_name", "train"):
            parser.error("--dataset-name must match the saved resume configuration.")
        experiment = saved["experiment"]
        inference_workers = (args.inference_workers if args.inference_workers is not None
                             else saved.get("inference_workers", 1))
        eval_workers = (args.eval_workers if args.eval_workers is not None
                        else saved.get("eval_workers", 1))
    else:
        if args.assurance_mode in {"enforce", "reflection_v1", "reflection_v2"} and not args.plan_guided_react:
            parser.error(f"--assurance-mode {args.assurance_mode} requires --plan-guided-react")
        if args.optimized_unified and args.context_mode != "unified":
            parser.error("--optimized-unified requires --context-mode unified")
        if (not 10 <= args.siwc_read_timeout <= 300 or not 3 <= args.siwc_connect_timeout <= 60
                or args.task_timeout_seconds < 1):
            parser.error("Timeouts must be read=10..300, connect=3..60 and task>=1 seconds")
        schema_aware = args.schema_aware_retrieval
        programmatic = args.programmatic_execution
        config = {"model": args.model, "dataset_name": args.dataset_name or "train",
                  "context_mode": args.context_mode,
                  "optimized_unified": args.optimized_unified,
                  "prompt_cache_key": args.prompt_cache_key,
                  "schema_aware_retrieval": args.plan_guided_react if schema_aware is None else schema_aware,
                  "programmatic_execution": args.plan_guided_react if programmatic is None else programmatic,
                  "max_steps": args.max_steps, "connect_timeout": args.siwc_connect_timeout,
                  "read_timeout": args.siwc_read_timeout,
                  "task_timeout_seconds": args.task_timeout_seconds,
                  "plan_guided_react": args.plan_guided_react,
                  "assurance_mode": args.assurance_mode,
                  "max_no_progress_steps": args.max_no_progress_steps}
        experiment = args.experiment or datetime.now().strftime("agentlab_pipeline_%Y%m%d_%H%M%S")
        run_root = ROOT / "runs" / experiment
        if run_root.exists():
            parser.error(f"Run directory already exists; use --resume {run_root}")
        inference_workers = args.inference_workers if args.inference_workers is not None else 1
        eval_workers = args.eval_workers if args.eval_workers is not None else 1
        if inference_workers < 1 or eval_workers < 1:
            parser.error("Worker counts must be positive.")
        run_root.mkdir(parents=True)
        (run_root / "effective_config.json").write_text(json.dumps({
            "experiment": experiment, "tasks": task_ids, "agent_config": config,
            "inference_workers": inference_workers, "eval_workers": eval_workers,
            "created_at": datetime.now().astimezone().isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    if inference_workers < 1 or eval_workers < 1:
        parser.error("Worker counts must be positive.")
    if args.compare_with and not args.compare_with.exists():
        parser.error("--compare-with directory does not exist")
    runner = PipelineRunner(run_root, experiment, config,
                            inference_workers=inference_workers, eval_workers=eval_workers,
                            compare_with=args.compare_with,
                            preflight=lambda: _credential_preflight(config.get("model")))
    runner.run(task_ids, resume=bool(args.resume))


if __name__ == "__main__":
    main()
