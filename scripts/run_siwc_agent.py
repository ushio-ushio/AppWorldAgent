"""Opt-in single-task AppWorld run through official SIWC Responses OAuth."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["APPWORLD_ROOT"] = str(Path(os.getenv("APPWORLD_ROOT", ROOT / "vendor" / "appworld")))
os.environ.setdefault("APPWORLD_CACHE", str(Path(os.environ["APPWORLD_ROOT"]) / ".tmp" / "agent-cache"))

from context_engineering.unified import UnifiedContextConfig  # noqa: E402
from agent.react import LangGraphAgent  # noqa: E402
from models.siwc_responses import ChatGPTResponsesAdapter  # noqa: E402
from appworld import AppWorld, load_task_ids  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--dataset-name", choices=("train", "dev", "test_normal", "test_challenge"),
                        default="train")
    parser.add_argument("--model", help="Account-visible SIWC model slug; default is first listed")
    parser.add_argument("--context-mode", choices=("full", "dynamic", "unified", "cache_first"), default="full")
    parser.add_argument("--optimized-unified", action="store_true")
    parser.add_argument("--prompt-cache-key", action="store_true", help="Send a task-stable SIWC prompt_cache_key")
    parser.add_argument("--schema-aware-retrieval", action=argparse.BooleanOptionalAction, default=None,
                        help="Include bounded official schemas in top API search results")
    parser.add_argument("--programmatic-execution", action=argparse.BooleanOptionalAction, default=None,
                        help="Guide batching of deterministic work inside execute_python")
    parser.add_argument("--experiment")
    parser.add_argument("--task-dir")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--siwc-read-timeout", type=int, default=90,
                        help="Seconds to wait for response headers or the next stream chunk")
    parser.add_argument("--siwc-connect-timeout", type=int, default=20,
                        help="Seconds to establish each SIWC proxy/TLS connection")
    parser.add_argument("--plan-guided-react", action="store_true",
                        help="One-shot advisory plan followed by ordinary ReAct execution")
    parser.add_argument("--assurance-mode", choices=("shadow", "enforce", "reflection_v1", "reflection_v2"), default="shadow",
                        help="Execution Assurance: shadow (baseline), enforce, reflection_v1, or reflection_v2")
    parser.add_argument("--max-no-progress-steps", type=int, default=18,
                        help="V3 consecutive steps without new evidence or error recovery")
    parser.add_argument("--defer-evaluation", action="store_true",
                        help="Leave official evaluation to a separate pipeline worker")
    args = parser.parse_args()
    if args.assurance_mode in {"enforce", "reflection_v1", "reflection_v2"} and not args.plan_guided_react:
        parser.error(f"--assurance-mode {args.assurance_mode} requires --plan-guided-react")
    if args.plan_guided_react and os.name == "nt" and not sys.flags.utf8_mode:
        parser.error("V2-Lite on Windows requires PYTHONUTF8=1 or python -X utf8 before startup")
    if args.schema_aware_retrieval is None:
        args.schema_aware_retrieval = args.plan_guided_react
    if args.programmatic_execution is None:
        args.programmatic_execution = args.plan_guided_react
    if args.task_id not in set(load_task_ids(args.dataset_name)):
        parser.error(f"Task ID is not in the selected {args.dataset_name} split")
    if args.optimized_unified and args.context_mode != "unified":
        parser.error("--optimized-unified requires --context-mode unified")
    if not 10 <= args.siwc_read_timeout <= 300:
        parser.error("--siwc-read-timeout must be between 10 and 300 seconds")
    if not 3 <= args.siwc_connect_timeout <= 60:
        parser.error("--siwc-connect-timeout must be between 3 and 60 seconds")
    session_id = str(uuid.uuid4())
    model = ChatGPTResponsesAdapter(args.model, timeout_seconds=args.siwc_read_timeout,
                                    connect_timeout_seconds=args.siwc_connect_timeout,
                                    session_id=session_id)
    config = UnifiedContextConfig(
        memory_selection=True if args.optimized_unified else False,
        dependency_selection=True if args.optimized_unified else False)
    experiment = args.experiment or "agentlab_siwc_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    model.prompt_cache_key = f"agentlab:{experiment}:{args.task_id}" if args.prompt_cache_key else None
    run_dir = Path(args.task_dir) if args.task_dir else ROOT / "runs" / experiment / args.task_id
    run_dir.mkdir(parents=True, exist_ok=args.task_dir is not None)
    if (run_dir / "trace.jsonl").exists():
        parser.error(f"Trace already exists; refusing to overwrite {run_dir / 'trace.jsonl'}")
    (run_dir / "effective_config.json").write_text(json.dumps({
        "task_id": args.task_id, "dataset": args.dataset_name, "model": model.model_name,
        "adapter": "official_siwc_responses_oauth", "context_mode": args.context_mode,
        "unified_config": config.__dict__ if args.context_mode == "unified" else None,
        "max_steps": args.max_steps,
        "siwc_read_timeout_seconds": args.siwc_read_timeout,
        "siwc_connect_timeout_seconds": args.siwc_connect_timeout,
        "max_cost_usd": 0.05, "max_input_tokens": None,
        "max_output_tokens_total": sys.maxsize,
        "architecture": "plan_guided_react" if args.plan_guided_react else "react",
        "execution_assurance": {"mode": args.assurance_mode,
                                "enabled": args.plan_guided_react,
                                "terminal_reflection": args.assurance_mode == "reflection_v1",
                                "terminal_audit": args.assurance_mode == "reflection_v2",
                                "api_boundary": "AppWorld in-process Requester.request"},
        "max_no_progress_steps": args.max_no_progress_steps,
        "efficient_tool_use": {"schema_aware_retrieval": args.schema_aware_retrieval,
                                "programmatic_execution": args.programmatic_execution},
        "store": False, "stream": True, "api_key_used": False,
        "prompt_cache_key_enabled": args.prompt_cache_key,
        "evaluation_deferred": args.defer_evaluation,
        "windows_python_utf8_mode": bool(sys.flags.utf8_mode),
        "per_request_dollar_cost": "not_available_for_ChatGPT_plan_usage",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    with closing(model), AppWorld(task_id=args.task_id, experiment_name=experiment) as world:
        state = LangGraphAgent(model, max_steps=args.max_steps,
                               max_input_tokens=None, max_output_tokens=sys.maxsize,
                               context_mode=args.context_mode,
                               schema_aware_retrieval=args.schema_aware_retrieval,
                               programmatic_execution=args.programmatic_execution,
                               unified_config=config,
                               plan_guided_react=args.plan_guided_react,
                               assurance_mode=args.assurance_mode,
                               max_no_progress_steps=args.max_no_progress_steps).solve(world, trace_path=run_dir / "trace.jsonl")
        report = None if args.defer_evaluation else world.evaluate().report(print_it=False, colorize=False)
        if report is not None:
            (run_dir / "evaluation_report.md").write_text(report, encoding="utf-8")
        print(json.dumps({"run_dir": str(run_dir), "steps": state.get("step_count"),
                          "stop_reason": state.get("stop_reason"), "usage": state.get("total_usage"),
                          "evaluation": "deferred" if args.defer_evaluation else "completed"},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
