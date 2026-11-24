# AppWorld AgentLab

This repository contains the V3 Shadow Plan-Guided ReAct runtime and its local execution-assurance safeguards. It uses AppWorld as the environment and official evaluator; AgentLab owns planning, tool dispatch, context, traces, and SIWC model access.

AppWorld source code, task data, hidden evaluation material, run traces, and SIWC credentials are not included. Install AppWorld and its permitted data separately using the official instructions. Do not add those files or credentials to this repository.

## Code map

- `src/agentlab/agent`: one-shot plan and Function Calling ReAct loop.
- `context_engineering`: full, dynamic, and unified model-input views.
- `memory`: task-local observations and evidence-linked working memory.
- `tool_engineering`: official API retrieval, schemas, and validated tool dispatch.
- `execution_assurance`: requester gates, completion guard, and local error recovery.
- `models`: official SIWC OAuth and Responses adapter.
- `environments`: persistent AppWorld Python Shell adapter.
- `runtime`: task state and trace writing.
- `evaluation`: parsing official evaluator reports.

`experiments/run_v3_shadow_train15.py` is the retained Train-15 batch runner. It reads `configs/v3_shadow_train15.json` and writes new runs to the sibling `runs` directory; it starts model-backed tasks only when explicitly run. The current Windows runner expects this checkout at `D:\worldapp\.v1_2_b_stage` and AppWorld at `D:\worldapp\vendor\appworld`.

## Offline checks

From this directory in the Windows Conda `agent` environment, with AppWorld installed at the path above:

```powershell
$env:APPWORLD_ROOT = "D:\worldapp\vendor\appworld"
$env:PYTHONPATH = "D:\worldapp\.v1_2_b_stage\src;$env:APPWORLD_ROOT\src"
python -m compileall -q src scripts experiments tests
python -m pytest -q
```

The single-task script is `scripts/run_siwc_agent.py`. It calls the SIWC model and AppWorld only when invoked with a task ID. SIWC uses the official ChatGPT sign-in flow and the current Windows user's protected credential store; follow `docs/siwc_manual_browser_login.md` for account authorization details. No API key is required or stored in this repository.
