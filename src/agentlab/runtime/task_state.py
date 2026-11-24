"""Task-local progress claims, kept separate from AppWorld completion/evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from agentlab.memory.store import ObservationStore


@dataclass
class Subtask:
    subtask_id: str
    description: str
    status: str = "pending"
    evidence_observation_id: str | None = None
    evidence_level: str = "none"  # none, agent_claim, observed
    updated_step: int = 0


class TaskState:
    def __init__(self, task_id: str, original_goal: str, store: ObservationStore):
        if store.task_id != task_id:
            raise ValueError("Task State and Observation Store must belong to one task")
        self.task_id = task_id
        self._original_goal = original_goal
        self.store = store
        self.focus: str | None = None
        self.subtasks: dict[str, Subtask] = {}
        self.issues: list[dict[str, Any]] = []

    @property
    def original_goal(self) -> str:
        return self._original_goal

    def update_task_state(self, action: str, *, subtask_id: str | None = None,
                          description: str | None = None, status: str | None = None,
                          observation_id: str | None = None, focus: str | None = None,
                          note: str | None = None) -> dict[str, Any]:
        if action == "focus":
            if not isinstance(focus, str) or not focus.strip() or len(focus) > 200:
                raise ValueError("focus must be short, non-empty text")
            self.focus = focus.strip()
            return {"focus": self.focus}
        if action == "issue":
            if not isinstance(note, str) or not note.strip() or len(note) > 300:
                raise ValueError("issue needs short non-empty text")
            source = self._source(observation_id) if observation_id else None
            issue = {"note": note.strip(), "observation_id": source.id if source else None,
                     "step": source.step if source else None}
            self.issues.append(issue)
            return issue
        if action != "subtask":
            raise ValueError("Unknown Task State action")
        if not isinstance(subtask_id, str) or not subtask_id or len(subtask_id) > 64:
            raise ValueError("subtask_id is required")
        if status not in {"pending", "in_progress", "done", "failed"}:
            raise ValueError("Invalid subtask status")
        previous = self.subtasks.get(subtask_id)
        allowed = {"pending": {"pending", "in_progress", "done", "failed"},
                   "in_progress": {"in_progress", "done", "failed"},
                   "done": {"done", "in_progress"},
                   "failed": {"failed", "in_progress"}}
        if previous and status not in allowed[previous.status]:
            raise ValueError("Illegal subtask transition")
        label = description if description is not None else (previous.description if previous else None)
        if not isinstance(label, str) or not label.strip() or len(label) > 200:
            raise ValueError("description is required and must be short")
        source = self._source(observation_id) if observation_id else None
        if status == "done" and source is None:
            raise ValueError("Done requires an Observation ID as evidence")
        # A successful tool call is not proof the business goal was reached.
        level = ("agent_claim" if status == "done" else "observed" if source else "none")
        row = Subtask(subtask_id, label.strip(), status, source.id if source else None,
                      level, source.step if source else (previous.updated_step if previous else 0))
        self.subtasks[subtask_id] = row
        return asdict(row)

    def _source(self, observation_id: str):
        record = next((r for r in self.store.records() if r.id == observation_id), None)
        if record is None or record.task_id != self.task_id:
            raise ValueError("Unknown Observation ID for this task")
        return record

    def snapshot(self) -> dict[str, Any]:
        return {"focus": self.focus,
                "subtasks": [asdict(s) for s in self.subtasks.values()],
                "issues": list(self.issues)}
