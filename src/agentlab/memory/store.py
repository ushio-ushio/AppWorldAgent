"""Task-local observations and evidence-linked working memory."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
import math
import re
from typing import Any
from uuid import uuid4

from agentlab.tool_engineering.tools import _tokens


@dataclass(frozen=True)
class Observation:
    id: str
    task_id: str
    tool_call_id: str
    tool_name: str
    step: int
    content: str  # Same string object held by the immutable source tool message.
    arguments: str  # Original tool-call arguments; paged retrieval, never executed.
    api_names: tuple[str, ...]
    index_terms: tuple[str, ...]
    error_type: str | None
    identifiers: tuple[str, ...] = ()
    executed_requests: tuple[dict[str, Any], ...] | None = None

    @property
    def size_bytes(self) -> int:
        return len(self.content.encode("utf-8"))


class ObservationStore:
    """Lives for one task invocation; no raw observations are written to disk."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self._prefix = f"obs_{uuid4().hex[:12]}_"
        self._records: dict[str, Observation] = {}
        self._by_call: dict[str, str] = {}

    def add(self, *, tool_call_id: str, tool_name: str, step: int,
            content: str, arguments: str, outcome: dict[str, Any],
            executed_requests: tuple[dict[str, Any], ...] | None = None) -> Observation:
        if tool_call_id in self._by_call:
            raise ValueError("Tool call already has an observation")
        try:
            parsed = json.loads(arguments)
        except (TypeError, json.JSONDecodeError):
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        api_names: set[str] = set()
        if tool_name == "get_api_schema":
            name = parsed.get("api_name")
            if isinstance(name, str):
                api_names.add(name.lower())
        elif tool_name == "search_api":
            for row in outcome.get("result", []) if isinstance(outcome.get("result"), list) else []:
                if isinstance(row, dict) and isinstance(row.get("api_name"), str):
                    api_names.add(row["api_name"].lower())
        elif tool_name == "execute_python":
            code = parsed.get("code", "")
            if isinstance(code, str):
                api_names.update(match.lower() for match in re.findall(
                    r"\bapis\.([A-Za-z_]+\.[A-Za-z_]+)", code))
        error = outcome.get("execution_error") or outcome.get("error")
        error_type = None
        if isinstance(error, str):
            match = re.search(r"\b([A-Za-z_]+Error|[A-Za-z_]+Exception)\b", error)
            error_type = match.group(1) if match else "tool_error"
        identifiers: set[str] = set()
        def collect(value: Any, depth: int = 0) -> None:
            if depth > 4 or len(identifiers) >= 32:
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    if (isinstance(key, str) and key.lower().endswith("_id")
                            and isinstance(item, (str, int)) and not isinstance(item, bool)):
                        identifier = str(item)
                        if 2 <= len(identifier) <= 64:
                            identifiers.add(identifier.lower())
                    elif not any(secret in str(key).lower() for secret in ("password", "token", "secret")):
                        collect(item, depth + 1)
            elif isinstance(value, list):
                for item in value[:100]:
                    collect(item, depth + 1)
        collect(outcome.get("result"))
        # Index source names, query/code terms, and output text. All remain in
        # process memory; search results disclose only provenance, never snippets.
        index_text = " ".join((tool_name, str(parsed.get("query", "")),
                               str(parsed.get("code", "")), " ".join(api_names),
                               content, error_type or ""))
        number = len(self._records) + 1
        record = Observation(
            id=f"{self._prefix}{number:04d}", task_id=self.task_id,
            tool_call_id=tool_call_id, tool_name=tool_name, step=step,
            content=content, arguments=arguments, api_names=tuple(sorted(api_names)),
            index_terms=tuple(_tokens(index_text)), error_type=error_type,
            identifiers=tuple(sorted(identifiers)),
            executed_requests=executed_requests,
        )
        self._records[record.id] = record
        self._by_call[tool_call_id] = record.id
        return record

    def for_call(self, tool_call_id: str) -> Observation | None:
        record_id = self._by_call.get(tool_call_id)
        return self._records.get(record_id) if record_id else None

    def records(self) -> tuple[Observation, ...]:
        return tuple(self._records.values())

    def get(self, observation_id: str) -> Observation | None:
        return self._records.get(observation_id)

    def read_observation(self, observation_id: str, offset: int = 0,
                         limit: int = 1600, field: str = "content") -> dict[str, Any]:
        record = self._records.get(observation_id)
        if record is None:
            raise ValueError("Unknown observation ID for this task")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 4096:
            raise ValueError("limit must be between 1 and 4096 characters")
        if field not in {"content", "arguments"}:
            raise ValueError("field must be content or arguments")
        source = record.content if field == "content" else record.arguments
        end = min(offset + limit, len(source))
        page = {"observation_id": record.id, "source_tool_call_id": record.tool_call_id,
                "source_tool": record.tool_name, "step": record.step,
                "offset": offset, "next_offset": end if end < len(source) else None,
                "total_chars": len(source), "content": source[offset:end]}
        if field != "content":
            page["field"] = field
        return page

    def search_history(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            return []
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")
        words = _tokens(query)
        if not words:
            return []
        records = list(self._records.values())
        n = len(records)
        lengths = [len(record.index_terms) for record in records]
        avg = sum(lengths) / n if n else 1
        df = Counter(term for record in records for term in set(record.index_terms))
        exact_id = query.strip().lower()
        exact_api = query.strip().lower().removeprefix("apis.")
        scored: list[tuple[int, float, Observation]] = []
        for record, length in zip(records, lengths):
            counts = Counter(record.index_terms)
            score = sum(
                math.log1p((n - df[word] + .5) / (df[word] + .5))
                * counts[word] * 2.5 / (counts[word] + 1.5 * (.25 + .75 * length / max(avg, 1)))
                for word in words if counts[word]
            )
            exact = 2 if record.id.lower() == exact_id else 1 if exact_api in record.api_names else 0
            if exact or score > 0:
                scored.append((exact, score, record))
        scored.sort(key=lambda row: (-row[0], -row[1], -row[2].step, row[2].id))
        return [
            {"rank": rank, "observation_id": record.id,
             "source_tool_call_id": record.tool_call_id, "source_tool": record.tool_name,
             "step": record.step, "api_names": list(record.api_names),
             "size_bytes": record.size_bytes, "error_type": record.error_type,
             "score": score}
            for rank, (_, score, record) in enumerate(scored[:top_k], 1)
        ]


@dataclass(frozen=True)
class MemoryFact:
    memory_id: str
    namespace: str
    entity: str
    key: str
    value: str | int | float | bool | None
    source_observation_id: str
    created_step: int
    updated_step: int
    status: str  # proposed, observed, verified, invalidated
    confirmation_observation_id: str | None = None
    invalidated_by: str | None = None


class WorkingMemory:
    def __init__(self, store: ObservationStore):
        self.store = store
        self.task_id = store.task_id
        self._facts: dict[str, MemoryFact] = {}

    def _source(self, observation_id: str):
        record = next((r for r in self.store.records() if r.id == observation_id), None)
        if record is None or record.task_id != self.task_id:
            raise ValueError("Observation ID is not available in this task")
        return record

    @staticmethod
    def _validate(namespace: str, entity: str, key: str, value: Any) -> None:
        if not all(isinstance(x, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}", x)
                   for x in (namespace, entity, key)):
            raise ValueError("namespace, entity, and key must be bounded identifiers")
        if any(secret in key.lower() for secret in ("password", "token", "secret", "credential")):
            raise ValueError("Credential-like fields cannot be stored in Working Memory")
        if not (value is None or type(value) in (str, int, float, bool)):
            raise ValueError("value must be a JSON scalar")
        if isinstance(value, str) and (len(value) > 160 or "\n" in value or "\r" in value):
            raise ValueError("String facts must be short, single-line data")

    def _write(self, *, namespace: str, entity: str, key: str, value: Any,
               observation_id: str, status: str) -> dict[str, Any]:
        self._validate(namespace, entity, key, value)
        source = self._source(observation_id)
        identity = (namespace, entity, key)
        current = [fact for fact in self._facts.values()
                   if fact.status != "invalidated" and (fact.namespace, fact.entity, fact.key) == identity]
        for fact in current:
            if type(fact.value) is type(value) and fact.value == value and fact.status == status:
                return self._public(fact)  # exact repeat; no duplicate fact
        memory_id = f"mem_{len(self._facts) + 1:04d}"
        fact = MemoryFact(memory_id, namespace, entity, key, value, observation_id,
                          source.step, source.step, status)
        self._facts[memory_id] = fact
        if status == "observed":
            # A later structured observation of the same field is a verifiable
            # newer snapshot. Equal-step or proposed conflicts remain explicit.
            for old in current:
                if old.status in {"observed", "verified"} and old.updated_step < source.step:
                    self._facts[old.memory_id] = replace(old, status="invalidated",
                                                         updated_step=source.step,
                                                         invalidated_by=memory_id)
                elif old.status == "proposed" and type(old.value) is type(value) and old.value == value:
                    self._facts[old.memory_id] = replace(old, status="invalidated",
                                                         updated_step=source.step,
                                                         invalidated_by=memory_id)
        return self._public(fact)

    def remember_fact(self, namespace: str, entity: str, key: str, value: Any,
                      observation_id: str) -> dict[str, Any]:
        return self._write(namespace=namespace, entity=entity, key=key, value=value,
                           observation_id=observation_id, status="proposed")

    def ingest(self, observation_id: str, outcome: dict[str, Any], arguments: str) -> list[dict[str, Any]]:
        source = self._source(observation_id)
        if source.tool_name != "execute_python" or not outcome.get("ok") or outcome.get("execution_error"):
            return []
        apps = set(re.findall(r"\bapis\.([A-Za-z_]+)\.[A-Za-z_]+", arguments)) - {"supervisor"}
        if len(apps) != 1:
            return []
        value = outcome.get("result")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError):
                return []  # arbitrary printed Python text is not structured evidence
        rows = value if isinstance(value, list) else [value]
        if len(rows) > 100:
            return []
        written = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            ids = [(key, item) for key, item in row.items()
                   if isinstance(key, str) and (key == "id" or key.endswith("_id"))
                   and type(item) in (str, int) and 1 <= len(str(item)) <= 64]
            if len(ids) != 1:
                continue  # ambiguous entity identity
            id_key, id_value = ids[0]
            entity = f"{id_key}:{id_value}"
            for key, item in row.items():
                if key == id_key or not isinstance(key, str) or key.endswith("_id"):
                    continue
                if type(item) is str and not re.fullmatch(r"[A-Za-z0-9_./:+-]{1,80}", item):
                    continue  # free text may contain instructions; do not promote it
                try:
                    before = len(self._facts)
                    fact = self._write(namespace=next(iter(apps)), entity=entity,
                                       key=key, value=item,
                                       observation_id=observation_id, status="observed")
                    if len(self._facts) > before:
                        written.append(fact)
                except ValueError:
                    continue
        return written

    def verify(self, memory_id: str, confirmation_observation_id: str,
               outcome: dict[str, Any], arguments: str) -> dict[str, Any]:
        fact = self._facts.get(memory_id)
        if fact is None or fact.status != "observed":
            raise ValueError("Only an observed fact can be verified")
        source = self._source(confirmation_observation_id)
        if source.id == fact.source_observation_id or source.step <= fact.created_step:
            raise ValueError("Verification requires a later, independent observation")
        if source.tool_name != "execute_python" or not outcome.get("ok"):
            raise ValueError("Confirmation must come from successful Python observation")
        # Parse with a temporary extractor but do not write or mutate facts.
        apps = set(re.findall(r"\bapis\.([A-Za-z_]+)\.[A-Za-z_]+", arguments)) - {"supervisor"}
        if apps != {fact.namespace}:
            raise ValueError("Confirmation namespace differs")
        result = outcome.get("result")
        try:
            result = json.loads(result) if isinstance(result, str) else result
        except ValueError as exc:
            raise ValueError("Confirmation is not structured JSON") from exc
        rows = result if isinstance(result, list) else [result]
        if not any(isinstance(row, dict)
                   and any(f"{key}:{value}" == fact.entity for key, value in row.items()
                           if key == "id" or key.endswith("_id"))
                   and type(row.get(fact.key)) is type(fact.value)
                   and row.get(fact.key) == fact.value for row in rows):
            raise ValueError("Confirmation does not contain the same entity and value")
        fact = replace(fact, status="verified", updated_step=source.step,
                       confirmation_observation_id=source.id)
        self._facts[memory_id] = fact
        return self._public(fact)

    def invalidate_memory(self, memory_id: str, observation_id: str, reason: str) -> dict[str, Any]:
        fact = self._facts.get(memory_id)
        source = self._source(observation_id)
        if fact is None or fact.status == "invalidated":
            raise ValueError("Memory is unknown or already invalidated")
        if source.step < fact.created_step or not isinstance(reason, str) or not reason.strip():
            raise ValueError("Invalidation requires a source and non-empty reason")
        if fact.status in {"observed", "verified"}:
            # A model's unverified statement cannot erase an observed fact.
            newer = [r for r in self._facts.values() if r.source_observation_id == source.id
                     and (r.namespace, r.entity, r.key) == (fact.namespace, fact.entity, fact.key)
                     and r.value != fact.value and r.status == "observed"]
            if not newer:
                raise ValueError("Observed facts require a contrary structured observation")
        fact = replace(fact, status="invalidated", updated_step=source.step,
                       invalidated_by=observation_id)
        self._facts[memory_id] = fact
        return self._public(fact)

    def _public(self, fact: MemoryFact) -> dict[str, Any]:
        value = asdict(fact)
        value["conflict_with"] = [other.memory_id for other in self._facts.values()
                                  if other.memory_id != fact.memory_id
                                  and other.status != "invalidated"
                                  and (other.namespace, other.entity, other.key) ==
                                  (fact.namespace, fact.entity, fact.key)
                                  and other.value != fact.value]
        return value

    def list_memory(self, namespace: str | None = None, entity: str | None = None,
                    include_invalidated: bool = False) -> list[dict[str, Any]]:
        rows = [fact for fact in self._facts.values()
                if (include_invalidated or fact.status != "invalidated")
                and (namespace is None or fact.namespace == namespace)
                and (entity is None or fact.entity == entity)]
        return [self._public(fact) for fact in sorted(rows, key=lambda r: (-r.updated_step, r.memory_id))[:30]]

    def all_facts(self) -> tuple[MemoryFact, ...]:
        return tuple(self._facts.values())

    def active_records(self) -> list[dict[str, Any]]:
        """All active facts for local context selection; tool listing remains bounded."""
        return [self._public(fact) for fact in self._facts.values()
                if fact.status != "invalidated"]
