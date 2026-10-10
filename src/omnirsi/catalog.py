"""Read-only package-data catalog. Knowledge text is never executable."""

from __future__ import annotations

import json
import re
from importlib.resources import files
from typing import Any

REPO_TYPES = ("vllm", "vllm_omni", "afd_plugin", "vllm_rlt")
BACKENDS = ("cpu", "cuda", "rocm", "ascend")
SCENARIO_ALIASES = {"diffusion.t2i": "diffusion.image_generation",
                    "diffusion.t2v": "diffusion.video_generation"}


def _walk(root):
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if child.is_dir():
            yield from _walk(child)
        else:
            yield child


def knowledge_records() -> list[dict[str, Any]]:
    root = files("omnirsi").joinpath("data", "repo_packs")
    records = []
    seen = set()
    for item in _walk(root):
        if item.name != "experience.json":
            continue
        record = json.loads(item.read_text(encoding="utf-8"))
        required = ("id", "title", "repo", "backend", "status", "sources", "summary")
        if any(key not in record for key in required):
            raise ValueError(f"Incomplete knowledge record: {item}")
        if record["id"] in seen:
            raise ValueError(f"Duplicate knowledge ID: {record['id']}")
        if record["repo"] not in REPO_TYPES:
            raise ValueError(f"Unknown knowledge repository: {record['repo']}")
        if record["backend"] not in (*BACKENDS, "common"):
            raise ValueError(f"Unknown knowledge backend: {record['backend']}")
        if not record["sources"]:
            raise ValueError(f"Knowledge has no source: {record['id']}")
        if record["status"] == "imported" and record.get("verification", {}).get("locally_verified") is not False:
            raise ValueError(f"Imported knowledge must state locally_verified=false: {record['id']}")
        seen.add(record["id"])
        records.append(record)
    return records


def search_knowledge(
    query: str = "", *, repo: str | None = None,
    backend: str | None = None, scenario: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be positive")
    scenario = SCENARIO_ALIASES.get(scenario, scenario)
    tokens = re.findall(r"[\w.-]+", query.casefold())
    scored = []
    for record in knowledge_records():
        if repo and record["repo"] != repo:
            continue
        if backend and record["backend"] not in (backend, "common"):
            continue
        if scenario:
            declared = record.get("scenario", "")
            if declared != scenario and not (declared.endswith(".*") and scenario.startswith(declared[:-1])):
                continue
        document = json.dumps(record, ensure_ascii=False).casefold()
        if any(token not in document for token in tokens):
            continue
        score = sum(document.count(token) for token in tokens)
        scored.append((score, record["id"], record))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:limit]]


def get_knowledge(record_id: str) -> dict[str, Any]:
    for record in knowledge_records():
        if record["id"] == record_id:
            return record
    raise ValueError(f"Unknown knowledge ID: {record_id}")


def repository_pack(repo_type: str) -> dict[str, Any]:
    if repo_type not in REPO_TYPES:
        raise ValueError(f"Unknown repository type: {repo_type}")
    path = files("omnirsi").joinpath("data", "repo_packs", repo_type, "pack.json")
    return json.loads(path.read_text(encoding="utf-8"))
