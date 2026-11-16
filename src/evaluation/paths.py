"""Repository paths and run metadata shared by evaluation modules."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]

RUN_MANIFEST_SCHEMA = "ascb-evaluation-run/v1"
ENTITY_NORMALIZATION_MODES = ("none", "wikidata_alias_cached")
IDK_RECOGNITION_DESCRIPTIONS = {
    "standard": "empty answer set or a sole case-insensitive idk token",
    "robust": "whole answer is empty or consists only of recognized null phrases",
    "extended": "robust recognition plus explicit null phrases mixed with entities",
}
DEFAULT_EVALUATION_POLICIES = {
    "answer_representation": "trimmed exact-string sets parsed from pipe-delimited responses",
    "partition_metric": "A4 = A1 - A3 AND A3 subseteq A1",
    "difference_only_metric": "diagnostic only; excluded from the five-relation CON mean",
    "repeat_control_role": "auxiliary same-question stability control; excluded from CON",
    "repeat_aggregation": "condition_mean",
    "repeat_source_policy": (
        "condition-level mean of isolated Base repeats and the CtE Q1 control; "
        "unpublished when isolated Base coverage is below 600"
    ),
    "main_abstention_scoring": "worst_case",
    "table4_abstention_scoring": "individual_empty_set",
    "missing_record_policy": "remain missing; never converted to abstentions or imputed",
    "denominator_policy": "use observed records; report 600 only when coverage is complete",
}


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "run"


def create_run_directory(base: Path, run_type: str, recognition: str,
                         label: str | None = None) -> tuple[Path, str]:
    """Create a collision-safe, UTC-timestamped directory for one evaluation run."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = [stamp, _slug(run_type)]
    if label:
        parts.append(_slug(label))
    parts.append(f"idk-{_slug(recognition)}")
    run_id = "__".join(parts)
    candidate = base / run_id
    suffix = 2
    while candidate.exists():
        candidate = base / f"{run_id}__{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate, candidate.name


def display_path(value: Path | str, *, base: Path = REPO_ROOT) -> str:
    """Return an anonymous, repository-relative path for logs and metadata."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        return path.as_posix() or "."
    # Use lexical normalization first so repository-local symlinks such as
    # .venv/bin/python remain reproducible repository-relative paths.
    lexical = Path(os.path.abspath(path))
    base_resolved = Path(os.path.abspath(base))
    try:
        relative = lexical.relative_to(base_resolved)
    except ValueError:
        # External locations must remain distinguishable without exposing a
        # username, home directory, or machine-specific directory hierarchy.
        return f"<external>/{lexical.name}" if lexical.name else "<external>"
    return relative.as_posix() or "."


def _safe_text(value: str) -> str:
    """Remove absolute home/repository paths embedded in manifest strings."""
    if value and Path(value).is_absolute():
        return display_path(value)

    repo = str(REPO_ROOT.resolve(strict=False))
    result = value.replace(repo, ".")
    home = str(Path.home().resolve(strict=False))
    result = result.replace(home, "<home>")
    # Also protect manifests produced on another macOS account.
    result = re.sub(r"/Users/[^/\s\"']+", "<home>", result)
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return display_path(value)
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_set_manifest(paths: Iterable[Path], root: Path, *, include_files: int = 100,
                      root_label: str | None = None) -> dict:
    """Fingerprint a set of files, including entries only when reasonably small."""
    root = Path(root).resolve()
    entries = []
    aggregate = hashlib.sha256()
    for path in sorted({Path(item).resolve() for item in paths if Path(item).is_file()}):
        digest = _sha256(path)
        try:
            name = str(path.relative_to(root))
        except ValueError:
            name = display_path(path)
        entries.append({"path": name, "sha256": digest, "bytes": path.stat().st_size})
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    result = {
        "root": root_label if root_label is not None else display_path(root),
        "count": len(entries),
        "aggregate_sha256": aggregate.hexdigest(),
    }
    if len(entries) <= include_files:
        result["files"] = entries
    return result


def start_run_manifest(run_dir: Path, *, run_id: str, run_type: str,
                       config: dict, recognition: str,
                       input_files: Iterable[Path] = (), input_root: Path = REPO_ROOT,
                       entity_normalization: str = "none",
                       evaluation_policies: dict | None = None) -> dict:
    """Write the initial manifest for an evaluation run."""
    now = datetime.now().astimezone()
    if entity_normalization not in ENTITY_NORMALIZATION_MODES:
        raise ValueError(f"Unknown entity-normalization mode: {entity_normalization}")
    policies = {**DEFAULT_EVALUATION_POLICIES, **(evaluation_policies or {})}
    manifest = {
        "manifest_schema": RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "run_type": run_type,
        "status": "running",
        "created_at": now.isoformat(timespec="seconds"),
        "created_at_utc": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "command": _jsonable([sys.executable, *sys.argv]),
        "working_directory": display_path(Path.cwd()),
        "runtime": {"python": sys.version.split()[0]},
        "config": _jsonable(config),
        "idk_handling": {
            "recognition": recognition,
            "definition": IDK_RECOGNITION_DESCRIPTIONS.get(recognition, "not applicable"),
            "missing_observations_are_abstentions": False,
            "primary_binary_metric_policy": "recognized abstentions receive zero",
            "primary_similarity_policy": {
                "equivalence_and_partition": "recognized abstentions receive zero",
                "disjointness": "recognized abstentions receive one",
            },
        },
        "entity_normalization": {
            "mode": entity_normalization,
            "unmapped_entity_policy": "preserve original string",
        },
        "evaluation_policies": _jsonable(policies),
        "inputs": file_set_manifest(input_files, input_root),
        "outputs": None,
    }
    path = Path(run_dir) / "manifest.json"
    path.write_text(
        json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def finish_run_manifest(run_dir: Path, manifest: dict, *, status: str = "completed",
                        error: str | None = None) -> None:
    """Finalize a run manifest with output fingerprints and completion status."""
    run_dir = Path(run_dir)
    outputs = [path for path in run_dir.rglob("*")
               if path.is_file() and path.name != "manifest.json"]
    now = datetime.now().astimezone()
    manifest["status"] = status
    manifest["completed_at"] = now.isoformat(timespec="seconds")
    manifest["completed_at_utc"] = now.astimezone(timezone.utc).isoformat(timespec="seconds")
    manifest["outputs"] = file_set_manifest(
        outputs, run_dir, include_files=500, root_label=".",
    )
    if error is not None:
        manifest["error"] = error
    (run_dir / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
