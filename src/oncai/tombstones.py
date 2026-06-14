"""Append-only tombstones for logical inbox-source removal."""

from __future__ import annotations

import getpass
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from oncai.config import OncaiConfig
from oncai.sidecar import ensure_sidecar

TOMBSTONE_FOLDER = "tombstones"
TOMBSTONE_SUFFIX = ".tombstone.json"
SUPPORTED_TOMBSTONE_KINDS = frozenset({"fc_extractions", "fc_reviews", "cohorts"})


class TombstoneAction(StrEnum):
    """A logical delete event action."""

    FORGET = "forget"
    REVIVE = "revive"


@dataclass(frozen=True)
class TombstoneEvent:
    """One immutable tombstone event."""

    event_id: str
    kind: str
    target: str
    action: TombstoneAction
    reason: str
    actor: str
    at: str
    path: Path | None = None


@dataclass(frozen=True)
class ResolvedTombstones:
    """All tombstone events plus the latest event per target."""

    events: tuple[TombstoneEvent, ...]
    latest_by_key: dict[tuple[str, str], TombstoneEvent]
    errors: tuple[str, ...]

    def active_targets(self, kind: str) -> set[str]:
        """Targets whose latest event is ``forget`` for ``kind``."""
        return {
            target
            for (event_kind, target), event in self.latest_by_key.items()
            if event_kind == kind and event.action == TombstoneAction.FORGET
        }

    def active_events(self) -> list[TombstoneEvent]:
        """Latest ``forget`` events across all supported kinds."""
        return [
            event
            for event in self.latest_by_key.values()
            if event.action == TombstoneAction.FORGET
        ]

    def is_active_event(self, event: TombstoneEvent) -> bool:
        """Whether ``event`` is the latest active forget for its target."""
        latest = self.latest_by_key.get((event.kind, event.target))
        return latest == event and event.action == TombstoneAction.FORGET


def tombstone_dir(config: OncaiConfig) -> Path:
    """Return the canonical inbox tombstone directory."""
    return config.inbox_path / TOMBSTONE_FOLDER


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_at(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid at timestamp: {value}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _event_sort_key(event: TombstoneEvent) -> tuple[datetime, str]:
    return _parse_at(event.at), event.event_id


def validate_tombstone_target(kind: str, target: str) -> None:
    """Validate the v1 tombstone target namespace."""
    if kind not in SUPPORTED_TOMBSTONE_KINDS:
        supported = ", ".join(sorted(SUPPORTED_TOMBSTONE_KINDS))
        raise ValueError(f"Unsupported tombstone kind '{kind}'. Supported: {supported}")
    if not target.strip():
        raise ValueError("Tombstone target cannot be blank")
    if "/" in target or "\\" in target:
        raise ValueError("Tombstone target must be a single source name, not a path")


def _event_from_payload(path: Path, payload: dict[str, Any]) -> TombstoneEvent:
    missing = [
        key
        for key in ("event_id", "kind", "target", "action", "reason", "actor", "at")
        if key not in payload
    ]
    if missing:
        raise ValueError(f"missing required field(s): {', '.join(missing)}")

    event_id = str(payload["event_id"])
    kind = str(payload["kind"])
    target = str(payload["target"])
    action = TombstoneAction(str(payload["action"]))
    validate_tombstone_target(kind, target)
    event = TombstoneEvent(
        event_id=event_id,
        kind=kind,
        target=target,
        action=action,
        reason=str(payload["reason"]),
        actor=str(payload["actor"]),
        at=str(payload["at"]),
        path=path,
    )
    _parse_at(event.at)
    return event


def resolve_tombstones(config: OncaiConfig) -> ResolvedTombstones:
    """Read tombstone files and resolve latest-wins state per target."""
    events: list[TombstoneEvent] = []
    latest_by_key: dict[tuple[str, str], TombstoneEvent] = {}
    errors: list[str] = []

    directory = tombstone_dir(config)
    if not directory.exists():
        return ResolvedTombstones(events=(), latest_by_key={}, errors=())

    for path in sorted(directory.glob(f"*{TOMBSTONE_SUFFIX}")):
        try:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                errors.append(
                    f"{path.name}: TypeError: tombstone payload must be a JSON object"
                )
                continue
            event = _event_from_payload(path, payload)
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue

        events.append(event)
        key = (event.kind, event.target)
        previous = latest_by_key.get(key)
        if previous is None or _event_sort_key(event) >= _event_sort_key(previous):
            latest_by_key[key] = event

    return ResolvedTombstones(
        events=tuple(events),
        latest_by_key=latest_by_key,
        errors=tuple(errors),
    )


def event_to_dict(event: TombstoneEvent) -> dict[str, str]:
    """Serialize a tombstone event for JSON output."""
    return {
        "event_id": event.event_id,
        "kind": event.kind,
        "target": event.target,
        "action": event.action.value,
        "reason": event.reason,
        "actor": event.actor,
        "at": event.at,
    }


def write_tombstone_event(
    config: OncaiConfig,
    *,
    kind: str,
    target: str,
    action: TombstoneAction,
    reason: str = "",
    actor: str | None = None,
    at: str | None = None,
    event_id: str | None = None,
) -> TombstoneEvent:
    """Append a tombstone event file to the inbox."""
    validate_tombstone_target(kind, target)
    directory = tombstone_dir(config)
    directory.mkdir(parents=True, exist_ok=True)

    resolved_event_id = event_id or secrets.token_hex(8)
    event = TombstoneEvent(
        event_id=resolved_event_id,
        kind=kind,
        target=target,
        action=action,
        reason=reason,
        actor=actor or getpass.getuser(),
        at=at or _utc_now(),
    )
    _parse_at(event.at)

    path = directory / f"{event.event_id}{TOMBSTONE_SUFFIX}"
    if path.exists():
        raise FileExistsError(f"Tombstone event already exists: {path}")
    path.write_text(json.dumps(event_to_dict(event), indent=2) + "\n")
    ensure_sidecar(path)
    return TombstoneEvent(
        event_id=event.event_id,
        kind=event.kind,
        target=event.target,
        action=event.action,
        reason=event.reason,
        actor=event.actor,
        at=event.at,
        path=path,
    )


def inbox_source_paths_for_target(
    config: OncaiConfig, kind: str, target: str
) -> list[Path]:
    """User-facing inbox paths associated with a tombstone target."""
    validate_tombstone_target(kind, target)
    inbox = config.inbox_path / kind
    if kind in {"fc_extractions", "fc_reviews"}:
        return [inbox / target]
    if kind == "cohorts":
        return [inbox / f"{target}.csv", inbox / f"{target}.sql"]
    raise ValueError(f"Unsupported tombstone kind: {kind}")


def lake_paths_for_target(config: OncaiConfig, kind: str, target: str) -> list[Path]:
    """Lake projection paths removed when a target is forgotten."""
    validate_tombstone_target(kind, target)
    lake = config.lake_path / kind
    if kind in {"fc_extractions", "fc_reviews"}:
        return [lake / f"{target}.parquet", lake / f"{target}.sql"]
    if kind == "cohorts":
        return [
            lake / f"{target}.parquet",
            lake / f"{target}.cohort.json",
            lake / f"{target}.sql",
        ]
    raise ValueError(f"Unsupported tombstone kind: {kind}")


def prune_lake_target(
    config: OncaiConfig, kind: str, target: str, *, dry_run: bool = False
) -> list[Path]:
    """Remove local lake projection files for a tombstoned source."""
    removed = [path for path in lake_paths_for_target(config, kind, target) if path.exists()]
    if dry_run:
        return removed
    for path in removed:
        path.unlink()
    return removed


def inbox_targets_for_kind(config: OncaiConfig, kind: str) -> set[str]:
    """Return source targets currently present in the inbox for ``kind``."""
    validate_tombstone_target(kind, "_probe")
    inbox = config.inbox_path / kind
    if not inbox.exists():
        return set()
    if kind == "fc_extractions":
        return {
            path.name
            for path in inbox.iterdir()
            if path.is_dir() and any(path.glob("*.jsonl"))
        }
    if kind == "fc_reviews":
        return {
            path.name
            for path in inbox.iterdir()
            if path.is_dir()
            and (
                any(path.glob("*.review_pkg.json"))
                or any(path.glob("*.reviews.jsonl"))
            )
        }
    if kind == "cohorts":
        return {path.stem for path in inbox.glob("*.csv")}
    raise ValueError(f"Unsupported tombstone kind: {kind}")


def lake_targets_for_kind(config: OncaiConfig, kind: str) -> set[str]:
    """Return targets currently represented by lake projection files."""
    validate_tombstone_target(kind, "_probe")
    lake = config.lake_path / kind
    if not lake.exists():
        return set()
    if kind in {"fc_extractions", "fc_reviews"}:
        paths = [*lake.glob("*.parquet"), *lake.glob("*.sql")]
        return {path.stem for path in paths}
    if kind == "cohorts":
        targets = {path.stem for path in lake.glob("*.parquet")}
        targets.update(path.stem for path in lake.glob("*.sql"))
        targets.update(
            path.name.removesuffix(".cohort.json")
            for path in lake.glob("*.cohort.json")
        )
        return targets
    raise ValueError(f"Unsupported tombstone kind: {kind}")
