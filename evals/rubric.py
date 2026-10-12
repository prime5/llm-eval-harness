"""
Rubric loader for the multi-turn conversation judge.

A rubric is a YAML file (see rubrics/conversation_v1.yaml) that gives each
dimension an anchor for every score on the scale 0-2. `load_rubric` validates it
against rubrics/rubric.schema.json and records the SHA-256 of the file bytes, so
every judge result can say exactly which rubric text produced it.

Duplicate keys are rejected: PyYAML would otherwise keep the last one silently,
which lets a copy-pasted dimension or anchor overwrite the real one.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "rubrics" / "rubric.schema.json"
SCALE = (0, 1, 2)


class RubricError(ValueError):
    """The rubric file is missing, unparsable or does not match the schema."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise RubricError(f"duplicate key {key!r} (line {key_node.start_mark.line + 1})")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True)
class Rubric:
    version: str
    dimensions: dict[str, dict[int, str]]
    sha256: str
    rules: tuple[str, ...] = ()
    scale: tuple[int, ...] = SCALE
    path: str = field(default="", compare=False)

    @property
    def dimension_ids(self) -> list[str]:
        return list(self.dimensions)

    def validate_scores(self, scores: Any) -> dict[str, int]:
        """Check a judge or human score dict: exactly the rubric's dimensions,
        each an int in the scale (bool is rejected). Returns a clean copy."""
        if not isinstance(scores, dict):
            raise RubricError(f"scores must be an object, got {type(scores).__name__}")
        missing = [d for d in self.dimensions if d not in scores]
        extra = [d for d in scores if d not in self.dimensions]
        if missing or extra:
            raise RubricError(f"scores must cover exactly the rubric dimensions; "
                              f"missing={missing} extra={extra}")
        clean = {}
        for dim, value in scores.items():
            if isinstance(value, bool) or not isinstance(value, int) or value not in self.scale:
                raise RubricError(f"score for {dim!r} must be one of {list(self.scale)}, got {value!r}")
            clean[dim] = value
        return clean

    def render(self) -> str:
        """The rubric as plain text for a judge prompt."""
        lines = []
        for dim, anchors in self.dimensions.items():
            lines.append(f"{dim}:")
            lines.extend(f"  {score}: {text}" for score, text in anchors.items())
        if self.rules:
            lines.append("Rules:")
            lines.extend(f"  - {r}" for r in self.rules)
        return "\n".join(lines)


def _stringify_keys(obj):
    """YAML reads `0:` as an int key; the JSON schema wants string keys."""
    if isinstance(obj, dict):
        return {str(k): _stringify_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_keys(v) for v in obj]
    return obj


def load_rubric(path: str | Path, schema_path: str | Path = SCHEMA_PATH) -> Rubric:
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RubricError(f"cannot read rubric {path}: {exc}") from exc

    try:
        data = yaml.load(raw, Loader=_UniqueKeyLoader)
    except RubricError as exc:
        raise RubricError(f"{path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise RubricError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise RubricError(f"{path}: top level must be a mapping")

    data = _stringify_keys(data)
    schema = json.loads(Path(schema_path).read_text())
    errors = sorted(Draft202012Validator(schema).iter_errors(data), key=lambda e: list(e.absolute_path))
    if errors:
        details = "; ".join(
            f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors[:5]
        )
        raise RubricError(f"{path}: schema violation: {details}")

    return Rubric(
        version=data["version"],
        dimensions={d: {int(s): t for s, t in anchors.items()} for d, anchors in data["dimensions"].items()},
        rules=tuple(data.get("rules", [])),
        sha256=hashlib.sha256(raw).hexdigest(),
        path=str(path),
    )
