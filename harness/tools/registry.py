"""The tool registry: loads tools.yaml, validates it, validates arguments.

Startup assertions, not runtime checks. A malformed tool definition should fail
`make test`, not a live call:

  * every side-effecting tool declares idempotency_fields, and they exist;
  * every requires_confirmation tool has a readback_template whose placeholders
    all resolve;
  * every parameters block is a well-formed object schema.

Argument validation is a small JSON-Schema subset implemented here rather than
a dependency: the schemas use type, required, enum, pattern, format=email,
minimum, maximum and maxLength, and nothing else.
"""

from __future__ import annotations

import functools
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DOMAINS_ROOT = Path(__file__).resolve().parent.parent.parent / "domains"

#: Placeholders a read-back template may use that are not tool parameters.
#: Resolved by harness/tools/readback.py from call state and order lookups.
DERIVED_PLACEHOLDERS = frozenset({"s", "event_name", "reason_label"})

_EMAIL = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")
_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,),
}


class RegistryError(ValueError):
    """tools.yaml is malformed. Raised at load time, never during a call."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    endpoint: str
    side_effecting: bool = False
    requires_verification: bool = False
    requires_confirmation: bool = False
    timeout_ms: int = 3000
    idempotency_fields: tuple[str, ...] = ()
    readback_template: str | None = None

    def schema(self) -> dict[str, Any]:
        """Provider-neutral schema for LLMRequest.tools."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def validate(self, args: dict[str, Any]) -> list[str]:
        """Return validation errors; empty means valid."""
        errors: list[str] = []
        properties = self.parameters.get("properties", {})
        for name in self.parameters.get("required", []):
            if name not in args or args[name] in (None, ""):
                errors.append(f"missing required argument {name!r}")
        for name, value in args.items():
            rules = properties.get(name)
            if rules is None:
                errors.append(f"unknown argument {name!r}")
                continue
            errors.extend(_check_value(name, value, rules))
        return errors


def _check_value(name: str, value: Any, rules: dict[str, Any]) -> list[str]:
    kind = rules.get("type")
    if kind and not (isinstance(value, _TYPES[kind]) and not (kind != "boolean" and isinstance(value, bool))):
        return [f"{name}: expected {kind}, got {type(value).__name__}"]
    errors = []
    if "enum" in rules and value not in rules["enum"]:
        errors.append(f"{name}: {value!r} is not one of {rules['enum']}")
    if "pattern" in rules and isinstance(value, str) and not re.fullmatch(rules["pattern"], value):
        errors.append(f"{name}: {value!r} does not match {rules['pattern']}")
    if rules.get("format") == "email" and isinstance(value, str) and not _EMAIL.match(value):
        errors.append(f"{name}: {value!r} is not an email address")
    if "minimum" in rules and isinstance(value, int | float) and value < rules["minimum"]:
        errors.append(f"{name}: {value} is below the minimum {rules['minimum']}")
    if "maximum" in rules and isinstance(value, int | float) and value > rules["maximum"]:
        errors.append(f"{name}: {value} is above the maximum {rules['maximum']}")
    if "maxLength" in rules and isinstance(value, str) and len(value) > rules["maxLength"]:
        errors.append(f"{name}: longer than {rules['maxLength']} characters")
    return errors


@dataclass
class Registry:
    tools: dict[str, ToolSpec] = field(default_factory=dict)

    def get(self, name: str) -> ToolSpec | None:
        return self.tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools.values()]

    @classmethod
    def empty(cls) -> Registry:
        return cls()


@functools.lru_cache(maxsize=8)
def load_registry(domain: str) -> Registry:
    path = DOMAINS_ROOT / domain / "tools.yaml"
    if not path.exists():
        return Registry.empty()
    return parse_registry(yaml.safe_load(path.read_text(encoding="utf-8")))


def parse_registry(raw: dict[str, Any]) -> Registry:
    registry = Registry()
    for entry in raw.get("tools", []):
        spec = ToolSpec(
            name=entry["name"],
            description=" ".join(str(entry.get("description", "")).split()),
            parameters=entry["parameters"],
            endpoint=entry["endpoint"],
            side_effecting=bool(entry.get("side_effecting", False)),
            requires_verification=bool(entry.get("requires_verification", False)),
            requires_confirmation=bool(entry.get("requires_confirmation", False)),
            timeout_ms=int(entry.get("timeout_ms", 3000)),
            idempotency_fields=tuple(entry.get("idempotency_fields", [])),
            readback_template=(
                " ".join(entry["readback_template"].split())
                if entry.get("readback_template") else None
            ),
        )
        _assert_valid(spec)
        if spec.name in registry.tools:
            raise RegistryError(f"duplicate tool {spec.name!r}")
        registry.tools[spec.name] = spec
    return registry


def _assert_valid(spec: ToolSpec) -> None:
    params = spec.parameters
    if params.get("type") != "object" or not isinstance(params.get("properties"), dict):
        raise RegistryError(f"{spec.name}: parameters must be an object schema with properties")
    properties = set(params["properties"])
    missing = set(params.get("required", [])) - properties
    if missing:
        raise RegistryError(f"{spec.name}: required fields not in properties: {sorted(missing)}")
    for name, rules in params["properties"].items():
        if rules.get("type") not in _TYPES:
            raise RegistryError(f"{spec.name}.{name}: unsupported type {rules.get('type')!r}")

    if spec.side_effecting:
        if not spec.idempotency_fields:
            raise RegistryError(
                f"{spec.name}: side-effecting tools must declare idempotency_fields"
            )
        unknown = set(spec.idempotency_fields) - properties
        if unknown:
            raise RegistryError(f"{spec.name}: idempotency_fields not in parameters: {sorted(unknown)}")

    if spec.requires_confirmation:
        if not spec.readback_template:
            raise RegistryError(f"{spec.name}: requires_confirmation needs a readback_template")
        placeholders = {
            field for _, field, _, _ in string.Formatter().parse(spec.readback_template) if field
        }
        unresolved = placeholders - properties - DERIVED_PLACEHOLDERS
        if unresolved:
            raise RegistryError(
                f"{spec.name}: readback placeholders cannot resolve: {sorted(unresolved)}"
            )
