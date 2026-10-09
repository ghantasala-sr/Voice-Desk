"""W1-2 — the architectural contract, as a test.

`make test` also runs `lint-imports`, which enforces the same rule from
pyproject.toml. This file is the backstop: it needs no extra tooling, so the
rule holds even if import-linter is not installed.

The rule: the harness must run without an audio stack. If `harness/` can
import pipecat, `eval/text_driver.py` stops being possible, and every fast
test in this suite goes with it.

Verify it works: add `import pipecat` to any file under harness/, watch this
fail, then revert.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

CONTRACTS = {
    "harness": {"pipecat", "app", "eval"},
    "retrieval": {"pipecat", "app", "eval"},
    "providers": {"pipecat", "app", "eval", "harness"},
    "core": {"pipecat", "app", "eval", "harness", "providers", "retrieval", "analytics"},
}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("package,forbidden", sorted(CONTRACTS.items()))
def test_package_does_not_import(package: str, forbidden: set[str]) -> None:
    directory = ROOT / package
    if not directory.exists():
        pytest.skip(f"{package}/ does not exist yet")

    violations: list[str] = []
    for file in sorted(directory.rglob("*.py")):
        for imported in _imported_roots(file) & forbidden:
            violations.append(f"{file.relative_to(ROOT)} imports {imported!r}")

    assert not violations, (
        f"{package}/ must not import {sorted(forbidden)}:\n  " + "\n  ".join(violations)
    )
