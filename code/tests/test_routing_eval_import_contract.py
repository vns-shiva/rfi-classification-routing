"""Guards routing_eval.py's import-decoupling contract: it may depend on the
narrow set of modules an evaluation script legitimately needs to score
predictions against gold (schema, gating, routing_policy, vocabulary,
agreement_metrics), but never on policy_audit.py, stratification.py, or
templates.py. Those three build/audit the corpus itself; if routing_eval.py
imported any of them, a change to corpus generation could silently change
what "gold" means during scoring, which would defeat the anti-circularity
architecture the rest of this package depends on. Run with:

    python -m unittest discover -s code/tests -t code
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
_ROUTING_EVAL = _CODE_DIR / "routing_eval.py"

_ALLOWED_PROJECT_IMPORTS = {"schema", "gating", "routing_policy", "vocabulary", "agreement_metrics"}
_FORBIDDEN_PROJECT_IMPORTS = {"policy_audit", "stratification", "templates"}


def _imported_module_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module.split(".")[0])
    return names


def _project_module_names() -> set[str]:
    return {p.stem for p in _CODE_DIR.glob("*.py")}


class TestRoutingEvalImportContract(unittest.TestCase):
    def test_project_imports_are_a_subset_of_the_allowed_dependencies(self):
        imported = _imported_module_names(_ROUTING_EVAL)
        project_imports = imported & _project_module_names()
        self.assertTrue(
            project_imports.issubset(_ALLOWED_PROJECT_IMPORTS),
            f"routing_eval.py imports project modules outside its contract: "
            f"{project_imports - _ALLOWED_PROJECT_IMPORTS}",
        )

    def test_forbidden_corpus_building_modules_are_never_imported(self):
        imported = _imported_module_names(_ROUTING_EVAL)
        self.assertEqual(set(), imported & _FORBIDDEN_PROJECT_IMPORTS)


if __name__ == "__main__":
    unittest.main()
