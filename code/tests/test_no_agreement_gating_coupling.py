"""Guards the intended separation between gating.py (a pipeline-time filter
applied to one label set) and annotator_agreement.py (a corpus-comparison
tool run across two label sets): neither should import the other. If gating
started depending on agreement internals (or vice versa), gating decisions
could silently start varying based on which other annotator's labels happen
to be on disk, which would defeat the "one label set in, filtered labels
out" contract gating.py's own docstring promises. Run with:

    python -m unittest discover -s code/tests -t code
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
_GATING = _CODE_DIR / "gating.py"
_AGREEMENT = _CODE_DIR / "annotator_agreement.py"


def _imported_module_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module.split(".")[0])
    return names


class TestGatingAgreementModulesAreDecoupled(unittest.TestCase):
    def test_gating_does_not_import_annotator_agreement(self):
        self.assertNotIn("annotator_agreement", _imported_module_names(_GATING))

    def test_annotator_agreement_does_not_import_gating(self):
        self.assertNotIn("gating", _imported_module_names(_AGREEMENT))


if __name__ == "__main__":
    unittest.main()
