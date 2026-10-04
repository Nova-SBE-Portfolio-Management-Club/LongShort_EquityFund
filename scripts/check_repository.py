"""Check Python/notebook syntax and local Markdown links without services."""

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    for folder in ("src", "scripts", "tests"):
        for path in (ROOT / folder).rglob("*.py"):
            try:
                ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
            except SyntaxError as error:
                errors.append(str(error))
    notebooks = list((ROOT / "src").rglob("*.ipynb"))
    for path in notebooks:
        document = json.loads(path.read_text())
        if document.get("nbformat") != 4 or "cells" not in document:
            errors.append(f"Invalid notebook structure: {path.relative_to(ROOT)}")
        for number, cell in enumerate(document.get("cells", []), start=1):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                source = re.sub(r"(?m)^\s*%matplotlib[^\n]*", "", source)
                try:
                    ast.parse(source, filename=f"{path.relative_to(ROOT)}:cell {number}")
                except SyntaxError as error:
                    errors.append(str(error))
    documents = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    documents.extend((ROOT / "docs").rglob("*.md"))
    documents.extend((ROOT / "src").rglob("*.md"))
    for path in documents:
        content = path.read_text()
        if len(re.findall(r"^```", content, re.MULTILINE)) % 2:
            errors.append(f"Unclosed code fence: {path.relative_to(ROOT)}")
        for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", content):
            if "://" in target or target.startswith("mailto:"):
                continue
            target = target.split("#")[0]
            if target and not (path.parent / target).exists():
                errors.append(f"Broken link in {path.relative_to(ROOT)}: {target}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Python syntax, {len(notebooks)} notebooks, and {len(documents)} documents passed.")


if __name__ == "__main__":
    main()
