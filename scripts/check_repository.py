"""Vérifications simples avant une publication sur GitHub."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DIRECTORIES = (
    "configs",
    "data",
    "demo",
    "docs",
    "notebooks",
    "report",
    "results",
    "src",
)
MAX_FILE_SIZE = 50 * 1024 * 1024


def main() -> None:
    errors = []

    for directory in REQUIRED_DIRECTORIES:
        if not (ROOT / directory).is_dir():
            errors.append(f"Dossier manquant : {directory}")

    for notebook in (ROOT / "notebooks").glob("*.ipynb"):
        try:
            content = json.loads(notebook.read_text(encoding="utf-8"))
            if content.get("nbformat") != 4:
                errors.append(f"Notebook non compatible : {notebook}")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"Notebook invalide : {notebook} ({error})")

    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            if path.stat().st_size > MAX_FILE_SIZE:
                errors.append(f"Fichier trop volumineux : {path.relative_to(ROOT)}")

    if errors:
        for error in errors:
            print(f"ERREUR: {error}")
        raise SystemExit(1)

    print("Vérification du dépôt réussie.")


if __name__ == "__main__":
    main()
