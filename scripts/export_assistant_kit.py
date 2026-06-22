#!/usr/bin/env python3
"""Copy an assistant kit to another local repo or folder for development."""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.rag_system import AssistantKitError, validate_kit  # noqa: E402


def _copytree_merge(source: Path, target: Path, *, overwrite: bool) -> list[str]:
    actions: list[str] = []
    for source_path in sorted(source.rglob("*")):
        relative = source_path.relative_to(source)
        target_path = target / relative
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            if filecmp.cmp(source_path, target_path, shallow=False):
                actions.append(f"unchanged {relative}")
                continue
            if not overwrite:
                raise AssistantKitError(f"Refusing to overwrite existing file without --overwrite: {target_path}")
        shutil.copy2(source_path, target_path)
        actions.append(f"copied {relative}")
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kit",
        default=str(ROOT / "assistant_kits" / "_template"),
        help="Source kit folder to copy. Defaults to assistant_kits/_template.",
    )
    parser.add_argument("--target", required=True, help="Destination folder in the other repo/local workspace.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing destination files.")
    args = parser.parse_args()

    source = Path(args.kit).expanduser().resolve()
    target = Path(args.target).expanduser().resolve()
    try:
        kit = validate_kit(source)
        actions = _copytree_merge(source, target, overwrite=args.overwrite)
    except AssistantKitError as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "assistant_id": kit.profile.assistant_id,
                "source": str(source),
                "target": str(target),
                "files": actions,
                "next_steps": [
                    "Open the target folder in the other repo and edit assistant.yaml plus corpus/manifest.yaml.",
                    "Keep the folder shape unchanged so install_assistant_kit.py can bring it back later.",
                    "After refinement, run scripts/install_assistant_kit.py --kit <target> from this backend repo.",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
