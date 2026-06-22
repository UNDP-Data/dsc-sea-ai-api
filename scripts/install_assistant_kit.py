#!/usr/bin/env python3
"""Install a copyable RAG assistant kit into this backend."""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.rag_system import AssistantKitError, validate_kit  # noqa: E402


def _copy_file(source: Path, target: Path, *, overwrite: bool) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if filecmp.cmp(source, target, shallow=False):
            return "unchanged"
        if not overwrite:
            raise AssistantKitError(f"Refusing to overwrite existing file without --overwrite: {target}")
    shutil.copy2(source, target)
    return "copied"


def _copy_kit_dir(source: Path, target: Path, *, overwrite: bool) -> str:
    source = source.resolve()
    target = target.resolve()
    if source == target:
        return "already-in-repo"
    if target.exists():
        if not overwrite:
            raise AssistantKitError(f"Refusing to overwrite existing kit without --overwrite: {target}")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return "copied"


def _run_import(manifest: Path, assistant_id: str, *, include_chunks: bool) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "import_corpus_manifest.py"),
        "--manifest",
        str(manifest),
        "--assistant-id",
        assistant_id,
    ]
    if include_chunks:
        command.append("--include-chunks")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", required=True, help="Path to assistant kit directory.")
    parser.add_argument(
        "--profiles-dir",
        default=str(ROOT / "config" / "rag_profiles"),
        help="Directory where assistant profiles are installed.",
    )
    parser.add_argument(
        "--kits-dir",
        default=str(ROOT / "assistant_kits"),
        help="Directory where kit files are copied for versioning.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing installed profile/kit files.")
    parser.add_argument("--import-corpus", action="store_true", help="Import the kit corpus manifest into LanceDB.")
    parser.add_argument("--include-chunks", action="store_true", help="When importing corpus, also upsert chunks.")
    args = parser.parse_args()

    try:
        kit = validate_kit(Path(args.kit))
        assistant_id = kit.profile.assistant_id
        profile_target = Path(args.profiles_dir) / f"{assistant_id}.yaml"
        kit_target = Path(args.kits_dir) / assistant_id
        profile_action = _copy_file(kit.profile_path, profile_target, overwrite=args.overwrite)
        kit_action = _copy_kit_dir(kit.path, kit_target, overwrite=args.overwrite)
        imported = False
        if args.import_corpus:
            if kit.manifest_path is None:
                raise AssistantKitError("Cannot import corpus: kit has no corpus/manifest.yaml.")
            manifest_for_import = kit_target / "corpus" / "manifest.yaml" if kit_target.exists() else kit.manifest_path
            _run_import(manifest_for_import, assistant_id, include_chunks=args.include_chunks)
            imported = True
    except (AssistantKitError, subprocess.CalledProcessError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "assistant_id": assistant_id,
                "profile": {"target": str(profile_target), "action": profile_action},
                "kit": {"target": str(kit_target), "action": kit_action},
                "tables": kit.profile.table_names,
                "imported_corpus": imported,
                "included_chunks": bool(args.import_corpus and args.include_chunks),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
