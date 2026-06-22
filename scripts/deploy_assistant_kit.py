#!/usr/bin/env python3
"""Deploy a local assistant kit back into this backend repo.

This wrapper is intended to be called from another local repo after prototyping a
copyable assistant kit there. It validates the kit, installs profile/kit files
into this backend, and optionally imports the manifest into LanceDB.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.rag_system import AssistantKitError, validate_kit  # noqa: E402


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def _json_from_stdout(result: subprocess.CompletedProcess[str]) -> dict:
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"stdout": result.stdout.strip()}
    return payload if isinstance(payload, dict) else {"stdout": result.stdout.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kit",
        default="sgp_ai",
        help="Path to the assistant kit from the current working directory. Defaults to ./sgp_ai.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite installed backend copy when needed. Enabled by default for deploy-back workflow.",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Fail instead of overwriting existing installed files.",
    )
    parser.add_argument(
        "--import-corpus",
        action="store_true",
        help="Also import corpus manifest into assistant-specific LanceDB tables.",
    )
    parser.add_argument(
        "--include-chunks",
        action="store_true",
        help="When importing corpus, include chunk rows.",
    )
    parser.add_argument(
        "--profiles-dir",
        default=str(ROOT / "config" / "rag_profiles"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--kits-dir",
        default=str(ROOT / "assistant_kits"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    source_cwd = Path.cwd()
    kit_path = Path(args.kit).expanduser()
    if not kit_path.is_absolute():
        kit_path = source_cwd / kit_path
    kit_path = kit_path.resolve()

    try:
        kit = validate_kit(kit_path)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "install_assistant_kit.py"),
            "--kit",
            str(kit_path),
            "--profiles-dir",
            str(Path(args.profiles_dir).expanduser()),
            "--kits-dir",
            str(Path(args.kits_dir).expanduser()),
        ]
        if args.overwrite:
            command.append("--overwrite")
        if args.import_corpus:
            command.append("--import-corpus")
        if args.include_chunks:
            command.append("--include-chunks")
        result = _run(command)
    except (AssistantKitError, subprocess.CalledProcessError) as error:
        error_payload = {"ok": False, "error": str(error), "kit": str(kit_path), "backend_root": str(ROOT)}
        if isinstance(error, subprocess.CalledProcessError):
            error_payload["stdout"] = (error.stdout or "").strip()
            error_payload["stderr"] = (error.stderr or "").strip()
        print(json.dumps(error_payload, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    install_payload = _json_from_stdout(result)
    print(
        json.dumps(
            {
                "ok": True,
                "assistant_id": kit.profile.assistant_id,
                "source_repo_cwd": str(source_cwd),
                "source_kit": str(kit_path),
                "backend_root": str(ROOT),
                "backend_profile": str(Path(args.profiles_dir).expanduser() / f"{kit.profile.assistant_id}.yaml"),
                "backend_kit": str(Path(args.kits_dir).expanduser() / kit.profile.assistant_id),
                "routes": {
                    "model": f"/assistants/{kit.profile.assistant_id}/model",
                    "documents": f"/assistants/{kit.profile.assistant_id}/documents",
                    "sources": f"/assistants/{kit.profile.assistant_id}/sources",
                    "debug_retrieve": f"/assistants/{kit.profile.assistant_id}/debug/retrieve",
                },
                "tables": kit.profile.table_names,
                "imported_corpus": bool(args.import_corpus),
                "included_chunks": bool(args.import_corpus and args.include_chunks),
                "install": install_payload,
                "next_steps": [
                    "Run backend tests from the backend repo.",
                    f"Commit the installed profile and assistant_kits/{kit.profile.assistant_id} folder in the backend repo.",
                    "Merge to main to deploy through the existing Azure Web App workflow.",
                    "Continue prototyping in the source repo; rerun this command whenever the kit is ready to sync back.",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
