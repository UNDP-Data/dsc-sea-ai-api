#!/usr/bin/env python3
"""Validate a copyable RAG assistant kit without writing to storage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.rag_system import AssistantKitError, validate_kit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", required=True, help="Path to assistant kit directory.")
    args = parser.parse_args()
    try:
        kit = validate_kit(Path(args.kit))
    except AssistantKitError as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "assistant_id": kit.profile.assistant_id,
                "display_name": kit.profile.display_name,
                "tables": kit.profile.table_names,
                "profile": str(kit.profile_path),
                "manifest": str(kit.manifest_path) if kit.manifest_path else None,
                "eval": str(kit.eval_path) if kit.eval_path else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
