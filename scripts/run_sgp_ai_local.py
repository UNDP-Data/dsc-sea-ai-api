#!/usr/bin/env python3
"""Run the SGP AI backend and tester UI with automatic local port fallback."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def _is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _find_free_port(host: str, preferred: int, *, attempts: int = 50) -> int:
    for port in range(preferred, preferred + attempts):
        if _is_port_free(host, port):
            return port
    raise RuntimeError(
        f"No free local port found from {preferred} to {preferred + attempts - 1}."
    )


def _http_ok(url: str, *, api_key: str | None = None, timeout: float = 2.0) -> bool:
    request = urllib.request.Request(url=url, method="GET")
    if api_key:
        request.add_header("X-Api-Key", api_key)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 500
    except (OSError, urllib.error.URLError):
        return False


def _wait_for_http(
    url: str,
    *,
    api_key: str | None = None,
    timeout_seconds: float = 45.0,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _http_ok(url, api_key=api_key):
            return True
        time.sleep(0.5)
    return False


def _start_process(command: list[str], *, env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(command, cwd=ROOT, env=env)


def _backend_is_reusable(base_url: str, api_key: str | None) -> bool:
    return _http_ok(f"{base_url.rstrip('/')}/assistants", api_key=api_key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=8016)
    parser.add_argument("--tester-port", type=int, default=8015)
    parser.add_argument(
        "--backend-base-url",
        default="",
        help="Use an already-running backend instead of starting one.",
    )
    parser.add_argument(
        "--no-reuse-backend",
        action="store_true",
        help="Start a new backend even if the requested backend port is already serving the API.",
    )
    parser.add_argument("--reload", action="store_true", help="Run both uvicorn apps with --reload.")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    env = os.environ.copy()
    api_key = (env.get("API_KEY") or "").strip()
    processes: list[subprocess.Popen] = []

    backend_base_url = args.backend_base_url.strip().rstrip("/")
    if backend_base_url:
        print(f"Using backend: {backend_base_url}", flush=True)
    else:
        preferred_backend_url = f"http://{args.host}:{args.backend_port}"
        if (
            not args.no_reuse_backend
            and not _is_port_free(args.host, args.backend_port)
            and _backend_is_reusable(preferred_backend_url, api_key)
        ):
            backend_base_url = preferred_backend_url
            print(f"Using existing backend: {backend_base_url}", flush=True)
        else:
            backend_port = _find_free_port(args.host, args.backend_port)
            backend_base_url = f"http://{args.host}:{backend_port}"
            backend_command = [
                sys.executable,
                "-m",
                "uvicorn",
                "main:app",
                "--host",
                args.host,
                "--port",
                str(backend_port),
            ]
            if args.reload:
                backend_command.append("--reload")
            print(f"Starting backend: {backend_base_url}", flush=True)
            processes.append(_start_process(backend_command, env=env))
            if not _wait_for_http(
                f"{backend_base_url}/assistants",
                api_key=api_key,
                timeout_seconds=60.0,
            ):
                print("Backend did not become ready within 60 seconds.", file=sys.stderr)

    tester_port = _find_free_port(args.host, args.tester_port)
    tester_base_url = f"http://{args.host}:{tester_port}"
    tester_env = {
        **env,
        "SGP_TESTER_API_BASE_URL": backend_base_url,
        "SGP_TESTER_BACKEND_BASE_URL": backend_base_url,
    }
    tester_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "frontend.sgp_ai_tester_app:app",
        "--host",
        args.host,
        "--port",
        str(tester_port),
    ]
    if args.reload:
        tester_command.append("--reload")
    print(f"Starting tester UI: {tester_base_url}/sgp-ai-tester", flush=True)
    processes.append(_start_process(tester_command, env=tester_env))
    _wait_for_http(
        f"{tester_base_url}/sgp-ai-tester/api/status",
        timeout_seconds=30.0,
    )

    print("", flush=True)
    print("SGP AI local stack is running.", flush=True)
    print(f"Backend:   {backend_base_url}", flush=True)
    print(f"Tester UI: {tester_base_url}/sgp-ai-tester", flush=True)
    print("Press Ctrl+C to stop processes started by this launcher.", flush=True)

    try:
        while True:
            for process in processes:
                if process.poll() is not None:
                    return int(process.returncode or 0)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping local SGP AI stack...", flush=True)
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
