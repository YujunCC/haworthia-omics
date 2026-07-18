"""Start the local API and Streamlit interface as one application."""

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"


def available_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((HOST, port))
        except OSError:
            return False
    return True


def wait_for_service(url, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Service exited before becoming ready (code {process.returncode})."
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.4)
    raise TimeoutError(f"Timed out waiting for {url}")


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Haworthia OMICS locally.")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--frontend-port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Start both services, verify readiness, and exit without opening a browser.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for port in (args.backend_port, args.frontend_port):
        if not 1 <= port <= 65535:
            raise SystemExit(f"Invalid port: {port}")
    if args.backend_port == args.frontend_port:
        raise SystemExit("Backend and frontend ports must be different.")
    occupied = [
        str(port)
        for port in (args.backend_port, args.frontend_port)
        if not available_port(port)
    ]
    if occupied:
        raise SystemExit(
            "Port(s) already in use: " + ", ".join(occupied)
            + ". Close the existing service or choose different ports."
        )

    api_url = f"http://{HOST}:{args.backend_port}/api"
    frontend_url = f"http://{HOST}:{args.frontend_port}"
    environment = os.environ.copy()
    environment["HAWORTHIA_API_URL"] = api_url
    environment.setdefault("PYTHONUNBUFFERED", "1")

    backend = None
    frontend = None
    try:
        print("Starting the local computation engine...")
        backend = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "main:app",
                "--host",
                HOST,
                "--port",
                str(args.backend_port),
            ],
            cwd=ROOT,
            env=environment,
        )
        wait_for_service(f"{api_url}/overview", backend, timeout=90)

        print("Starting the local user interface...")
        frontend = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.address",
                HOST,
                "--server.port",
                str(args.frontend_port),
                "--server.headless",
                "true",
            ],
            cwd=ROOT,
            env=environment,
        )
        wait_for_service(f"{frontend_url}/_stcore/health", frontend, timeout=90)

        print(f"Haworthia OMICS is ready: {frontend_url}")
        if args.smoke_test:
            print("Smoke test passed.")
            return
        print("Keep this window open. Press Ctrl+C to stop both services.")
        if not args.no_browser:
            webbrowser.open(frontend_url)

        while backend.poll() is None and frontend.poll() is None:
            time.sleep(0.5)
        failed = backend if backend.poll() is not None else frontend
        raise RuntimeError(f"A service stopped unexpectedly (code {failed.returncode}).")
    except KeyboardInterrupt:
        print("Stopping Haworthia OMICS...")
    finally:
        stop_process(frontend)
        stop_process(backend)


if __name__ == "__main__":
    main()
