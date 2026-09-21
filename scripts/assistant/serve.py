"""CLI runner for the PNTC AI Assistant server.

Usage:
    python scripts/assistant/serve.py [--host 127.0.0.1] [--port 8000]
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir / "src"))

from xmvad.assistant.server import run_server


def main():
    parser = argparse.ArgumentParser(description="Run PNTC AI Assistant HTTP Server & Dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
