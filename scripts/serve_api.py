"""Run PNTC Inspect FastAPI Backend Server.

Usage:
    python scripts/serve_api.py [--host 127.0.0.1] [--port 8000] [--reload]
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(root_dir / ".env")
except ImportError:
    pass

import uvicorn
from xmvad.api.database import init_db


def main():
    parser = argparse.ArgumentParser(description="Start PNTC Inspect FastAPI Backend")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable live auto-reload")
    args = parser.parse_args()

    init_db()
    print(f"Starting PNTC Inspect backend at http://{args.host}:{args.port}")
    uvicorn.run("xmvad.api.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
