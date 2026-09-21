"""Interactive command-line conversational assistant for PNTC inspection results.

Usage:
    python scripts/assistant/chat_cli.py [--sample sample_001] [--provider gemini] [--mode TECHNICAL]
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir / "src"))

from xmvad.assistant import (
    PNTCAssistant,
    ResponseMode,
    create_provider,
)
from xmvad.assistant.server import CANONICAL_SAMPLES


def main():
    parser = argparse.ArgumentParser(description="Chat with PNTC AI Assistant in terminal")
    parser.add_argument("--sample", type=str, default="sample_001", help="Sample ID to inspect")
    parser.add_argument("--provider", type=str, default=None, help="gemini or grok")
    parser.add_argument("--mode", type=str, default="TECHNICAL", choices=["SIMPLE", "TECHNICAL", "VIVA"])
    args = parser.parse_args()

    prov = create_provider(args.provider) if args.provider else None
    assistant = PNTCAssistant(provider=prov)

    sample_data = CANONICAL_SAMPLES.get(args.sample, CANONICAL_SAMPLES["sample_001"])
    mode = ResponseMode(args.mode)

    print("=" * 60)
    print(f"PNTC AI ASSISTANT CLI — [{assistant.provider.provider_name.upper()}] — MODE: {mode.value}")
    print(f"Inspecting sample: {args.sample}")
    print("Type 'exit' or 'quit' to end. Type 'explain' for full summary.")
    print("=" * 60)

    # Print default explanation
    summary = assistant.generate_inspection_summary(sample_data, response_mode=mode)
    print(f"\n[ASSISTANT INITIAL SUMMARY]:\n{summary.message}\n")

    conv_id = f"cli_{args.sample}"

    while True:
        try:
            user_input = input("\nYOU > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting PNTC Assistant.")
                break

            if user_input.lower() == "explain":
                resp = assistant.generate_inspection_summary(sample_data, response_mode=mode, force_refresh=True)
            else:
                resp = assistant.chat(
                    sample_id=args.sample,
                    message=user_input,
                    conversation_id=conv_id,
                    inspection_result=sample_data,
                    response_mode=mode,
                )

            print(f"\nASSISTANT [{resp.model}] >\n{resp.message}")
            if resp.ui_action:
                print(f"  [⚡ UI ACTION EMITTED]: {resp.ui_action.type} (Defect: {resp.ui_action.defect_id})")
            if resp.warnings:
                print(f"  [⚠️ WARNINGS]: {resp.warnings}")

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


if __name__ == "__main__":
    main()
