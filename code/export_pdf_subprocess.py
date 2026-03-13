"""
Standalone subprocess entry point for PDF export.
Called by mcp_server.py's export_pdf() tool.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="")
    parser.add_argument("--solution", default="")
    parser.add_argument("--verify", default="")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--base-name", required=True)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    from latex_pipeline import export_to_pdf

    api_key = ""
    if args.provider in ("openai", "gpt"):
        api_key = os.environ.get("OPENAI_API_KEY", "")
    elif args.provider in ("kimi", "moonshot"):
        api_key = os.environ.get("KIMI_API_KEY", "")
    else:
        api_key = os.environ.get("GOOGLE_API_KEY", "")

    ok, tex = export_to_pdf(
        args.problem,
        args.solution,
        args.verify,
        args.log_dir,
        args.base_name,
        api_key,
        provider=args.provider,
        model_name=args.model or None,
    )

    if ok:
        print(f"PDF export successful.")
        sys.exit(0)
    else:
        print(f"PDF export failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
