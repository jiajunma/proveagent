"""
Standalone subprocess entry point for the proof agent.
Called by mcp_server.py's _run_job() via subprocess.Popen.
All output goes to stdout (captured to log file by the caller).
"""

import argparse
import multiprocessing
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--mem", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--base-name", required=True)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Load extra prompts from mem file if resuming
    other_prompts = []
    verify_prompts = []
    if os.path.exists(args.mem):
        import json
        try:
            with open(args.mem, "r", encoding="utf-8") as f:
                mem = json.load(f)
            other_prompts = mem.get("other_prompts") or []
            verify_prompts = mem.get("verify_comments") or mem.get("verify_prompts") or []
        except Exception as e:
            print(f"Warning: could not read mem file: {e}")

    from agent_worker import run_agent_worker

    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=run_agent_worker,
        kwargs=dict(
            problem_statement=args.problem,
            other_prompts=other_prompts,
            verify_prompts=verify_prompts,
            memory_file=args.mem,
            resume=args.resume,
            log_dir=args.log_dir,
            base_name=args.base_name,
            result_queue=result_queue,
            streaming=True,
            show_thinking=True,
            interactive=False,
            provider_name=args.provider,
            model_name=args.model or None,
        ),
    )
    p.start()
    p.join()

    status, result = result_queue.get() if not result_queue.empty() else ("error", "no result")
    if status == "ok":
        print(f"\n[MCP] Proof complete.")
        sys.exit(0)
    else:
        print(f"\n[MCP] Proof failed: {result}")
        sys.exit(1)


if __name__ == "__main__":
    main()
