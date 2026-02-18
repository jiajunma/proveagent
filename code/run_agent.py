#!/usr/bin/env python3
"""
Command-line tool to run the IMO agent on a problem file.
Automatically generates log and memory files. If a solution is found,
converts it to LaTeX using res2md.py.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Run IMO agent on a problem file. Auto-generates log, memory, and tex (if solution found)."
    )
    parser.add_argument(
        "question_file",
        help="Path to the question/problem statement file",
    )
    parser.add_argument(
        "--log-dir",
        "-d",
        type=str,
        default="run_logs",
        help="Directory for log and memory files (default: run_logs)",
    )
    parser.add_argument(
        "--other-prompts",
        "-o",
        type=str,
        help="Other prompts for the agent (comma-separated)",
    )
    parser.add_argument(
        "--max-runs",
        "-m",
        type=int,
        default=10,
        help="Maximum number of agent runs (default: 10)",
    )
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Resume from existing memory file if present",
    )
    parser.add_argument(
        "--agent",
        "-a",
        type=str,
        default="agent.py",
        help="Agent script to run (default: agent.py)",
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    question_path = os.path.abspath(args.question_file)
    if not os.path.exists(question_path):
        print(f"Error: Question file not found: {question_path}", file=sys.stderr)
        sys.exit(1)

    # Derive base name from question file (without extension)
    question_basename = os.path.splitext(os.path.basename(question_path))[0]

    # Create log directory
    log_dir = os.path.abspath(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Generate log and memory file paths (no timestamp - we want consistent names for resume)
    log_file = os.path.join(log_dir, f"{question_basename}.log")
    memory_file = os.path.join(log_dir, f"{question_basename}.mem")

    # Build agent command
    agent_path = os.path.join(script_dir, args.agent)
    if not os.path.exists(agent_path):
        agent_path = args.agent  # Try as path directly

    cmd = [
        sys.executable,
        agent_path,
        question_path,
        "--log",
        log_file,
        "--memory",
        memory_file,
        "--max_runs",
        str(args.max_runs),
    ]

    if args.other_prompts:
        cmd.extend(["--other_prompts", args.other_prompts])

    if args.resume:
        cmd.append("--resume")

    print(f"Running agent on: {question_path}")
    print(f"Log file: {log_file}")
    print(f"Memory file: {memory_file}")
    print("-" * 50)

    # Run agent (cwd = script_dir so agent finds its modules)
    result = subprocess.run(
        cmd,
        cwd=script_dir,
    )

    # Check if solution was found
    solution_found = False
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            log_content = f.read()
            solution_found = "Found a correct solution in run" in log_content
    except Exception:
        pass

    if solution_found and result.returncode == 0:
        # Generate tex file using res2md.py
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tex_filename = f"{question_basename}-{timestamp}.tex"
        tex_file = os.path.join(log_dir, tex_filename)

        res2md_path = os.path.join(script_dir, "res2md.py")
        res2md_cmd = [
            sys.executable,
            res2md_path,
            memory_file,
            "--tex",
            tex_file,
        ]

        print(f"\nSolution found! Generating LaTeX: {tex_file}")
        tex_result = subprocess.run(res2md_cmd, cwd=script_dir)
        if tex_result.returncode == 0:
            print(f"TeX file written: {tex_file}")
        else:
            print(f"Warning: res2md failed to generate tex (exit code {tex_result.returncode})")
    else:
        if not solution_found:
            print("\nNo solution found in this run.")
        else:
            print(f"\nAgent exited with code {result.returncode}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
