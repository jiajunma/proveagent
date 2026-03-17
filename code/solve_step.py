#!/usr/bin/env python3
"""
Single-step solver for conversational proof workflow.

Does ONE solve or verify call and outputs result to stdout as JSON.
Designed to be called by OpenClaw agent orchestration.

Usage:
  # Solve a problem
  python3 solve_step.py solve --problem-file problems/exercise_4_9.txt --model gemini-2.5-pro --thinking-budget 131072

  # Solve with hints
  python3 solve_step.py solve --problem-file problems/exercise_4_9.txt --hint "Try induction" --hint "Consider the orbit structure"

  # Improve a solution given a bug report
  python3 solve_step.py improve --problem-file problems/exercise_4_9.txt --solution-file run_logs/exercise_4_9_iter0_solution.md --bug-report-file run_logs/exercise_4_9_iter0_validation.md

  # Verify a solution
  python3 solve_step.py verify --problem-file problems/exercise_4_9.txt --solution-file run_logs/exercise_4_9_iter0_solution.md

  # Solve from stdin (problem text piped in)
  echo "Prove that..." | python3 solve_step.py solve --model gemini-2.5-pro
"""

import argparse
import json
import os
import sys
import subprocess
from datetime import datetime

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

WORKSPACE = "/home/hoxideclaw/.openclaw/workspace-waverider"
PROVEAGENT_DIR = os.path.join(WORKSPACE, "proveagent")
RUN_LOGS_DIR = os.path.join(PROVEAGENT_DIR, "run_logs")


def load_api_key():
    """Load GOOGLE_API_KEY."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return api_key
    try:
        result = subprocess.run(
            ["bash", "-c", "source ~/.profile && echo $GOOGLE_API_KEY"],
            capture_output=True, text=True, timeout=5
        )
        api_key = result.stdout.strip()
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            return api_key
    except Exception:
        pass
    return None


def solve(problem_text, model_name="gemini-2.5-pro", thinking_budget=131072, hints=None):
    """Generate a solution for the problem."""
    from model_providers import GeminiProvider
    import prompts

    api_key = load_api_key()
    if not api_key:
        return {"error": "GOOGLE_API_KEY not found"}

    provider = GeminiProvider(api_key, model_name)
    
    # Check capabilities (enables streaming + thinking)
    provider.check_capabilities()

    # Build payload with custom thinking budget
    other_prompts = hints or []
    payload = provider.build_request_payload(
        system_prompt=prompts.STEP1_SYSTEM_PROMPT,
        question_prompt=problem_text,
        other_prompts=other_prompts if other_prompts else None,
        enable_thinking=True,
        streaming=False  # No streaming for subprocess usage
    )

    # Override thinking budget
    if "generationConfig" not in payload:
        payload["generationConfig"] = {}
    payload["generationConfig"]["thinkingConfig"] = {
        "thinkingBudget": thinking_budget
    }

    # Send request (non-streaming for clean output)
    print(f"Calling {model_name} with thinking budget {thinking_budget}...", file=sys.stderr)
    response_data = provider.send_api_request(payload, streaming=False, show_thinking=False)
    text, thinking = provider.extract_text_from_response(response_data)

    return {
        "solution": text,
        "thinking_length": len(thinking),
        "model": model_name,
        "thinking_budget": thinking_budget,
        "hints_used": len(other_prompts),
        "timestamp": datetime.now().isoformat()
    }


def improve(problem_text, current_solution, bug_report, model_name="gemini-2.5-pro",
            thinking_budget=131072, hints=None):
    """Improve a solution based on a bug report."""
    from model_providers import GeminiProvider
    import prompts

    api_key = load_api_key()
    if not api_key:
        return {"error": "GOOGLE_API_KEY not found"}

    provider = GeminiProvider(api_key, model_name)
    provider.check_capabilities()

    # Build correction prompt
    correction_prompt = prompts.build_correction_prompt(
        problem_text, current_solution, bug_report
    )

    other_prompts = list(hints or [])
    other_prompts.append(prompts.SELF_IMPROVEMENT_PROMPT)

    payload = provider.build_request_payload(
        system_prompt=prompts.STEP1_SYSTEM_PROMPT,
        question_prompt=correction_prompt,
        other_prompts=other_prompts,
        enable_thinking=True,
        streaming=False
    )

    # Override thinking budget
    if "generationConfig" not in payload:
        payload["generationConfig"] = {}
    payload["generationConfig"]["thinkingConfig"] = {
        "thinkingBudget": thinking_budget
    }

    print(f"Improving with {model_name}, thinking budget {thinking_budget}...", file=sys.stderr)
    response_data = provider.send_api_request(payload, streaming=False, show_thinking=False)
    text, thinking = provider.extract_text_from_response(response_data)

    return {
        "solution": text,
        "thinking_length": len(thinking),
        "model": model_name,
        "thinking_budget": thinking_budget,
        "action": "improve",
        "timestamp": datetime.now().isoformat()
    }


def verify(problem_text, solution_text, model_name="gemini-2.5-pro", thinking_budget=65536):
    """Verify a solution and return validation result."""
    from model_providers import GeminiProvider
    import prompts

    api_key = load_api_key()
    if not api_key:
        return {"error": "GOOGLE_API_KEY not found"}

    provider = GeminiProvider(api_key, model_name)
    provider.check_capabilities()

    # Build verification prompt
    verification_prompt = prompts.build_verification_prompt(problem_text, solution_text)

    payload = provider.build_request_payload(
        system_prompt=prompts.VERIFICATION_SYSTEM_PROMPT,
        question_prompt=verification_prompt,
        other_prompts=None,
        enable_thinking=True,
        streaming=False
    )

    # Override thinking budget (verification needs less than solving)
    if "generationConfig" not in payload:
        payload["generationConfig"] = {}
    payload["generationConfig"]["thinkingConfig"] = {
        "thinkingBudget": thinking_budget
    }

    print(f"Verifying with {model_name}, thinking budget {thinking_budget}...", file=sys.stderr)
    response_data = provider.send_api_request(payload, streaming=False, show_thinking=False)
    verification_text, thinking = provider.extract_text_from_response(response_data)

    # Check if verification passed
    check_prompt = prompts.build_verification_check_prompt(verification_text)
    check_payload = provider.build_request_payload(
        system_prompt="",
        question_prompt=check_prompt,
        enable_thinking=False,
        streaming=False
    )
    # Remove thinking config for simple check
    if "generationConfig" in check_payload and "thinkingConfig" in check_payload["generationConfig"]:
        del check_payload["generationConfig"]["thinkingConfig"]
    
    check_response = provider.send_api_request(check_payload, streaming=False, show_thinking=False)
    check_text, _ = provider.extract_text_from_response(check_response)
    passed = "yes" in check_text.lower()

    return {
        "verification": verification_text,
        "passed": passed,
        "thinking_length": len(thinking),
        "model": model_name,
        "action": "verify",
        "timestamp": datetime.now().isoformat()
    }


def save_validation(job_name, iteration, validation_text, passed, output_dir=None):
    """Save validation result to standard file format."""
    output_dir = output_dir or RUN_LOGS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{job_name}_iter{iteration}_validation.md"
    filepath = os.path.join(output_dir, filename)
    
    status = "PASSED" if passed else "FAILED"
    content = f"""# Validation (Iteration {iteration})

**Generated:** {datetime.now().isoformat()}
**Status:** {status}

## Verification Result

{validation_text}
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    return filepath


def save_solution(job_name, iteration, solution_text, output_dir=None):
    """Save solution to standard file format."""
    output_dir = output_dir or RUN_LOGS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{job_name}_iter{iteration}_solution.md"
    filepath = os.path.join(output_dir, filename)
    
    content = f"""# Solution (Iteration {iteration})

**Generated:** {datetime.now().isoformat()}

## Solution

{solution_text}
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Single-step proof solver")
    parser.add_argument("action", choices=["solve", "improve", "verify"],
                       help="Action to perform")
    parser.add_argument("--problem-file", "-p", help="Problem file path")
    parser.add_argument("--problem-text", help="Problem text (alternative to file)")
    parser.add_argument("--solution-file", "-s", help="Current solution file (for improve)")
    parser.add_argument("--bug-report-file", "-b", help="Bug report file (for improve)")
    parser.add_argument("--model", "-m", default="gemini-2.5-pro",
                       help="Model to use (default: gemini-2.5-pro)")
    parser.add_argument("--thinking-budget", "-t", type=int, default=131072,
                       help="Thinking budget (default: 131072)")
    parser.add_argument("--hint", "-H", action="append", default=[],
                       help="Hints to guide the solver")
    parser.add_argument("--job-name", "-j", help="Job name for saving files")
    parser.add_argument("--iteration", "-i", type=int, default=0,
                       help="Iteration number for saving")
    parser.add_argument("--save", action="store_true",
                       help="Save solution to run_logs/")
    parser.add_argument("--output-format", choices=["json", "text"], default="text",
                       help="Output format")

    args = parser.parse_args()

    # Load problem text
    if args.problem_file:
        with open(args.problem_file, "r", encoding="utf-8") as f:
            problem_text = f.read()
    elif args.problem_text:
        problem_text = args.problem_text
    elif not sys.stdin.isatty():
        problem_text = sys.stdin.read()
    else:
        print("Error: provide --problem-file, --problem-text, or pipe to stdin", file=sys.stderr)
        sys.exit(1)

    if args.action == "solve":
        result = solve(problem_text, args.model, args.thinking_budget, args.hint)
    elif args.action == "improve":
        # Load current solution and bug report
        if not args.solution_file or not args.bug_report_file:
            print("Error: --solution-file and --bug-report-file required for improve", file=sys.stderr)
            sys.exit(1)
        with open(args.solution_file, "r", encoding="utf-8") as f:
            current_solution = f.read()
        with open(args.bug_report_file, "r", encoding="utf-8") as f:
            bug_report = f.read()
        result = improve(problem_text, current_solution, bug_report,
                        args.model, args.thinking_budget, args.hint)
    elif args.action == "verify":
        # Load solution to verify
        if not args.solution_file:
            print("Error: --solution-file required for verify", file=sys.stderr)
            sys.exit(1)
        with open(args.solution_file, "r", encoding="utf-8") as f:
            solution_text = f.read()
        # Use smaller thinking budget for verification (default 65536)
        verify_budget = min(args.thinking_budget, 65536)
        result = verify(problem_text, solution_text, args.model, verify_budget)

    if "error" in result:
        print(json.dumps(result), file=sys.stderr)
        sys.exit(1)

    # Save if requested
    job_name = args.job_name or (os.path.splitext(os.path.basename(args.problem_file))[0] if args.problem_file else "unnamed")
    
    if args.save:
        if "solution" in result:
            filepath = save_solution(job_name, args.iteration, result["solution"])
            result["saved_to"] = filepath
            print(f"Saved to: {filepath}", file=sys.stderr)
        elif "verification" in result:
            filepath = save_validation(job_name, args.iteration, result["verification"], result.get("passed", False))
            result["saved_to"] = filepath
            print(f"Saved to: {filepath}", file=sys.stderr)

    # Output
    if args.output_format == "json":
        print(json.dumps(result, ensure_ascii=False))
    else:
        if "solution" in result:
            print(result.get("solution", ""))
        elif "verification" in result:
            status = "PASSED" if result.get("passed") else "FAILED"
            print(f"Status: {status}\n")
            print(result.get("verification", ""))


if __name__ == "__main__":
    main()
