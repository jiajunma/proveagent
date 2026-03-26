#!/usr/bin/env python3
"""
Proof agent with md saving + chat hints integration.
Each iteration:
1. Save solution.md and validation.md
2. Check for new hints from chat
3. Incorporate hints into next generation
"""

import json
import os
import sys
from datetime import datetime
import threading
import time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

# Hints queue (for receiving hints from chat)
hints_queue = []
hints_lock = threading.Lock()


def save_md(problem: str, solution: str, validation: str, 
            iteration: int, base_name: str, output_dir: str):
    """Save solution and validation to md files."""
    
    sol_file = os.path.join(output_dir, f"{base_name}_iter{iteration}_solution.md")
    sol_content = f"""# Solution (Iteration {iteration})

**Generated:** {datetime.now().isoformat()}

## Problem

{problem}

## Solution

{solution}
"""
    with open(sol_file, "w", encoding="utf-8") as f:
        f.write(sol_content)
    
    val_file = os.path.join(output_dir, f"{base_name}_iter{iteration}_validation.md")
    val_content = f"""# Validation (Iteration {iteration})

**Generated:** {datetime.now().isoformat()}

## Verification Result

{validation}
"""
    with open(val_file, "w", encoding="utf-8") as f:
        f.write(val_content)
    
    return sol_file, val_file


def add_hint(hint: str):
    """Add a hint from user (called from chat)."""
    with hints_lock:
        hints_queue.append({
            "hint": hint,
            "timestamp": datetime.now().isoformat()
        })
    print(f"\n>>> NEW HINT RECEIVED: {hint[:100]}...")


def get_hints():
    """Get all pending hints and clear the queue."""
    with hints_lock:
        hints = [h["hint"] for h in hints_queue]
        hints_queue.clear()
        return hints


def hints_watcher(problem_file: str, check_interval: float = 2.0):
    """Watch for hints file (simulates receiving from chat)."""
    hints_file = problem_file.replace(".txt", "_hints.txt")
    last_check = 0
    
    while True:
        try:
            if os.path.exists(hints_file):
                mtime = os.path.getmtime(hints_file)
                if mtime > last_check:
                    last_check = mtime
                    with open(hints_file, "r") as f:
                        new_hints = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                    for hint in new_hints:
                        add_hint(hint)
        except Exception:
            pass
        time.sleep(check_interval)


def run_agent(problem_file: str, output_dir: str = "run_logs", 
              model_name: str = "gemini-2.0-flash"):
    """Run the agent with md file saving and hint integration."""
    
    with open(problem_file, "r", encoding="utf-8") as f:
        problem = f.read()
    
    base_name = os.path.splitext(os.path.basename(problem_file))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    # Get API key
    import subprocess
    result = subprocess.run(["bash", "-c", "source ~/.profile && echo $GOOGLE_API_KEY"], 
                           capture_output=True, text=True)
    api_key = result.stdout.strip()
    
    if not api_key:
        print("Error: GOOGLE_API_KEY not set")
        return
    
    os.environ["GOOGLE_API_KEY"] = api_key
    
    from proof_engine import ProofEngine
    from llm_clients import GeminiClient
    
    client = GeminiClient(api_key=api_key, model_name=model_name)
    engine = ProofEngine(client=client)
    
    print(f"Running agent for: {base_name}")
    print(f"Model: {model_name}")
    print(f"Hints file: {problem_file.replace('.txt', '_hints.txt')}")
    
    # Initial exploration
    print("\n--- Iteration 0: Initial exploration ---")
    success = engine.initialize_exploration(problem, [], streaming=False, show_thinking=False)
    
    if not success:
        print("Failed to get initial solution")
        return
    
    # Get any initial hints
    hints = get_hints()
    if hints:
        print(f"Incorporating {len(hints)} hint(s) from initial setup")
        engine.state.other_prompts.extend(hints)
    
    # Save iteration 0
    sol_file, val_file = save_md(problem, engine.state.solution or "", 
                                  "Initial solution generated", 0, base_name, output_dir)
    print(f"Saved: {os.path.basename(sol_file)}")
    print(f"Saved: {os.path.basename(val_file)}")
    
    # Report to user
    print("\n" + "=" * 50)
    print("ITERATION 0 - SOLUTION (preview)")
    print("=" * 50)
    print((engine.state.solution or "")[:800])
    print("\n" + "=" * 50)
    print("ITERATION 0 - VALIDATION")
    print("=" * 50)
    print("Initial solution generated")
    
    # Verification loop
    consecutive_correct = 0
    iteration = 0
    
    while consecutive_correct < 5:
        iteration += 1
        
        # Check for new hints before verification
        hints = get_hints()
        if hints:
            print(f"\n>>> Incorporating {len(hints)} new hint(s):")
            for h in hints:
                print(f"   - {h[:80]}...")
            engine.state.other_prompts.extend(hints)
        
        print(f"\n--- Iteration {iteration}: Verification (with {len(engine.state.other_prompts)} hints) ---")
        
        engine.verify_solution(streaming=False, show_thinking=False)
        
        verification = engine.state.full_verification or "Verification complete"
        is_correct = engine.state.verification_passed
        
        if is_correct:
            consecutive_correct += 1
            status = f"PASSED (consecutive: {consecutive_correct}/5)"
        else:
            consecutive_correct = 0
            status = "FAILED - will retry with hints"
        
        print(f"Status: {status}")
        
        # Save iteration
        sol_file, val_file = save_md(problem, engine.state.solution or "", 
                                      verification, iteration, base_name, output_dir)
        print(f"Saved: {os.path.basename(sol_file)}")
        
        # Report to user
        print("\n" + "=" * 50)
        print(f"ITERATION {iteration} - VALIDATION")
        print("=" * 50)
        print(f"Status: {status}")
        print(f"Hints used: {len(engine.state.other_prompts)}")
        print("\nVerification result:")
        print(verification[:300] if len(verification) > 300 else verification)
        
        if consecutive_correct >= 5:
            print("\n*** Solution verified 5 times! Complete! ***")
            break
        
        # If failed, check for more hints before next iteration
        print("\nWaiting for hints... (add to _hints.txt or press Enter to continue)")
    
    # Save final
    final_file = os.path.join(output_dir, f"{base_name}_final.md")
    with open(final_file, "w", encoding="utf-8") as f:
        f.write(f"# Final Solution\n\n")
        f.write(f"Verified: {consecutive_correct} consecutive times\n")
        f.write(f"Total hints used: {len(engine.state.other_prompts)}\n\n")
        f.write(f"## Problem\n\n{problem}\n\n")
        f.write(f"## Solution\n\n{engine.state.solution or ''}\n\n")
        if engine.state.other_prompts:
            f.write(f"## Hints Applied\n\n")
            for i, h in enumerate(engine.state.other_prompts, 1):
                f.write(f"{i}. {h}\n")
    
    print(f"\nFinal: {os.path.basename(final_file)}")
    print("\nAll files saved to:", output_dir + "/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_file")
    parser.add_argument("--output-dir", default="run_logs")
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--hint", "-H", action="append", default=[], 
                       help="Add hint(s) to the agent")
    args = parser.parse_args()
    
    # Add initial hints
    for h in args.hint:
        add_hint(h)
    
    run_agent(args.problem_file, args.output_dir, args.model)