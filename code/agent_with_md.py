#!/usr/bin/env python3
"""
Proof agent with md file saving for each iteration.
"""

import json
import os
import sys
from datetime import datetime

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)


def save_md(problem: str, solution: str, validation: str, 
            iteration: int, base_name: str, output_dir: str):
    """Save solution and validation to md files."""
    
    sep = "=" * 50
    
    # Solution file
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
    
    # Validation file
    val_file = os.path.join(output_dir, f"{base_name}_iter{iteration}_validation.md")
    val_content = f"""# Validation (Iteration {iteration})

**Generated:** {datetime.now().isoformat()}

## Verification Result

{validation}
"""
    with open(val_file, "w", encoding="utf-8") as f:
        f.write(val_content)
    
    return sol_file, val_file


def run_agent(problem_file: str, output_dir: str = "run_logs", 
              model_name: str = "gemini-2.0-flash"):
    """Run the agent with md file saving."""
    
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
    
    # Initial exploration
    print("\n--- Iteration 0: Initial exploration ---")
    success = engine.initialize_exploration(problem, [], streaming=False, show_thinking=False)
    
    if not success:
        print("Failed to get initial solution")
        return
    
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
        print(f"\n--- Iteration {iteration}: Verification ---")
        
        engine.verify_solution(streaming=False, show_thinking=False)
        
        verification = engine.state.full_verification or engine.state.verify or "Verification complete"
        is_correct = engine.state.verification_passed
        
        if is_correct:
            consecutive_correct += 1
            status = f"PASSED (consecutive: {consecutive_correct}/5)"
        else:
            consecutive_correct = 0
            status = "FAILED"
        
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
        print("\nVerification result:")
        print(verification[:500] if len(verification) > 500 else verification)
        
        if consecutive_correct >= 5:
            print("\n*** Solution verified 5 times! Complete! ***")
            break
    
    # Save final
    final_file = os.path.join(output_dir, f"{base_name}_final.md")
    with open(final_file, "w", encoding="utf-8") as f:
        f.write(f"# Final Solution\n\n")
        f.write(f"Verified: {consecutive_correct} consecutive times\n\n")
        f.write(f"## Problem\n\n{problem}\n\n")
        f.write(f"## Solution\n\n{engine.state.solution or ''}\n")
    
    print(f"\nFinal: {os.path.basename(final_file)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_file")
    parser.add_argument("--output-dir", default="run_logs")
    parser.add_argument("--model", default="gemini-2.0-flash")
    args = parser.parse_args()
    
    run_agent(args.problem_file, args.output_dir, args.model)