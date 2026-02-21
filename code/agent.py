"""
MIT License

Copyright (c) 2025 Lin Yang, Yichen Huang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Gemini Agent for IMO problem solving.

This module provides the Gemini-specific agent implementation using the
refactored architecture with ProofEngine and LLMClient.
"""

import os
import sys
import argparse
from typing import Optional, Callable

# Import the new architecture components
try:
    from .proof_engine import ProofEngine
    from .llm_clients import GeminiClient, set_default_client
    from . import utils
except ImportError:
    from proof_engine import ProofEngine
    from llm_clients import GeminiClient, set_default_client
    import utils

# Configuration
MODEL_NAME = "gemini-2.0-flash"

# Re-export utility functions for backwards compatibility
set_log_file = utils.set_log_file
close_log_file = utils.close_log_file
save_memory = utils.save_memory
load_memory = utils.load_memory
extract_detailed_solution = utils.extract_detailed_solution
read_file_content = utils.read_file_content

# Re-export quota manager for backwards compatibility
quota_manager = utils.quota_manager
TOKEN_QUOTA_EXCEEDED = quota_manager.quota_exceeded
TOKEN_QUOTA_WARNING = quota_manager.quota_warning


def check_token_quota(api_key: str) -> bool:
    """Check API token quota."""
    # Quota checking is now handled by the quota_manager
    return not quota_manager.quota_exceeded


def set_model(model_name: str):
    """Set the model name."""
    global MODEL_NAME
    MODEL_NAME = model_name
    print(f"Gemini model set to: {model_name}")


def agent(problem_statement: str,
          other_prompts: Optional[list] = None,
          verify_prompts: Optional[list] = None,
          memory_file: Optional[str] = None,
          resume_from_memory: bool = False,
          on_iteration_result: Optional[Callable] = None,
          streaming: bool = True,
          show_thinking: bool = True,
          interactive: bool = False) -> Optional[str]:
    """
    Main agent function that processes and solves mathematical problems.
    
    Args:
        problem_statement: The mathematical problem to solve
        other_prompts: Additional prompts to guide the model
        memory_file: File to save/load state
        resume_from_memory: Whether to resume from saved state
        on_iteration_result: Callback function for iteration results
        streaming: Whether to use streaming output
        show_thinking: Whether to display thinking process
        interactive: Whether to enable interactive mode
        
    Returns:
        The final solution, or None if failed
    """
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)
    
    # Create LLM client
    client = GeminiClient(api_key=api_key, model_name=MODEL_NAME)
    set_default_client(client)
    
    # Create proof engine
    engine = ProofEngine(
        client=client,
        memory_file=memory_file,
        on_iteration=on_iteration_result,
        interactive=interactive
    )
    
    # Load state if resuming
    if resume_from_memory and memory_file:
        engine.load_state(memory_file)
    
    # Run the proof flow
    return engine.run(
        problem_statement=problem_statement,
        other_prompts=other_prompts or [],
        verify_prompts=verify_prompts or [],
        resume=resume_from_memory,
        streaming=streaming,
        show_thinking=show_thinking
    )


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description="Gemini Agent for IMO problem solving")
    parser.add_argument("problem_file", help="Path to the problem file")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Model name to use")
    parser.add_argument("--memory", help="Memory file to resume from")
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming")
    parser.add_argument("--no-thinking", action="store_true", help="Hide thinking process")
    
    args = parser.parse_args()
    
    # Set model if specified
    if args.model:
        set_model(args.model)
    
    # Read problem
    problem = read_file_content(args.problem_file)
    
    # Run agent
    streaming = not args.no_streaming
    show_thinking = not args.no_thinking
    
    solution = agent(
        problem,
        memory_file=args.memory,
        resume_from_memory=bool(args.memory),
        streaming=streaming,
        show_thinking=show_thinking
    )
    
    if solution:
        print("\n" + "="*50)
        print("FINAL SOLUTION:")
        print("="*50)
        print(solution)


if __name__ == "__main__":
    main()
