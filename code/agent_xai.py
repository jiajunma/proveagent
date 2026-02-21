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
X.AI (Grok) Agent for IMO problem solving.

This module provides the X.AI-specific agent implementation using the
refactored architecture with ProofEngine and LLMClient.
"""

import os
import sys
import argparse
from typing import Optional, Callable

# Import the new architecture components
try:
    from .proof_engine import ProofEngine, LLMClient, LLMResponse
    from .llm_clients import set_default_client
    from . import model_providers
    from . import utils
except ImportError:
    from proof_engine import ProofEngine, LLMClient, LLMResponse
    from llm_clients import set_default_client
    import model_providers
    import utils

# Configuration
MODEL_NAME = "grok-4-0709"

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


class XAIClient(LLMClient):
    """X.AI (Grok) API client implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "grok-4-0709"):
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            print("Error: XAI_API_KEY environment variable not set.")
            sys.exit(1)
        self.model_name = model_name
        self.api_url = "https://api.x.ai/v1/chat/completions"
        self._streaming_supported = True
    
    def generate(self, system_prompt: str, user_prompt: str,
                 other_prompts: Optional[list] = None,
                 enable_thinking: bool = True, streaming: bool = True) -> 'LLMResponse':
        """Generate response from X.AI API."""
        import requests
        import json
        
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        if other_prompts:
            for prompt in other_prompts:
                messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
        }
        
        if streaming and self._streaming_supported:
            payload["stream"] = True
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            if streaming and self._streaming_supported:
                accumulated_text = ""
                response = requests.post(self.api_url, headers=headers, 
                                        json=payload, stream=True, timeout=7200)
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    line_text = line.decode('utf-8')
                    if not line_text.startswith("data: "):
                        continue
                    data = line_text[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if "choices" in chunk and chunk["choices"]:
                            choice = chunk["choices"][0]
                            if "delta" in choice and "content" in choice["delta"]:
                                text_chunk = choice["delta"]["content"]
                                accumulated_text += text_chunk
                                print(text_chunk, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
                
                return LLMResponse(text=accumulated_text, thinking="")
            else:
                # Non-streaming
                payload.pop("stream", None)
                response = requests.post(self.api_url, headers=headers,
                                        json=payload, timeout=7200)
                response.raise_for_status()
                result = response.json()
                text = result["choices"][0]["message"]["content"]
                return LLMResponse(text=text, thinking="")
                
        except Exception as e:
            print(f"Error during X.AI API request: {e}")
            raise
    
    def check_quota(self) -> bool:
        """Check if API quota is available."""
        return not quota_manager.quota_exceeded
    
    @property
    def supports_thinking(self) -> bool:
        """X.AI models don't support thinking display."""
        return False
    
    @property
    def supports_streaming(self) -> bool:
        """Check if streaming is supported."""
        return self._streaming_supported


def check_token_quota(api_key: str) -> bool:
    """Check API token quota."""
    return not quota_manager.quota_exceeded


def set_model(model_name: str):
    """Set the model name."""
    global MODEL_NAME
    MODEL_NAME = model_name
    print(f"X.AI model set to: {model_name}")


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
    Main agent function that processes and solves mathematical problems using X.AI API.
    
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
    # Create LLM client
    client = XAIClient(model_name=MODEL_NAME)
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
    parser = argparse.ArgumentParser(description="X.AI Agent for IMO problem solving")
    parser.add_argument("problem_file", help="Path to the problem file")
    parser.add_argument("--model", default="grok-4-0709", help="Model name to use")
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
