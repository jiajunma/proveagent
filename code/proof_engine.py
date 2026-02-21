"""
Proof Engine - Core proof flow logic.

This module provides the high-level proof flow that is independent of specific API providers.
It uses an abstract LLM client interface to communicate with different backends.
"""

import sys
from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Tuple, Dict, Any
from dataclasses import dataclass, field

# Import prompts and utilities
try:
    from . import prompts
    from .utils import (
        extract_detailed_solution, 
        save_memory, 
        load_memory,
        quota_manager
    )
except ImportError:
    # Allow running as standalone script
    import prompts
    from utils import (
        extract_detailed_solution, 
        save_memory, 
        load_memory,
        quota_manager
    )


# ============================================================================
# Data Classes
# ============================================================================

# Number of consecutive verification passes required to accept a solution
REQUIRED_CONSECUTIVE_PASS = 5


@dataclass
class ProofState:
    """Represents the current state of a proof attempt."""
    problem_statement: str = ""
    other_prompts: List[str] = field(default_factory=list)
    verify_prompts: List[str] = field(default_factory=list)  # Additional prompts for verification only
    solution: Optional[str] = None
    bug_report: str = ""
    full_verification: str = ""
    verification_passed: bool = False
    current_iteration: int = 0
    max_iterations: int = 30
    error_count: int = 0
    correct_count: int = 0
    consecutive_correct: int = 0  # Consecutive verification passes (resets on failure)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "problem_statement": self.problem_statement,
            "other_prompts": self.other_prompts,
            "verify_prompts": self.verify_prompts,
            "solution": self.solution,
            "bug_report": self.bug_report,
            "full_verification": self.full_verification,
            "verification_passed": self.verification_passed,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "error_count": self.error_count,
            "correct_count": self.correct_count,
            "consecutive_correct": self.consecutive_correct,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProofState":
        """Create state from dictionary."""
        state = cls()
        state.problem_statement = data.get("problem_statement", "")
        state.other_prompts = data.get("other_prompts", [])
        state.verify_prompts = data.get("verify_prompts", [])
        state.solution = data.get("solution")
        state.bug_report = data.get("bug_report", "")
        state.full_verification = data.get("full_verification", "")
        state.verification_passed = data.get("verification_passed", False)
        state.current_iteration = data.get("current_iteration", 0)
        state.max_iterations = data.get("max_iterations", 30)
        state.error_count = data.get("error_count", 0)
        state.correct_count = data.get("correct_count", 0)
        state.consecutive_correct = data.get("consecutive_correct", 0)
        return state


@dataclass
class LLMResponse:
    """Response from LLM."""
    text: str
    thinking: str = ""
    raw_response: Any = None


# ============================================================================
# Abstract LLM Client Interface
# ============================================================================

class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    Implementations must provide methods to interact with specific APIs
    (Gemini, OpenAI, Kimi, etc.)
    """
    
    @abstractmethod
    def generate(self, 
                 system_prompt: str,
                 user_prompt: str,
                 other_prompts: Optional[List[str]] = None,
                 enable_thinking: bool = True,
                 streaming: bool = True) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            system_prompt: System prompt to guide the model
            user_prompt: Main user question/prompt
            other_prompts: Additional user prompts
            enable_thinking: Whether to enable thinking process
            streaming: Whether to use streaming output
            
        Returns:
            LLMResponse containing text and thinking
        """
        pass
    
    @abstractmethod
    def check_quota(self) -> bool:
        """
        Check if API quota is available.
        
        Returns:
            True if quota is available, False if exceeded
        """
        pass
    
    @property
    @abstractmethod
    def supports_thinking(self) -> bool:
        """Whether this client supports thinking process display."""
        pass
    
    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this client supports streaming output."""
        pass


# ============================================================================
# Callback Types
# ============================================================================

IterationCallback = Callable[[str, str, str, int], None]  # problem, solution, verification, iteration
InteractiveCallback = Callable[[], str]  # Returns user input


# ============================================================================
# Proof Engine
# ============================================================================

class ProofEngine:
    """
    Core proof engine that manages the proof flow.
    
    This class is agnostic to the specific LLM API being used.
    It uses an LLMClient implementation to communicate with the backend.
    """
    
    def __init__(self, 
                 client: LLMClient,
                 memory_file: Optional[str] = None,
                 on_iteration: Optional[IterationCallback] = None,
                 interactive: bool = False):
        """
        Initialize the proof engine.
        
        Args:
            client: LLM client implementation
            memory_file: Optional file to save/load state
            on_iteration: Optional callback for iteration results
            interactive: Whether to enable interactive mode
        """
        self.client = client
        self.memory_file = memory_file
        self.on_iteration = on_iteration
        self.interactive = interactive
        self.state = ProofState()
    
    def load_state(self, memory_file: Optional[str] = None) -> bool:
        """
        Load state from memory file.
        
        Args:
            memory_file: Path to memory file (uses self.memory_file if None)
            
        Returns:
            True if state was loaded successfully
        """
        mf = memory_file or self.memory_file
        if not mf:
            return False
            
        memory = load_memory(mf)
        if memory:
            self.state = ProofState.from_dict(memory)
            print(f"Resuming from iteration {self.state.current_iteration}")
            return True
        else:
            print("Failed to load memory, starting fresh")
            self.state = ProofState()
            return False
    
    def save_state(self, memory_file: Optional[str] = None) -> bool:
        """
        Save current state to memory file.
        
        Args:
            memory_file: Path to memory file (uses self.memory_file if None)
            
        Returns:
            True if state was saved successfully
        """
        mf = memory_file or self.memory_file
        if not mf:
            return False
            
        result = save_memory(
            mf,
            self.state.problem_statement,
            self.state.other_prompts,
            self.state.current_iteration,
            self.state.max_iterations,
            self.state.solution,
            "yes" if self.state.verification_passed else "no",
            self.state.full_verification
        )
        # Save extra state fields that save_memory doesn't cover
        if result:
            try:
                import json
                with open(mf, "r", encoding="utf-8") as f:
                    mem = json.load(f)
                mem["consecutive_correct"] = self.state.consecutive_correct
                mem["verify_prompts"] = self.state.verify_prompts
                with open(mf, "w", encoding="utf-8") as f:
                    json.dump(mem, f, indent=2, ensure_ascii=False)
            except Exception:
                pass  # Best effort
        return result
    
    def initialize_exploration(self, 
                              problem_statement: str,
                              other_prompts: Optional[List[str]] = None,
                              streaming: bool = True,
                              show_thinking: bool = True) -> bool:
        """
        Initialize the exploration phase to generate an initial solution.
        
        Args:
            problem_statement: The problem to solve
            other_prompts: Additional prompts to guide the model
            streaming: Whether to use streaming output
            show_thinking: Whether to show thinking process
            
        Returns:
            True if initialization was successful
        """
        self.state.problem_statement = problem_statement
        if other_prompts:
            self.state.other_prompts = other_prompts
        
        print(">>>>>>> Starting initial exploration...")
        
        try:
            response = self.client.generate(
                system_prompt=prompts.STEP1_SYSTEM_PROMPT,
                user_prompt=problem_statement,
                other_prompts=self.state.other_prompts,
                enable_thinking=show_thinking,
                streaming=streaming
            )
            
            self.state.solution = response.text
            
            if not self.state.solution:
                print(">>>>>>> Failed to get initial solution.")
                return False
                
            print(">>>>>>> Initial solution generated.")
            return True
            
        except Exception as e:
            print(f">>>>>>> Error during initial exploration: {e}")
            return False
    
    def verify_solution(self, 
                       streaming: bool = True,
                       show_thinking: bool = True) -> bool:
        """
        Verify the current solution.
        
        Args:
            streaming: Whether to use streaming output
            show_thinking: Whether to show thinking process
            
        Returns:
            True if verification passed
        """
        if not self.state.solution:
            print(">>>>>>> No solution to verify.")
            return False
        
        print(">>>>>>> Starting verification...")
        
        # Build verification prompt
        verification_prompt = prompts.build_verification_prompt(
            self.state.problem_statement,
            self.state.solution
        )

        # Append verification-specific comments if any
        verify_other = None
        if self.state.verify_prompts:
            verify_other = list(self.state.verify_prompts)

        try:
            # Get verification result
            response = self.client.generate(
                system_prompt=prompts.VERIFICATION_SYSTEM_PROMPT,
                user_prompt=verification_prompt,
                other_prompts=verify_other,
                enable_thinking=show_thinking,
                streaming=streaming
            )
            
            verification_output = response.text
            self.state.full_verification = verification_output
            
            # Check if verification passed
            check_prompt = prompts.build_verification_check_prompt(verification_output)
            
            check_response = self.client.generate(
                system_prompt="",
                user_prompt=check_prompt,
                enable_thinking=False,
                streaming=False
            )
            
            check_result = check_response.text.lower()
            self.state.verification_passed = "yes" in check_result
            
            # Extract bug report if verification failed
            if not self.state.verification_passed:
                self.state.bug_report = extract_detailed_solution(
                    verification_output, "Detailed Verification", after=False
                )
                self.state.consecutive_correct = 0  # Reset on failure
            else:
                self.state.bug_report = ""
                self.state.correct_count += 1
                self.state.consecutive_correct += 1

            print(f">>>>>>> Verification {'passed' if self.state.verification_passed else 'failed'}."
                  f" (consecutive pass: {self.state.consecutive_correct}/{REQUIRED_CONSECUTIVE_PASS})")
            return self.state.verification_passed
            
        except Exception as e:
            print(f">>>>>>> Error during verification: {e}")
            self.state.error_count += 1
            return False
    
    def improve_solution(self,
                        streaming: bool = True,
                        show_thinking: bool = True) -> bool:
        """
        Improve the solution based on the bug report.
        
        Args:
            streaming: Whether to use streaming output
            show_thinking: Whether to show thinking process
            
        Returns:
            True if improvement was successful
        """
        if not self.state.bug_report:
            print(">>>>>>> No bug report to improve from.")
            return False
        
        print(">>>>>>> Improving solution...")
        
        # Build improvement prompt
        correction_prompt = prompts.build_correction_prompt(
            self.state.problem_statement,
            self.state.solution,
            self.state.bug_report
        )
        
        # Add self-improvement prompt to other_prompts
        improvement_prompts = self.state.other_prompts + [prompts.SELF_IMPROVEMENT_PROMPT]
        
        try:
            response = self.client.generate(
                system_prompt=prompts.STEP1_SYSTEM_PROMPT,
                user_prompt=correction_prompt,
                other_prompts=improvement_prompts,
                enable_thinking=show_thinking,
                streaming=streaming
            )
            
            new_solution = response.text
            
            if not new_solution:
                print(">>>>>>> Failed to get improved solution.")
                self.state.error_count += 1
                return False
            
            self.state.solution = new_solution
            self.state.consecutive_correct = 0  # Reset on solution change
            print(">>>>>>> Solution improved.")
            return True
            
        except Exception as e:
            print(f">>>>>>> Error during improvement: {e}")
            self.state.error_count += 1
            return False
    
    def check_completeness(self) -> bool:
        """
        Check if the solution claims to be complete.
        
        Returns:
            True if solution claims to be complete
        """
        if not self.state.solution:
            return False
        
        check_prompt = prompts.build_completeness_check_prompt(self.state.solution)
        
        try:
            response = self.client.generate(
                system_prompt="",
                user_prompt=check_prompt,
                enable_thinking=False,
                streaming=False
            )
            
            return "yes" in response.text.lower()
            
        except Exception as e:
            print(f">>>>>>> Error checking completeness: {e}")
            return False
    
    def run_iteration(self,
                     streaming: bool = True,
                     show_thinking: bool = True) -> Tuple[bool, bool]:
        """
        Run a single iteration of the proof-improvement loop.
        
        Args:
            streaming: Whether to use streaming output
            show_thinking: Whether to show thinking process
            
        Returns:
            Tuple of (should_continue, success)
            - should_continue: Whether to continue iterating
            - success: Whether a complete solution was found
        """
        print(f"\n===== Iteration {self.state.current_iteration} =====")
        print(f"Correct: {self.state.correct_count} (consecutive: {self.state.consecutive_correct}/{REQUIRED_CONSECUTIVE_PASS}), Errors: {self.state.error_count}")
        
        # Check quota
        if quota_manager.quota_exceeded:
            print("⚠️ TOKEN QUOTA EXCEEDED! Cannot continue.")
            self.save_state()
            return False, False
        
        if quota_manager.quota_warning:
            print("⚠️ WARNING: Token quota is running low.")
        
        # Verify current solution
        is_correct = self.verify_solution(streaming, show_thinking)

        # Call callback after verification (PDF update with new verification report)
        if self.on_iteration and self.state.solution:
            try:
                self.on_iteration(
                    self.state.problem_statement,
                    self.state.solution,
                    self.state.full_verification,
                    self.state.current_iteration
                )
            except Exception as e:
                print(f">>>>>>> Callback error: {e}")

        if is_correct:
            print(f">>>>>>> Verification passed! ({self.state.consecutive_correct}/{REQUIRED_CONSECUTIVE_PASS})")

            if self.state.consecutive_correct >= REQUIRED_CONSECUTIVE_PASS and self.check_completeness():
                print(f">>>>>>> Solution verified {REQUIRED_CONSECUTIVE_PASS} consecutive times. Complete!")
                self.save_state()
                return False, True
            elif self.state.consecutive_correct >= REQUIRED_CONSECUTIVE_PASS:
                print(">>>>>>> Solution passed all verifications but not claimed complete. Continuing...")
                is_correct = False  # Force improvement
            else:
                print(f">>>>>>> Need {REQUIRED_CONSECUTIVE_PASS - self.state.consecutive_correct} more consecutive pass(es). Re-verifying...")
                # Don't improve — just re-verify in next iteration
                self.save_state()
                self.state.current_iteration += 1
                if self.state.current_iteration >= self.state.max_iterations:
                    print(">>>>>>> Maximum iterations reached.")
                    return False, False
                return True, False

        # Improve solution if needed
        if not is_correct:
            print(f">>>>>>> Solution needs improvement.")

            success = self.improve_solution(streaming, show_thinking)

            if not success:
                if self.state.error_count >= 3:
                    print(">>>>>>> Too many errors. Stopping.")
                    self.save_state()
                    return False, False

            # Call callback after improvement (PDF update with new solution)
            if self.on_iteration and self.state.solution:
                try:
                    self.on_iteration(
                        self.state.problem_statement,
                        self.state.solution,
                        self.state.full_verification,
                        self.state.current_iteration
                    )
                except Exception as e:
                    print(f">>>>>>> Callback error: {e}")

            # Save state
            self.save_state()
        
        self.state.current_iteration += 1
        
        # Check max iterations
        if self.state.current_iteration >= self.state.max_iterations:
            print(">>>>>>> Maximum iterations reached.")
            return False, False
        
        return True, False
    
    def run(self,
           problem_statement: Optional[str] = None,
           other_prompts: Optional[List[str]] = None,
           verify_prompts: Optional[List[str]] = None,
           resume: bool = False,
           streaming: bool = True,
           show_thinking: bool = True) -> Optional[str]:
        """
        Run the complete proof flow.
        
        Args:
            problem_statement: The problem to solve (uses loaded state if None)
            other_prompts: Additional prompts
            resume: Whether to resume from saved state
            streaming: Whether to use streaming output
            show_thinking: Whether to show thinking process
            
        Returns:
            The final solution, or None if failed
        """
        # Load state if resuming
        if resume and self.memory_file:
            self.load_state()
        
        # Set problem if provided
        if problem_statement:
            self.state.problem_statement = problem_statement
            if other_prompts:
                self.state.other_prompts = other_prompts
            if verify_prompts:
                self.state.verify_prompts = verify_prompts
        
        # Check if we have a problem
        if not self.state.problem_statement:
            print(">>>>>>> No problem statement provided.")
            return None
        
        # Initialize if no solution yet
        if not self.state.solution:
            success = self.initialize_exploration(
                self.state.problem_statement,
                self.state.other_prompts,
                streaming,
                show_thinking
            )
            
            if not success:
                print(">>>>>>> Failed in finding a complete solution.")
                self.save_state()
                return None
            
            # First verification
            self.verify_solution(streaming, show_thinking)
            
            # Call callback for first iteration
            if self.on_iteration and self.state.solution:
                try:
                    self.on_iteration(
                        self.state.problem_statement,
                        self.state.solution,
                        self.state.full_verification,
                        -1
                    )
                except Exception as e:
                    print(f">>>>>>> Callback error: {e}")
            
            self.save_state()
        else:
            # Verify existing solution
            self.verify_solution(streaming, show_thinking)

            # Call callback after verification (PDF update with verification report)
            if self.on_iteration and self.state.solution:
                try:
                    self.on_iteration(
                        self.state.problem_statement,
                        self.state.solution,
                        self.state.full_verification,
                        -1
                    )
                except Exception as e:
                    print(f">>>>>>> Callback error: {e}")

            self.save_state()

        # Main loop
        while True:
            should_continue, success = self.run_iteration(streaming, show_thinking)

            if not should_continue:
                if success:
                    print(">>>>>>> Proof complete!")
                else:
                    print(">>>>>>> Proof process ended.")
                break

        # Final save to ensure latest state is persisted
        self.save_state()
        return self.state.solution
