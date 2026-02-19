"""
Utility functions shared across the IMO agent codebase.
"""

import os
import sys
import json
from typing import Optional, Tuple


# ============================================================================
# Logging Utilities
# ============================================================================

_log_file = None
_original_print = print


def log_print(*args, **kwargs):
    """
    Custom print function that writes to both stdout and log file.
    """
    message = ' '.join(str(arg) for arg in args)
    
    # Add timestamp to lines starting with ">>>>>"
    if message.startswith('>>>>>'):
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f"[{timestamp}] {message}"
    
    # Print to stdout
    _original_print(message)
    
    # Also write to log file if specified
    if _log_file is not None:
        _log_file.write(message + '\n')
        _log_file.flush()


def set_log_file(log_file_path: str) -> bool:
    """Set the log file for output."""
    global _log_file
    if log_file_path:
        try:
            _log_file = open(log_file_path, 'w', encoding='utf-8')
            return True
        except Exception as e:
            print(f"Error opening log file {log_file_path}: {e}")
            return False
    return True


def close_log_file():
    """Close the log file if it's open."""
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None


def get_log_file():
    """Get the current log file handle."""
    return _log_file


# ============================================================================
# Memory Management
# ============================================================================

def save_memory(memory_file: str, problem_statement: str, other_prompts: list,
                current_iteration: int, max_runs: int,
                solution: Optional[str] = None,
                verify: Optional[str] = None,
                full_verification: Optional[str] = None) -> bool:
    """
    Save the current state to a memory file.
    
    Args:
        memory_file: Path to the memory file
        problem_statement: The problem statement
        other_prompts: Additional prompts
        current_iteration: Current iteration number
        max_runs: Maximum number of runs
        solution: Current solution (optional)
        verify: Verification result (optional)
        full_verification: Full verification output (optional)
        
    Returns:
        True if successful, False otherwise
    """
    memory = {
        "problem_statement": problem_statement,
        "other_prompts": other_prompts,
        "current_iteration": current_iteration,
        "max_runs": max_runs,
        "solution": solution,
        "verify": verify,
        "full_verification": full_verification,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(memory_file)), exist_ok=True)
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
        print(f"Memory saved to {memory_file}")
        return True
    except Exception as e:
        print(f"Error saving memory to {memory_file}: {e}")
        return False


def load_memory(memory_file: str) -> Optional[dict]:
    """
    Load the state from a memory file.
    
    Args:
        memory_file: Path to the memory file
        
    Returns:
        The memory dictionary, or None if loading failed
    """
    try:
        with open(memory_file, 'r', encoding='utf-8') as f:
            memory = json.load(f)
        print(f"Memory loaded from {memory_file}")
        return memory
    except Exception as e:
        print(f"Error loading memory from {memory_file}: {e}")
        return None


# ============================================================================
# Text Processing
# ============================================================================

def extract_detailed_solution(solution: str, marker: str = 'Detailed Solution', after: bool = True) -> str:
    """
    Extract text from solution based on marker.
    
    Args:
        solution: The full solution text
        marker: The marker to search for
        after: If True, return text after marker; else return text before marker
        
    Returns:
        Extracted text, or empty string if marker not found
    """
    idx = solution.find(marker)
    if idx == -1:
        return ''
    if after:
        return solution[idx + len(marker):].strip()
    else:
        return solution[:idx].strip()


def read_file_content(filepath: str) -> str:
    """
    Read and return the content of a file.
    
    Args:
        filepath: Path to the file
        
    Returns:
        File content
        
    Raises:
        SystemExit: If file cannot be read
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)


def extract_latex(out: str) -> str:
    """
    Extract LaTeX code from markdown-style code blocks.
    
    Args:
        out: Text that may contain LaTeX in code blocks
        
    Returns:
        Clean LaTeX text
    """
    out = out.strip()
    if out.startswith("```"):
        lines = out.split("\n")
        if "latex" in lines[0].lower() or "tex" in lines[0].lower() or lines[0] == "```":
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        out = "\n".join(lines)
    return out.strip()


# ============================================================================
# API Response Processing
# ============================================================================

def extract_text_from_response(response_data: dict, provider_type: str = "gemini") -> Tuple[str, str]:
    """
    Extract text and thinking from API response based on provider type.
    
    Args:
        response_data: The API response dictionary
        provider_type: Type of provider ('gemini', 'openai', 'kimi')
        
    Returns:
        Tuple of (text, thinking)
    """
    try:
        if provider_type == "gemini":
            text = response_data['candidates'][0]['content']['parts'][0]['text']
            thinking = response_data.get('thinking_process', '')
            return text, thinking
            
        elif provider_type in ("openai", "kimi"):
            text = response_data["choices"][0]["message"]["content"]
            # Check for reasoning_content (Kimi thinking)
            message = response_data["choices"][0].get("message", {})
            thinking = message.get("reasoning_content", "")
            if not thinking:
                thinking = response_data.get("thinking_process", "")
            return text, thinking
            
        else:
            # Generic fallback
            if "candidates" in response_data:
                text = response_data['candidates'][0]['content']['parts'][0]['text']
            elif "choices" in response_data:
                text = response_data["choices"][0]["message"]["content"]
            else:
                text = str(response_data)
            return text, ""
            
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error extracting text from response: {e}")
        print(json.dumps(response_data, indent=2)[:500])
        return "", ""


# ============================================================================
# Quota Management
# ============================================================================

class QuotaManager:
    """Manages API quota status."""
    
    def __init__(self):
        self.quota_exceeded = False
        self.quota_warning = False
        self.last_check_time = 0
    
    def check_quota(self, headers: dict) -> bool:
        """
        Check response headers for quota information.
        
        Args:
            headers: Response headers
            
        Returns:
            True if quota is OK, False if exceeded
        """
        import time
        
        # Common header patterns for quota information
        quota_headers = [
            ("X-RateLimit-Remaining", "X-RateLimit-Limit"),
            ("X-Quota-Remaining", "X-Quota-Limit"),
            ("X-API-Quota-Remaining", "X-API-Quota-Limit"),
            ("X-Daily-Remaining", "X-Daily-Limit"),
        ]
        
        for remaining_header, limit_header in quota_headers:
            if remaining_header in headers and limit_header in headers:
                try:
                    remaining = int(headers.get(remaining_header, "0"))
                    limit = int(headers.get(limit_header, "1"))
                    
                    if remaining <= 0:
                        self.quota_exceeded = True
                        print(f"⚠️ TOKEN QUOTA EXCEEDED! ({remaining}/{limit})")
                        return False
                    
                    if remaining < 0.1 * limit:  # Less than 10% remaining
                        self.quota_warning = True
                        print(f"⚠️ TOKEN QUOTA WARNING: {remaining}/{limit} remaining ({remaining/limit*100:.1f}%)")
                    else:
                        self.quota_warning = False
                    
                    return True
                    
                except (ValueError, TypeError):
                    continue
        
        return True
    
    def check_response_for_quota_error(self, response_text: str) -> bool:
        """
        Check if response contains quota error.
        
        Args:
            response_text: Error response text
            
        Returns:
            True if quota exceeded detected
        """
        quota_keywords = [
            "quota exceeded",
            "rate limit",
            "too many requests",
            "resource exhausted",
            "usage limit",
            "billing limit",
            "daily limit"
        ]
        
        text_lower = response_text.lower()
        if any(keyword in text_lower for keyword in quota_keywords):
            self.quota_exceeded = True
            return True
        return False


# Global quota manager instance
quota_manager = QuotaManager()
