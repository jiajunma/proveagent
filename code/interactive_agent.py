#!/usr/bin/env python3
"""
Interactive IMO agent - Claude Code style interface.
- Slash commands: /run, /load, /export, /help, etc.
- Bare input = additional prompt for next run
- Uses agent's memory files (.mem) for resume and export
- When solution+verification exist: compose tex, compile, open PDF
"""

import argparse
import json
from typing import Optional, Tuple
import multiprocessing
import os
import platform
import sys
import model_providers

def _save_readline_history():
    """Save readline history on exit. No-op if readline not available."""
    pass

try:
    import readline

    # Slash-command completions
    _READLINE_COMMANDS = [
        "/run", "/r", "/load", "/problem", "/prompt", "/add", "/p",
        "/export", "/e", "/status", "/s", "/list", "/l", "/clear",
        "/edit", "/edit_problem", "/done", "/edit_existing", "/save_as",
        "/streaming", "/thinking", "/interactive", "/run_mode", "/quota",
        "/provider", "/model", "/providers",
        "/help", "/h", "/quit", "/q", "/exit",
    ]

    def _readline_completer(text: str, state: int) -> Optional[str]:
        line = readline.get_line_buffer()
        if not line.strip().startswith("/"):
            return None
        # Normalize: text may be "ru" or "/ru" depending on delimiters
        prefix = text if text.startswith("/") else "/" + text
        matches = [c for c in _READLINE_COMMANDS if c.startswith(prefix)]
        matches = sorted(set(matches))
        if state < len(matches):
            return matches[state]
        return None

    readline.set_completer(_readline_completer)
    readline.parse_and_bind("tab: complete")
    readline.set_completer_delims(" \t\n;")
    # History file
    _histfile = os.path.join(os.path.expanduser("~"), ".imo_interactive_history")
    try:
        readline.read_history_file(_histfile)
        readline.set_history_length(500)
    except OSError:
        pass

    def _save_readline_history():
        try:
            readline.write_history_file(_histfile)
        except OSError:
            pass
except ImportError:
    pass
import subprocess
import sys
from datetime import datetime

import requests

# Import from agent module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent as agent_module
from res2md import memory_to_tex as res2md_memory_to_tex

# --- Fast models for cheap tex composition/fixing (per provider) ---
# For kimi: no fast/cheap model, use current model with thinking disabled
FAST_MODELS = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "kimi": None,  # Use current model with thinking disabled
}

# Legacy constants for backward compatibility
FLASH_MODEL = FAST_MODELS["gemini"]
FLASH_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{FLASH_MODEL}:generateContent"

TEX_PREAMBLE = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{geometry}
\geometry{a4paper, margin=1in}

"""

TEX_COMPOSE_TEMPLATE = """You are a LaTeX expert. Convert the following mathematical solution and verification report into a single, valid LaTeX document.

Requirements:
- Preserve all mathematical content exactly; use $...$ for inline math, \\[...\\] for display math
- Convert markdown **bold** to \\textbf{}, ### sections to \\section{}, etc.
- Output ONLY the complete LaTeX source, no explanations

Structure:
1. Problem Statement
2. Solution (Summary + Detailed Solution)
3. Verification Report

=== PROBLEM ===
<<<PROBLEM>>>

=== SOLUTION ===
<<<SOLUTION>>>

=== VERIFICATION ===
<<<VERIFICATION>>>
"""

TEX_FIX_TEMPLATE = """Fix the LaTeX compilation errors. Output ONLY the corrected complete LaTeX source.

=== ERROR ===
<<<ERROR>>>

=== LATEX ===
<<<LATEX>>>
"""


def call_fast_model(prompt: str, api_key: str, provider: str = "gemini", 
                    model_name: str = None, enable_thinking: bool = True, timeout: int = 60) -> str:
    """Call the fast/cheap model for the specified provider.
    
    For kimi: if model_name is provided, use it with thinking disabled (for LaTeX tasks).
    """
    provider = provider.lower()
    
    if provider == "gemini":
        model = model_name or FAST_MODELS.get("gemini", "gemini-2.0-flash")
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "topP": 1.0},
        }
        headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}
        resp = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=timeout)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    
    elif provider == "openai":
        model = model_name or FAST_MODELS.get("openai", "gpt-4o-mini")
        api_url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        resp = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    
    elif provider == "kimi":
        # For kimi: use provided model_name (e.g., kimi-k2) or fallback to a default
        # For LaTeX tasks, thinking is disabled by default
        model = model_name or "kimi-k2"  # Default to kimi-k2 if no model specified
        api_url = "https://api.moonshot.cn/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        # For kimi-k2 or kimi-k1.5, disable thinking when enable_thinking is False
        if not enable_thinking:
            payload["extra_body"] = {
                "enable_thinking": False
            }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        resp = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    
    else:
        # Fallback to Gemini
        return call_fast_model(prompt, api_key, "gemini", model_name, enable_thinking, timeout)


# Backward compatibility alias
def call_flash(prompt: str, api_key: str, timeout: int = 60) -> str:
    """Legacy function, defaults to Gemini flash model."""
    return call_fast_model(prompt, api_key, "gemini", timeout)


def call_fast_model_chat(
    system_prompt: str,
    contents: list,
    api_key: str,
    provider: str = "gemini",
    model_name: str = None,
    enable_thinking: bool = True,
    timeout: int = 60,
) -> str:
    """Multi-turn chat with fast model. 
    contents format for gemini: [{"role":"user","parts":[{"text":"..."}]}, ...]
    contents format for openai/kimi: [{"role":"user","content":"..."}, ...]
    
    For kimi: if model_name is provided, use it with thinking disabled (for LaTeX tasks).
    """
    provider = provider.lower()
    
    if provider == "gemini":
        model = model_name or FAST_MODELS.get("gemini", "gemini-2.0-flash")
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "systemInstruction": {"role": "system", "parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.3, "topP": 1.0},
        }
        headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}
        resp = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=timeout)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    
    elif provider == "openai":
        # Convert Gemini-style contents to OpenAI-style messages
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        
        for item in contents:
            role = item.get("role", "user")
            # Handle Gemini format: parts[0].text
            if "parts" in item and item["parts"]:
                content = item["parts"][0].get("text", "")
            else:
                content = item.get("content", "")
            messages.append({"role": role, "content": content})
        
        model = model_name or FAST_MODELS.get("openai", "gpt-4o-mini")
        api_url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        resp = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    
    elif provider == "kimi":
        # Convert Gemini-style contents to OpenAI-style messages
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        
        for item in contents:
            role = item.get("role", "user")
            # Handle Gemini format: parts[0].text
            if "parts" in item and item["parts"]:
                content = item["parts"][0].get("text", "")
            else:
                content = item.get("content", "")
            messages.append({"role": role, "content": content})
        
        # For kimi: use provided model_name (e.g., kimi-k2) or fallback
        # For LaTeX tasks, thinking is disabled by default
        model = model_name or "kimi-k2"
        api_url = "https://api.moonshot.cn/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
        }
        # For kimi-k2 or kimi-k1.5, disable thinking when enable_thinking is False
        if not enable_thinking:
            payload["extra_body"] = {
                "enable_thinking": False
            }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        resp = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    
    else:
        # Fallback to Gemini
        return call_fast_model_chat(system_prompt, contents, api_key, "gemini", model_name, enable_thinking, timeout)


# Backward compatibility alias
def call_flash_chat(
    system_prompt: str,
    contents: list,
    api_key: str,
    timeout: int = 60,
) -> str:
    """Legacy function, defaults to Gemini flash model."""
    return call_fast_model_chat(system_prompt, contents, api_key, "gemini", None, True, timeout)


EDIT_PROBLEM_SYSTEM = """You help the user formulate or refine a mathematical problem for an IMO-style solver.

- Ask clarifying questions. Suggest structure (hypotheses, goal, constraints).
- Use TeX for math: $n$, $\\mathbb{R}$, $$display math$$
- When the user indicates they're done (e.g. "done", "that's it"), output ONLY the final problem statement, nothing else.
- The final problem should be self-contained, clear, and suitable for rigorous proof.
- Be concise."""


def _extract_latex(out: str) -> str:
    out = out.strip()
    if out.startswith("```"):
        lines = out.split("\n")
        if "latex" in lines[0].lower() or "tex" in lines[0].lower() or lines[0] == "```":
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        out = "\n".join(lines)
    return out.strip()


def compose_tex_with_fast_model(problem: str, solution: str, verification: str, api_key: str, 
                                 provider: str = "gemini", model_name: str = None) -> str:
    """Compose LaTeX using the provider's fast/cheap model.
    
    For kimi: uses the current model with thinking disabled (no fast/cheap model available).
    """
    prompt = TEX_COMPOSE_TEMPLATE.replace("<<<PROBLEM>>>", problem)
    prompt = prompt.replace("<<<SOLUTION>>>", solution or "(No solution yet)")
    prompt = prompt.replace("<<<VERIFICATION>>>", verification or "(No verification yet)")
    
    # For kimi, disable thinking for LaTeX tasks (no fast model, but thinking disabled)
    if provider.lower() == "kimi":
        out = call_fast_model(prompt, api_key, provider, model_name, enable_thinking=False)
    else:
        out = call_fast_model(prompt, api_key, provider, model_name)
    
    latex = _extract_latex(out)
    if "\\documentclass" in latex:
        return latex
    if "\\begin{document}" in latex and "\\end{document}" in latex:
        return TEX_PREAMBLE + "\n" + latex
    return TEX_PREAMBLE + "\n\\begin{document}\n\n" + latex + "\n\n\\end{document}\n"


def fix_tex_with_fast_model(latex: str, error: str, api_key: str, 
                            provider: str = "gemini", model_name: str = None) -> str:
    """Fix LaTeX errors using the provider's fast/cheap model.
    
    For kimi: uses the current model with thinking disabled (no fast/cheap model available).
    """
    prompt = TEX_FIX_TEMPLATE.replace("<<<ERROR>>>", error).replace("<<<LATEX>>>", latex)
    
    # For kimi, disable thinking for LaTeX tasks (no fast model, but thinking disabled)
    if provider.lower() == "kimi":
        return _extract_latex(call_fast_model(prompt, api_key, provider, model_name, enable_thinking=False))
    else:
        return _extract_latex(call_fast_model(prompt, api_key, provider, model_name))


# Backward compatibility aliases
def compose_tex_with_flash(problem: str, solution: str, verification: str, api_key: str) -> str:
    """Legacy function, defaults to Gemini flash model."""
    return compose_tex_with_fast_model(problem, solution, verification, api_key, "gemini")


def fix_tex_with_flash(latex: str, error: str, api_key: str) -> str:
    """Legacy function, defaults to Gemini flash model."""
    return fix_tex_with_fast_model(latex, error, api_key, "gemini")


def compile_latex(tex_path: str, work_dir: str) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", os.path.basename(tex_path)],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        err = (result.stderr or "") + (result.stdout or "")
        return result.returncode == 0, err
    except subprocess.TimeoutExpired:
        return False, "Compilation timed out"
    except FileNotFoundError:
        return False, "pdflatex not found. Install TeX Live or MacTeX."


def open_pdf(pdf_path: str) -> None:
    path = os.path.abspath(pdf_path)
    if not os.path.exists(path):
        print(f"PDF not found: {path}")
        return
    system = platform.system()
    if system == "Darwin":
        subprocess.run(["open", path], check=False)
    elif system == "Linux":
        subprocess.run(["xdg-open", path], check=False)
    elif system == "Windows":
        os.startfile(path)  # type: ignore
    else:
        print(f"PDF: {path}")


def export_to_pdf(
    problem: str,
    solution: str,
    verification: str,
    output_dir: str,
    base_name: str,
    api_key: str,
    max_attempts: int = 5,
    provider: str = "gemini",
    model_name: str = None,
) -> bool:
    """Export solution to PDF using the provider's fast model for LaTeX composition.
    
    For kimi: no fast model available, uses current model with thinking disabled.
    """
    tex_name = f"{base_name}-temp.tex"
    tex_path = os.path.join(output_dir, tex_name)
    pdf_path = os.path.join(output_dir, f"{base_name}-temp.pdf")
    os.makedirs(output_dir, exist_ok=True)

    # Get the model name for display
    provider_lower = provider.lower()
    fast_model = FAST_MODELS.get(provider_lower)
    
    if provider_lower == "kimi":
        # For kimi: no fast model, use current model with thinking disabled
        display_model = model_name or "kimi-k2"
        print(f"  Composing LaTeX using {provider} ({display_model}, thinking disabled)...")
    else:
        # For other providers: use fast/cheap model
        display_model = fast_model or model_name or "default"
        print(f"  Composing LaTeX using {provider}'s fast model ({display_model})...")

    skip_fast_fix = False
    try:
        latex = compose_tex_with_fast_model(problem, solution, verification or "Verification pending.", 
                                             api_key, provider, model_name)
    except Exception as e:
        print(f"  Fast model failed ({e}), using res2md fallback...")
        skip_fast_fix = True
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            tmp = f.name
        try:
            mem = {"problem_statement": problem, "solution": solution}
            res2md_memory_to_tex(mem, tmp)
            with open(tmp, "r", encoding="utf-8") as f:
                latex = f.read()
            if verification:
                from res2md import markdown_to_latex
                latex = latex.replace(
                    "\\end{document}",
                    "\n\\section{Verification Report}\n\n" + markdown_to_latex(verification) + "\n\\end{document}",
                )
        finally:
            os.unlink(tmp)

    if "\\begin{document}" not in latex:
        latex = latex.rstrip() + "\n\n\\begin{document}\n\n\\end{document}\n"
    if "\\end{document}" not in latex:
        latex = latex.rstrip() + "\n\n\\end{document}\n"

    for attempt in range(max_attempts):
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex)
        success, err = compile_latex(tex_path, output_dir)
        if success:
            print(f"  ✓ PDF: {pdf_path}")
            open_pdf(pdf_path)
            return True
        err_snippet = "\n".join(err.strip().split("\n")[-60:])
        print(f"  Compile attempt {attempt + 1}/{max_attempts} failed")
        if attempt < max_attempts - 1 and not skip_fast_fix:
            try:
                latex = fix_tex_with_fast_model(latex, err_snippet, api_key, provider, model_name)
            except Exception as e:
                print(f"  Fast model fix failed: {e}")
                break
    print("  Could not fix LaTeX.")
    return False


def _agent_worker(
    problem_statement: str,
    other_prompts: list,
    memory_file: Optional[str],
    resume: bool,
    log_dir: str,
    base_name: str,
    result_queue: "multiprocessing.Queue",
    streaming: bool = True,
    show_thinking: bool = True,
    interactive: bool = True,
    provider_name: str = "gemini",
    model_name: str = None
) -> None:
    """Run agent in subprocess; put result in queue. Terminating process kills in-flight API calls."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # 根据provider_name选择不同的模块
    try:
        if provider_name.lower() in ["openai", "gpt"]:
            import agent_oai as ag
            api_key = os.environ.get("OPENAI_API_KEY", "")
        elif provider_name.lower() in ["kimi", "moonshot"]:
            # 使用Kimi专用模块
            print("  Using Kimi API")
            import agent_kimi as ag
            api_key = os.environ.get("KIMI_API_KEY", "")
        else:
            # 默认使用Gemini
            import agent as ag
            api_key = os.environ.get("GOOGLE_API_KEY", "")
    except ImportError as e:
        print(f"  Error importing agent module for {provider_name}: {e}")
        print("  Falling back to default agent module")
        import agent as ag
        api_key = os.environ.get("GOOGLE_API_KEY", "")

    def on_result(prob, sol, verif, _iter):
        print(f"  [iter {_iter}] Exporting PDF...")
        export_to_pdf(prob, sol, verif, log_dir, base_name, api_key, provider=provider_name, model_name=model_name)

    log_path = os.path.join(log_dir, f"{base_name}_interactive.log")
    ag.set_log_file(log_path)

    try:
        # 设置模型名称
        if model_name:
            # 优先使用set_model函数（agent_kimi支持）
            if hasattr(ag, "set_model"):
                ag.set_model(model_name)
            # 否则尝试直接设置MODEL_NAME
            elif hasattr(ag, "MODEL_NAME"):
                ag.MODEL_NAME = model_name
                print(f"  Set model to {model_name}")

        # 启动agent处理
        sol = ag.agent(
            problem_statement,
            other_prompts,
            memory_file=memory_file,
            resume_from_memory=resume,
            on_iteration_result=on_result,
            streaming=streaming,
            show_thinking=show_thinking,
            interactive=interactive
        )
        result_queue.put(("ok", sol))
    except Exception as e:
        result_queue.put(("error", str(e)))
    finally:
        ag.close_log_file()


def list_memory_files(log_dir: str) -> list:
    """List .mem files in log_dir (agent memory files)."""
    if not os.path.isdir(log_dir):
        return []
    return sorted(
        [f for f in os.listdir(log_dir) if f.endswith(".mem")],
        key=lambda x: os.path.getmtime(os.path.join(log_dir, x)),
        reverse=True,
    )


def save_problem_to_file(problem_content: str, file_path: str) -> bool:
    """Save problem content to a file. Returns True if successful."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(problem_content)
        return True
    except Exception as e:
        print(f"  Error saving problem: {e}")
        return False

def render_status_bar(problem_loaded: bool, memory_file: Optional[str], solution: bool, prompts: int,
                     edit_mode: bool = False, original_problem: Optional[str] = None,
                     streaming: bool = True, thinking: bool = True, interactive: bool = True,
                     provider: str = "gemini", model: str = None) -> str:
    """Render a compact status line like Claude Code."""
    parts = []
    if problem_loaded:
        parts.append("problem ✓")
    else:
        parts.append("problem —")
    if memory_file:
        parts.append(f"mem:{os.path.basename(memory_file)}")
    parts.append("solution ✓" if solution else "solution —")

    # Add API provider and model info
    if provider:
        provider_info = provider
        if model:
            provider_info += f":{model.split('-')[0]}"  # Show first part of model name
        parts.append(provider_info)

    if edit_mode:
        if original_problem:
            parts.append(f"editing:{os.path.basename(original_problem)}")
        else:
            parts.append("edit")

    # Add mode indicators
    mode_parts = []
    if streaming:
        mode_parts.append("stream")
    if thinking:
        mode_parts.append("think")
    if interactive:
        mode_parts.append("interact")

    if mode_parts:
        parts.append("+".join(mode_parts))

    # Add token quota status if available
    if hasattr(agent_module, 'TOKEN_QUOTA_EXCEEDED') and agent_module.TOKEN_QUOTA_EXCEEDED:
        parts.append("⚠️quota:exceeded")
    elif hasattr(agent_module, 'TOKEN_QUOTA_WARNING') and agent_module.TOKEN_QUOTA_WARNING:
        parts.append("⚠️quota:low")

    if prompts:
        parts.append(f"+{prompts} prompt(s)")
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Interactive IMO agent (Claude Code style)")
    # Main positional argument (can be problem or memory file)
    parser.add_argument("path", nargs="?", help="Problem file or memory file (.mem) to load")

    # Add specific problem file loading option
    parser.add_argument("--problem", "-f", type=str,
                       help="Explicitly load a problem file (specify problem filename)")

    # Add specific memory file loading option
    parser.add_argument("--mem", "--memory", type=str,
                       help="Explicitly load a memory file (.mem) to resume from a previous session")

    # Add option to list available memory files
    parser.add_argument("--list-mem", action="store_true",
                       help="List available memory files and exit")

    # Add command line options for streaming, thinking, interactive modes
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming output")
    parser.add_argument("--no-thinking", action="store_true", help="Hide thinking process")
    parser.add_argument("--no-interactive", action="store_true", help="Disable interactive mode")

    # Directory for logs and memory files
    parser.add_argument("--log-dir", "-d", default="run_logs", help="Directory for logs and memory files")

    # API provider selection options
    parser.add_argument("--provider", "-p", choices=["gemini", "openai", "kimi"],
                       help="Select API provider (gemini, openai, kimi). Kimi provider uses kimi-k1.5 by default for thinking capability.")
    parser.add_argument("--model", "-m", type=str,
                       help="Specify model name for the selected provider")
    parser.add_argument("--list-providers", action="store_true",
                       help="List available API providers and exit")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # State
    problem_statement = ""
    other_prompts = []
    memory_file: Optional[str] = None
    base_name = "interactive"
    solution: Optional[str] = None
    full_verification = ""
    api_key = None
    edit_history: list = []
    in_edit_mode: bool = False  # True = bare input goes to problem-editing dialogue
    original_problem_path: Optional[str] = None  # Path to the original problem being edited
    # New mode settings
    enable_streaming: bool = True
    enable_thinking: bool = True
    enable_interactive: bool = True
    # API provider settings
    provider_name: str = os.getenv("DEFAULT_MODEL_PROVIDER", "gemini")
    model_name: Optional[str] = None

    def load_from_memory(path: str) -> bool:
        nonlocal problem_statement, other_prompts, solution, full_verification, memory_file, base_name
        path = os.path.abspath(path)
        if not os.path.exists(path):
            print(f"  File not found: {path}")
            return False
        if not path.endswith(".mem"):
            print(f"  Expected .mem file. Use /load for memory or /problem for problem file.")
            return False
        mem = agent_module.load_memory(path)
        if not mem:
            return False
        problem_statement = mem.get("problem_statement", "")
        other_prompts = mem.get("other_prompts", [])
        solution = mem.get("solution")
        full_verification = mem.get("full_verification", mem.get("verify", ""))
        memory_file = path
        base_name = os.path.splitext(os.path.basename(path))[0]
        return True

    def load_from_problem(path: str) -> bool:
        nonlocal problem_statement, memory_file, base_name
        path = os.path.abspath(path)
        if not os.path.exists(path):
            print(f"  File not found: {path}")
            return False
        problem_statement = agent_module.read_file_content(path)
        base_name = os.path.splitext(os.path.basename(path))[0]
        memory_file = os.path.join(log_dir, f"{base_name}.mem")
        return True

    # Handle listing providers
    if args.list_providers:
        print("Available API providers:")
        available_providers = model_providers.get_available_providers()
        if not available_providers:
            print("  No API providers found. Please set at least one API key environment variable:")
            print("  - GOOGLE_API_KEY for Gemini")
            print("  - OPENAI_API_KEY for OpenAI")
            print("  - KIMI_API_KEY for Kimi")
        else:
            for provider in available_providers:
                print(f"  - {provider}")
        sys.exit(0)

    # Handle listing memory files
    if args.list_mem:
        print("Available memory files in", log_dir, ":")
        mem_files = list_memory_files(log_dir)
        if not mem_files:
            print("  No memory files found.")
        else:
            for i, mem_file in enumerate(mem_files):
                mem_path = os.path.join(log_dir, mem_file)
                try:
                    # Try to load memory to display problem info
                    mem_data = agent_module.load_memory(mem_path)
                    problem_preview = mem_data.get('problem_statement', 'No problem statement')
                    # Truncate for display
                    if len(problem_preview) > 60:
                        problem_preview = problem_preview[:57] + "..."
                    # Clean up newlines and format
                    problem_preview = problem_preview.replace('\n', ' ').strip()
                    print(f"  [{i+1}] {mem_file} - {problem_preview}")
                except:
                    print(f"  [{i+1}] {mem_file}")
        sys.exit(0)

    # Priority 1: Explicit memory file (--mem)
    if args.mem:
        mem_path = args.mem
        # If not absolute path, assume it's in log_dir
        if not os.path.isabs(mem_path):
            # Handle both with and without .mem extension
            if not mem_path.endswith('.mem'):
                mem_path = mem_path + '.mem'
            mem_path = os.path.join(log_dir, mem_path)

        print(f"Loading memory file: {mem_path}")
        if not load_from_memory(mem_path):
            sys.exit(1)

    # Priority 2: Explicit problem file (--problem)
    elif args.problem:
        problem_path = args.problem
        # If not absolute path and doesn't exist in current dir, try problems/ directory
        if not os.path.isabs(problem_path) and not os.path.exists(problem_path):
            problems_dir = os.path.join(script_dir, "..", "problems")
            if os.path.exists(os.path.join(problems_dir, problem_path)):
                problem_path = os.path.join(problems_dir, problem_path)
        
        p = os.path.abspath(problem_path)
        print(f"Loading problem file: {p}")
        if not load_from_problem(p):
            sys.exit(1)
        # If agent memory exists for this problem, load it to resume
        if memory_file and os.path.exists(memory_file):
            load_from_memory(memory_file)

    # Priority 3: Path argument (could be problem or memory)
    elif args.path:
        p = os.path.abspath(args.path)
        if p.endswith(".mem"):
            if not load_from_memory(p):
                sys.exit(1)
        else:
            if not load_from_problem(p):
                sys.exit(1)
            # If agent memory exists for this problem, load it to resume
            if memory_file and os.path.exists(memory_file):
                load_from_memory(memory_file)

    # Set up run modes from command line arguments
    if args.no_streaming:
        enable_streaming = False
        print("  Streaming output: Disabled (by command line argument)")
    if args.no_thinking:
        enable_thinking = False
        print("  Thinking process: Hidden (by command line argument)")
    if args.no_interactive:
        enable_interactive = False
        print("  Interactive mode: Disabled (by command line argument)")

    # 选择API提供商
    provider_name = args.provider if args.provider else os.getenv("DEFAULT_MODEL_PROVIDER", "gemini")
    model_name = args.model

    # 创建提供商实例
    print(f"  Using {provider_name} API" + (f" with model {model_name}" if model_name else ""))
    try:
        model_provider = model_providers.create_provider(provider_name, model_name=model_name)
        # 更新状态以记录当前提供商和模型
        provider_name = model_provider.get_name().lower()
        model_name = model_provider.model_name
    except Exception as e:
        print(f"  Error initializing {provider_name} provider: {e}")
        print("  Falling back to Gemini API")
        model_provider = model_providers.GeminiProvider()
        provider_name = "gemini"
        model_name = model_provider.model_name

    # 获取API密钥
    api_key = model_provider.get_api_key()
    log_path = os.path.join(log_dir, f"{base_name}_interactive.log")
    agent_module.set_log_file(log_path)

    # 检查模型能力
    print("  Checking model capabilities...")
    model_provider.check_capabilities()

    # 更新状态基于模型支持的功能
    if not model_provider.streaming_supported:
        enable_streaming = False
        enable_thinking = False
        print("  Note: Selected model does not support streaming or thinking display.")
        print("  These features have been automatically disabled.")

    def do_edit_problem(user_msg: str) -> bool:
        """Chat with agent to draft/edit problem. Returns True if problem was finalized."""
        nonlocal problem_statement, base_name, memory_file, edit_history, in_edit_mode, original_problem_path
        # Build contents for API
        contents = list(edit_history)
        contents.append({"role": "user", "parts": [{"text": user_msg}]})
        try:
            reply = call_fast_model_chat(EDIT_PROBLEM_SYSTEM, contents, api_key, provider_name, model_name)
        except Exception as e:
            print(f"  API error: {e}")
            return False
        # Check if reply looks like final problem (user said "done" etc)
        is_done = user_msg.strip().lower() in (
            "done", "that's it", "finish", "finalize", "save", "/done"
        ) or "i'm done" in user_msg.lower() or "output only the final problem" in user_msg.lower()
        if is_done:
            # Use reply as the final problem
            problem_statement = reply.strip()
            if problem_statement:
                # If editing existing problem, don't change the base_name
                if not original_problem_path:
                    base_name = "draft"
                    memory_file = os.path.join(log_dir, "draft.mem")
                edit_history = []
                in_edit_mode = False
                print(f"  Problem saved ({len(problem_statement)} chars). Use /run to solve.")
                # If we were editing an existing problem, prompt to save as new file
                if original_problem_path:
                    print("  Use /save_as <filename> to save the edited problem to a new file.")
                return True
            in_edit_mode = False
            return True
        # Normal turn: show reply, update history
        edit_history.append({"role": "user", "parts": [{"text": user_msg}]})
        edit_history.append({"role": "model", "parts": [{"text": reply}]})
        print(f"  Agent: {reply}")
        return False

    def do_export():
        if not solution:
            print("  No solution yet. Run /run first.")
            return
        export_to_pdf(
            problem_statement, solution, full_verification,
            log_dir, base_name, api_key, provider=provider_name, model_name=model_name,
        )

    def do_run(streaming=True, show_thinking=True, interactive=True,
               use_provider=None, use_model=None):
        nonlocal solution, full_verification, provider_name, model_name

        # 如果提供了特定的provider和model，则使用它们
        current_provider = use_provider if use_provider is not None else provider_name
        current_model = use_model if use_model is not None else model_name

        # Run agent in subprocess so Ctrl+C can terminate in-flight API calls
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_agent_worker,
            args=(
                problem_statement,
                other_prompts,
                memory_file,
                bool(memory_file and solution),
                log_dir,
                base_name,
                result_queue,
                streaming,
                show_thinking,
                interactive,
                current_provider,
                current_model,
            ),
        )
        mode_str = []
        if streaming:
            mode_str.append("streaming")
        if show_thinking:
            mode_str.append("thinking")
        if interactive:
            mode_str.append("interactive")

        mode_info = f" [{', '.join(mode_str)}]" if mode_str else ""
        provider_info = f" using {current_provider}" + (f":{current_model}" if current_model else "")
        print(f"  Running agent (+{len(other_prompts)} prompt(s)){mode_info}{provider_info}. Ctrl+C to abort.")
        process.start()
        try:
            while process.is_alive():
                process.join(timeout=0.5)
        except KeyboardInterrupt:
            print("\n  Terminating agent (aborting API calls)...")
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
            print("  Interrupted. Back to prompt.")
            return

        try:
            status, sol = result_queue.get_nowait()
            if status == "ok" and sol:
                solution = sol
                print("  Solution found!")
            elif status == "error":
                print(f"  Agent error: {sol}")
        except Exception:  # Empty or terminated before result
            pass
        if memory_file and os.path.exists(memory_file):
            with open(memory_file, "r", encoding="utf-8") as f:
                mem = json.load(f)
                solution = mem.get("solution") or solution
                full_verification = mem.get("full_verification", mem.get("verify", ""))

    # Banner
    print()
    print("  IMO Interactive Agent")
    print("  Type /help for commands.")
    if not problem_statement:
        print("  No problem loaded. Type to chat with agent, or /edit to draft, /load or /problem to load.")
    print()

    # Main loop
    while True:
        status = render_status_bar(
            bool(problem_statement), memory_file, bool(solution), len(other_prompts),
            in_edit_mode, original_problem_path,
            enable_streaming, enable_thinking, enable_interactive,
            provider_name, model_name
        )
        try:
            line = input(f"  [{status}]\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exit.")
            break

        if not line:
            continue

        # Bare "quit"/"exit" without /
        if line.lower() in ("quit", "q", "exit"):
            break

        # Slash commands (Claude Code style)
        if line.startswith("/"):
            parts = line[1:].split(maxsplit=1)
            cmd = (parts[0] if parts else "").lower()
            rest = parts[1] if len(parts) > 1 else ""

            if cmd in ("quit", "q", "exit"):
                break
            elif cmd == "help" or cmd == "h":
                print("  /run, /r          Run the agent")
                print("  /load <path>      Load agent memory file (.mem)")
                print("  /problem <path>   Load problem file")
                print("  /edit             Draft or refine problem with agent (no problem = direct chat)")
                print("  /edit_existing    Browse and edit an existing problem file")
                print("  /done             Save current draft as problem (in edit mode)")
                print("  /save_as <name>   Save edited problem to a new file")
                print("  /prompt <text>    Add prompt for next run (or type without /)")
                print("  /export, /e       Generate PDF (solution + verification)")
                print("  /status, /s       Show state")
                print("  /list, /l         List memory files in log-dir")
                print("  /clear            Clear prompts")
                print("  /streaming on|off Enable/disable streaming output")
                print("  /thinking on|off  Enable/disable thinking process display")
                print("  /interactive on|off Enable/disable interactive mode")
                print("  /run_mode         Show current run mode settings")
                print("  /quota            Check API token quota status")
                print("  /provider <name>  Switch API provider (gemini, openai, kimi)")
                print("  /model <name>     Set specific model for current provider")
                print("  /providers        List available API providers")
                print("  /help, /h         This message")
                print("  /quit, /exit      Exit")
            elif cmd in ("run", "r"):
                if not problem_statement:
                    print("  Load a problem (/problem) or memory (/load) first.")
                    continue
                try:
                    do_run(
                        streaming=enable_streaming,
                        show_thinking=enable_thinking,
                        interactive=enable_interactive
                    )
                except KeyboardInterrupt:
                    print("\n  Interrupted. Back to prompt.")
            elif cmd in ("export", "e"):
                try:
                    do_export()
                except KeyboardInterrupt:
                    print("\n  Interrupted. Back to prompt.")
            elif cmd == "load":
                if rest:
                    load_from_memory(rest.strip())
                else:
                    mems = list_memory_files(log_dir)
                    if not mems:
                        print(f"  No .mem files in {log_dir}")
                    else:
                        print("  Memory files (use /load <name>):")
                        for m in mems[:15]:
                            print(f"    {m}")
            elif cmd == "problem":
                if rest:
                    load_from_problem(rest.strip())
                else:
                    print("  Usage: /problem <path>")
            elif cmd in ("prompt", "add", "p"):
                if rest:
                    other_prompts.append(rest)
                    print(f"  +prompt #{len(other_prompts)}")
                else:
                    print("  Usage: /prompt <instruction>")
            elif cmd in ("status", "s"):
                print(f"  Problem: {len(problem_statement)} chars")
                print(f"  Memory: {memory_file or '—'}")
                print(f"  Prompts: {len(other_prompts)}")
                print(f"  Solution: {'Yes' if solution else 'No'}")
                print(f"  Verification: {'Yes' if full_verification else 'No'}")
            elif cmd == "list" or cmd == "l":
                mems = list_memory_files(log_dir)
                if not mems:
                    print(f"  No .mem files in {log_dir}")
                else:
                    for m in mems[:20]:
                        print(f"    {m}")
            elif cmd == "clear":
                other_prompts.clear()
                in_edit_mode = False
                edit_history.clear()
                original_problem_path = None
                print("  Prompts cleared.")
            elif cmd in ("edit", "edit_problem"):
                in_edit_mode = True
                if problem_statement:
                    try:
                        do_edit_problem(f"Current draft:\n\n{problem_statement}\n\nHelp me refine it. What should I change?")
                    except Exception as e:
                        print(f"  Error: {e}")
                    print("  Type your feedback or /done to save.")
                else:
                    print("  Edit mode. Chat with agent to draft a problem. Type /done when finished.")
            elif cmd == "done":
                if in_edit_mode or edit_history:
                    do_edit_problem("I'm done. Please output ONLY the final problem statement, nothing else.")
                else:
                    print("  Not in edit mode. Use /edit first.")
            elif cmd == "edit_existing":
                # Browse problems directory and select a file to edit
                problems_dir = os.path.join(os.path.dirname(script_dir), "problems")
                if not os.path.isdir(problems_dir):
                    print(f"  Problems directory not found: {problems_dir}")
                    continue

                problem_files = sorted([f for f in os.listdir(problems_dir)
                                       if os.path.isfile(os.path.join(problems_dir, f)) and
                                       (f.endswith(".md") or f.endswith(".txt"))])

                if not problem_files:
                    print("  No problem files found in problems directory.")
                    continue

                print("  Available problem files:")
                for i, f in enumerate(problem_files):
                    print(f"    [{i+1}] {f}")

                try:
                    selection = input("  Select a file to edit (number): ").strip()
                    if not selection.isdigit() or int(selection) < 1 or int(selection) > len(problem_files):
                        print("  Invalid selection.")
                        continue

                    selected_file = problem_files[int(selection) - 1]
                    selected_path = os.path.join(problems_dir, selected_file)

                    # Load the problem file
                    if not load_from_problem(selected_path):
                        continue

                    # Set up edit mode
                    original_problem_path = selected_path
                    in_edit_mode = True

                    # Start editing process
                    try:
                        do_edit_problem(f"Current problem:\n\n{problem_statement}\n\nHelp me refine it. What should I change?")
                    except Exception as e:
                        print(f"  Error: {e}")

                    print("  Type your feedback or /done to save.")
                    print("  When finished editing, use /save_as <name> to save to a new file.")

                except (ValueError, IndexError, KeyboardInterrupt):
                    print("  Operation cancelled.")

            elif cmd == "save_as":
                if not problem_statement:
                    print("  No problem to save. Edit a problem first.")
                    continue

                if not rest:
                    print("  Usage: /save_as <filename>")
                    print("  Example: /save_as new_problem")
                    continue

                # Create a filename with proper extension
                filename = rest.strip()
                if not (filename.endswith(".md") or filename.endswith(".txt")):
                    filename += ".md"  # Default to markdown

                # Save to problems directory
                problems_dir = os.path.join(os.path.dirname(script_dir), "problems")
                os.makedirs(problems_dir, exist_ok=True)

                new_path = os.path.join(problems_dir, filename)

                # Confirm if file exists
                if os.path.exists(new_path):
                    confirm = input(f"  File '{filename}' already exists. Overwrite? (y/N): ").lower()
                    if confirm != 'y':
                        print("  Save cancelled.")
                        continue

                # Save the file
                if save_problem_to_file(problem_statement, new_path):
                    print(f"  Problem saved to {filename}")

                    # Update state to use the new file
                    original_problem_path = new_path
                    base_name = os.path.splitext(filename)[0]
                    memory_file = os.path.join(log_dir, f"{base_name}.mem")
                else:
                    print("  Failed to save problem.")
            elif cmd == "streaming":
                if rest.lower() in ["on", "true", "yes", "1"]:
                    enable_streaming = True
                    print("  Streaming output: Enabled")
                    print("  Note: If the model doesn't support streaming, it will automatically fall back to non-streaming mode.")
                elif rest.lower() in ["off", "false", "no", "0"]:
                    enable_streaming = False
                    print("  Streaming output: Disabled")
                else:
                    print(f"  Current streaming mode: {'Enabled' if enable_streaming else 'Disabled'}")
                    print("  Usage: /streaming on|off")
            elif cmd == "thinking":
                if rest.lower() in ["on", "true", "yes", "1"]:
                    enable_thinking = True
                    print("  Thinking process: Visible")
                elif rest.lower() in ["off", "false", "no", "0"]:
                    enable_thinking = False
                    print("  Thinking process: Hidden")
                else:
                    print(f"  Current thinking mode: {'Visible' if enable_thinking else 'Hidden'}")
                    print("  Usage: /thinking on|off")
            elif cmd == "interactive":
                if rest.lower() in ["on", "true", "yes", "1"]:
                    enable_interactive = True
                    print("  Interactive mode: Enabled")
                elif rest.lower() in ["off", "false", "no", "0"]:
                    enable_interactive = False
                    print("  Interactive mode: Disabled")
                else:
                    print(f"  Current interactive mode: {'Enabled' if enable_interactive else 'Disabled'}")
                    print("  Usage: /interactive on|off")
            elif cmd == "run_mode":
                print("  Current run modes:")
                print(f"  - Streaming output: {'Enabled' if enable_streaming else 'Disabled'}")
                print(f"  - Thinking process: {'Visible' if enable_thinking else 'Hidden'}")
                print(f"  - Interactive mode: {'Enabled' if enable_interactive else 'Disabled'}")
                print("\n  Use /streaming, /thinking, or /interactive to change settings.")
            elif cmd == "quota":
                print("  Checking API token quota status...")
                # Reset quota check time to force a fresh check
                if hasattr(agent_module, 'QUOTA_CHECK_TIME'):
                    agent_module.QUOTA_CHECK_TIME = 0

                # Force a quota check
                if agent_module.check_token_quota(api_key):
                    if hasattr(agent_module, 'TOKEN_QUOTA_WARNING') and agent_module.TOKEN_QUOTA_WARNING:
                        print("  ⚠️ API token quota is running low. Consider spacing out API calls.")
                    else:
                        print("  ✓ API token quota is available.")
                else:
                    print("  ⚠️ API token quota is exceeded. Please wait for quota to reset.")
            elif cmd == "provider":
                if not rest:
                    print(f"  Current API provider: {provider_name}")
                    print("  Usage: /provider <name>")
                    print("  Available providers: gemini, openai, kimi")
                else:
                    new_provider = rest.lower()
                    if new_provider in ["gemini", "google", "openai", "gpt", "kimi", "moonshot"]:
                        try:
                            # 创建新的提供商实例
                            old_provider = provider_name
                            # 使用函数内的临时变量，不使用外部变量
                            model_provider = model_providers.create_provider(new_provider, model_name=model_name)
                            # 更新全局状态变量
                            provider_name = model_provider.get_name().lower()
                            model_name = model_provider.model_name
                            print(f"  Switched provider: {old_provider} → {provider_name}")

                            # 检查新提供商的能力
                            print("  Checking model capabilities...")
                            provider_supports_streaming = model_provider.check_capabilities()

                            if not provider_supports_streaming:
                                print("  Note: This provider does not support streaming or thinking.")
                        except Exception as e:
                            print(f"  Error switching provider: {e}")
                    else:
                        print(f"  Unknown provider: {new_provider}")
                        print("  Available providers: gemini, openai, kimi")
            elif cmd == "model":
                if not rest:
                    print(f"  Current model: {model_name or 'default'} (provider: {provider_name})")
                    print("  Usage: /model <name>")
                    if provider_name == "gemini":
                        print("  Available models: gemini-2.5-pro, gemini-1.5-flash, gemini-1.5-pro, ...")
                    elif provider_name == "openai":
                        print("  Available models: gpt-4o, gpt-4-turbo, gpt-3.5-turbo, ...")
                    elif provider_name == "kimi":
                        print("  Available models:")
                        print("    - kimi-k1.5 (default, with thinking)")
                        print("    - kimi-k1.5-preview (with thinking)")
                        print("    - kimi-k1.5-long-context (with thinking)")
                        print("    - moonshot-v1-128k (no thinking)")
                        print("    - moonshot-v1-32k (no thinking)")
                        print("    - moonshot-v1-8k (no thinking)")
                        print("  Note: Use kimi-k1.5 series for thinking capability")
                else:
                    model_name = rest
                    print(f"  Model set to: {model_name}")
                    print("  Note: Will use this model on next run.")
            elif cmd == "providers":
                print("  Available API providers:")
                available_providers = model_providers.get_available_providers()
                if not available_providers:
                    print("  No API keys configured. Please set environment variables:")
                    print("  - GOOGLE_API_KEY for Gemini")
                    print("  - OPENAI_API_KEY for OpenAI")
                    print("  - KIMI_API_KEY for Kimi")
                else:
                    for provider in available_providers:
                        if provider == provider_name:
                            print(f"  - {provider} (current)")
                        else:
                            print(f"  - {provider}")
                print(f"\n  Current provider: {provider_name}")
                print(f"  Current model: {model_name or 'default'}")
            else:
                print(f"  Unknown /{cmd}. Type /help.")
            continue

        # Bare input: no problem -> chat to draft; in_edit_mode -> chat; else -> add prompt
        if line:
            if not problem_statement or in_edit_mode:
                try:
                    do_edit_problem(line)
                except KeyboardInterrupt:
                    print("\n  Interrupted.")
            else:
                other_prompts.append(line)
                preview = line[:50] + "..." if len(line) > 50 else line
                print(f"  +prompt #{len(other_prompts)}: {preview}")

    agent_module.close_log_file()
    _save_readline_history()


def print_cli_help():
    """Print command line usage information"""
    print("\nCommand Line Usage:")
    print("  python interactive_agent.py [path] [options]")
    print("\nExamples:")
    print("  python interactive_agent.py                          # Start with no file loaded")
    print("  python interactive_agent.py problem.md               # Load a problem file")
    print("  python interactive_agent.py solution.mem             # Resume from memory file")
    print("  python interactive_agent.py --mem solution           # Resume from memory file (auto-adds .mem)")
    print("  python interactive_agent.py --list-mem               # List available memory files")
    print("  python interactive_agent.py --no-streaming           # Disable streaming output")
    print("  python interactive_agent.py --no-thinking            # Hide thinking process")
    print("  python interactive_agent.py --no-interactive         # Disable interactive mode")
    print("  python interactive_agent.py --log-dir custom_logs    # Use custom logs directory")
    print("\nAPI Provider options:")
    print("  python interactive_agent.py --provider openai        # Use OpenAI API")
    print("  python interactive_agent.py --provider kimi          # Use Kimi API")
    print("  python interactive_agent.py --model gpt-4o           # Specify model name")
    print("  python interactive_agent.py --list-providers         # List available API providers")
    print("\nOnce running, type /help for interactive commands.")

if __name__ == "__main__":
    # Print help if explicitly requested with --help or -h (argparse handles this)
    # Additional help can be shown with a custom argument
    if len(sys.argv) == 2 and sys.argv[1] in ['--examples', '--usage']:
        print_cli_help()
        sys.exit(0)

    main()
