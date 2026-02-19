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
import multiprocessing
import os
import platform
try:
    import readline  # noqa: F401 - enables arrow-key history
except ImportError:
    pass
import subprocess
import sys
from datetime import datetime
from typing import Optional, Tuple

import requests

# Import from agent module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent as agent_module
from res2md import memory_to_tex as res2md_memory_to_tex

# --- Flash model for cheap tex composition/fixing ---
FLASH_MODEL = "gemini-3-flash-preview"
# FLASH_MODEL = "gemini-2.0-flash"
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


def call_flash(prompt: str, api_key: str, timeout: int = 60) -> str:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "topP": 1.0},
    }
    headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}
    resp = requests.post(FLASH_API_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


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


def compose_tex_with_flash(problem: str, solution: str, verification: str, api_key: str) -> str:
    prompt = TEX_COMPOSE_TEMPLATE.replace("<<<PROBLEM>>>", problem)
    prompt = prompt.replace("<<<SOLUTION>>>", solution or "(No solution yet)")
    prompt = prompt.replace("<<<VERIFICATION>>>", verification or "(No verification yet)")
    out = call_flash(prompt, api_key)
    latex = _extract_latex(out)
    if "\\documentclass" in latex:
        return latex
    if "\\begin{document}" in latex and "\\end{document}" in latex:
        return TEX_PREAMBLE + "\n" + latex
    return TEX_PREAMBLE + "\n\\begin{document}\n\n" + latex + "\n\n\\end{document}\n"


def fix_tex_with_flash(latex: str, error: str, api_key: str) -> str:
    prompt = TEX_FIX_TEMPLATE.replace("<<<ERROR>>>", error).replace("<<<LATEX>>>", latex)
    return _extract_latex(call_flash(prompt, api_key))


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
) -> bool:
    tex_name = f"{base_name}-temp.tex"
    tex_path = os.path.join(output_dir, tex_name)
    pdf_path = os.path.join(output_dir, f"{base_name}-temp.pdf")
    os.makedirs(output_dir, exist_ok=True)

    skip_flash_fix = False
    try:
        print("  Composing LaTeX...")
        latex = compose_tex_with_flash(problem, solution, verification or "Verification pending.", api_key)
    except Exception as e:
        print(f"  Flash failed ({e}), using res2md fallback...")
        skip_flash_fix = True
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
        if attempt < max_attempts - 1 and not skip_flash_fix:
            try:
                latex = fix_tex_with_flash(latex, err_snippet, api_key)
            except Exception as e:
                print(f"  Flash fix failed: {e}")
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
) -> None:
    """Run agent in subprocess; put result in queue. Terminating process kills in-flight API calls."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import agent as ag

    def on_result(prob, sol, verif, _iter):
        print(f"  [iter {_iter}] Exporting PDF...")
        export_to_pdf(prob, sol, verif, log_dir, base_name, os.environ.get("GOOGLE_API_KEY", ""))

    log_path = os.path.join(log_dir, f"{base_name}_interactive.log")
    ag.set_log_file(log_path)
    try:
        sol = ag.agent(
            problem_statement,
            other_prompts,
            memory_file=memory_file,
            resume_from_memory=resume,
            on_iteration_result=on_result,
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


def render_status_bar(problem_loaded: bool, memory_file: Optional[str], solution: bool, prompts: int) -> str:
    """Render a compact status line like Claude Code."""
    parts = []
    if problem_loaded:
        parts.append("problem ✓")
    else:
        parts.append("problem —")
    if memory_file:
        parts.append(f"mem:{os.path.basename(memory_file)}")
    parts.append("solution ✓" if solution else "solution —")
    if prompts:
        parts.append(f"+{prompts} prompt(s)")
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Interactive IMO agent (Claude Code style)")
    parser.add_argument("path", nargs="?", help="Problem file or memory file (.mem) to load")
    parser.add_argument("--log-dir", "-d", default="run_logs", help="Directory for logs and memory files")
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

    # Initial load from argparse
    if args.path:
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

    api_key = agent_module.get_api_key()
    log_path = os.path.join(log_dir, f"{base_name}_interactive.log")
    agent_module.set_log_file(log_path)

    def do_export():
        if not solution:
            print("  No solution yet. Run /run first.")
            return
        export_to_pdf(
            problem_statement, solution, full_verification,
            log_dir, base_name, api_key,
        )

    def do_run():
        nonlocal solution, full_verification

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
            ),
        )
        print(f"  Running agent (+{len(other_prompts)} prompt(s)). Ctrl+C to abort.")
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
    print("  Type /help for commands. Plain text = additional prompt.")
    print()

    # Main loop
    while True:
        status = render_status_bar(bool(problem_statement), memory_file, bool(solution), len(other_prompts))
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
                print("  /run, /r        Run the agent")
                print("  /load <path>    Load agent memory file (.mem)")
                print("  /problem <path> Load problem file")
                print("  /prompt <text>  Add prompt for next run (or type without /)")
                print("  /export, /e      Generate PDF (solution + verification)")
                print("  /status, /s     Show state")
                print("  /list, /l       List memory files in log-dir")
                print("  /clear          Clear prompts")
                print("  /help, /h       This message")
                print("  /quit, /exit    Exit")
            elif cmd in ("run", "r"):
                if not problem_statement:
                    print("  Load a problem (/problem) or memory (/load) first.")
                    continue
                try:
                    do_run()
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
                print("  Prompts cleared.")
            else:
                print(f"  Unknown /{cmd}. Type /help.")
            continue

        # Bare input = additional prompt (Claude Code: natural language)
        if line:
            other_prompts.append(line)
            preview = line[:50] + "..." if len(line) > 50 else line
            print(f"  +prompt #{len(other_prompts)}: {preview}")

    agent_module.close_log_file()


if __name__ == "__main__":
    main()
