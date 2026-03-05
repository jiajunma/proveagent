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
        "/partial", "/paste", "/stash", "/st",
        "/comment", "/c", "/pcomment", "/vcomment",
        "/comments", "/del_comment", "/clear_comments",
        "/export", "/e", "/status", "/s", "/list", "/l", "/clear",
        "/analyze", "/edit", "/edit_problem", "/done", "/edit_existing", "/save_as",
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

# Optional: prompt_toolkit for Ctrl+V image paste
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.patch_stdout import patch_stdout as _pt_patch_stdout
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

# Import from agent module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent as agent_module
from interactive_prompts import (
    FAST_MODELS, TEX_PREAMBLE, TEX_COMPOSE_TEMPLATE, TEX_FIX_TEMPLATE,
    PARTIAL_SOLUTION_EXTRACT_TEMPLATE, PROBLEM_ANALYSIS_PROMPT,
    EDIT_PROBLEM_SYSTEM, IMAGE_OCR_PROMPT,
)
from fast_model_client import call_fast_model, call_fast_model_chat, call_flash, call_flash_chat
from latex_pipeline import (
    compose_tex_with_fast_model, fix_tex_with_fast_model,
    compose_tex_with_flash, fix_tex_with_flash,
    compile_latex, open_pdf, export_to_md, export_to_pdf,
    extract_partial_solution, analyze_problem,
)
from image_ocr import get_clipboard_image, ocr_image_to_latex
from interactive_utils import (
    list_files_by_ext, pick_file, list_memory_files,
    save_problem_to_file, render_status_bar,
)
from agent_worker import run_agent_worker


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

    # Add partial solution file loading option
    parser.add_argument("--partial", type=str,
                       help="Load a LaTeX file as partial solution (requires --problem or path)")


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
                       help="Select API provider (gemini, openai, kimi). Kimi provider uses kimi-k2-thinking by default for thinking capability.")
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
    proof_comments: list = []  # Comments for guiding proof improvement
    verify_comments: list = []  # Comments for guiding verification
    memory_file: Optional[str] = None
    base_name = "interactive"
    solution: Optional[str] = None
    full_verification = ""
    cached_tex: Optional[str] = None  # Cached LaTeX source for PDF generation
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
    # Image stash: OCR'd LaTeX from clipboard images (via Ctrl+V)
    image_stash: list = []  # [{"latex": str|None, "status": "ready"|"pending"|"failed"}]
    import threading as _threading
    _stash_lock = _threading.Lock()

    def load_from_memory(path: str) -> bool:
        nonlocal problem_statement, other_prompts, proof_comments, verify_comments, solution, full_verification, memory_file, base_name, cached_tex
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
        # Load comments: support both old format (single "comments" list) and new split format
        old_comments = mem.get("comments", [])
        proof_comments = mem.get("proof_comments", [])
        verify_comments = mem.get("verify_comments", [])
        if old_comments and not proof_comments and not verify_comments:
            # Migrate old single-list comments to proof_comments
            proof_comments = old_comments
        solution = mem.get("solution")
        full_verification = mem.get("full_verification", mem.get("verify", ""))
        cached_tex = mem.get("cached_tex")
        memory_file = path
        base_name = os.path.splitext(os.path.basename(path))[0]
        total_comments = len(proof_comments) + len(verify_comments)
        if total_comments:
            print(f"  Loaded {len(proof_comments)} proof comment(s), {len(verify_comments)} verify comment(s).")
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
    log_path = os.path.join(log_dir, f"{base_name}_interactive.prooflog")
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

    # Handle --partial flag: load LaTeX as partial solution at startup
    if args.partial:
        if not problem_statement:
            print("  Error: --partial requires a problem to be loaded first (use --problem or positional arg).")
            sys.exit(1)
        partial_path = os.path.abspath(args.partial)
        if not os.path.exists(partial_path):
            print(f"  Partial solution file not found: {partial_path}")
            sys.exit(1)
        try:
            with open(partial_path, "r", encoding="utf-8") as f:
                latex_content = f.read()
            if not latex_content.strip():
                print("  Partial solution file is empty.")
                sys.exit(1)
            print(f"  Extracting partial solution from {os.path.basename(partial_path)}...")
            partial_sol = extract_partial_solution(
                problem_statement, latex_content, api_key,
                provider=provider_name, model_name=model_name
            )
            if partial_sol and partial_sol.strip():
                solution = partial_sol
                if memory_file:
                    agent_module.save_memory(
                        memory_file, problem_statement, other_prompts,
                        0, 30, solution, "no", ""
                    )
                print(f"  Partial solution loaded ({len(partial_sol)} chars). Use /run to continue.")
                # Auto-generate PDF
                print("  Auto-generating PDF for partial solution...")
                try:
                    ok, final_tex = export_to_pdf(
                        problem_statement, solution, "",
                        log_dir, base_name, api_key,
                        provider=provider_name, model_name=model_name,
                    )
                    if ok and final_tex:
                        cached_tex = final_tex
                        # Save tex to mem
                        if memory_file and os.path.exists(memory_file):
                            with open(memory_file, "r", encoding="utf-8") as f:
                                mem_data = json.load(f)
                            mem_data["cached_tex"] = final_tex
                            with open(memory_file, "w", encoding="utf-8") as f:
                                json.dump(mem_data, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"  PDF generation failed: {e}")
            else:
                print("  Warning: Failed to extract partial solution. Starting without it.")
        except Exception as e:
            print(f"  Error loading partial solution: {e}")
            sys.exit(1)

    def do_analyze_problem():
        """Analyze the current problem statement for quality and auto-fix if possible."""
        nonlocal problem_statement, cached_tex, in_edit_mode
        if not problem_statement:
            print("  No problem loaded.")
            return
        print(f"  Analyzing problem statement ({len(problem_statement)} chars)...")
        result = analyze_problem(problem_statement, api_key, provider_name, model_name)
        verdict = result["verdict"]

        if verdict == "ERROR":
            print(f"  Analysis failed: {result['analysis']}")
            return

        if verdict == "PASS":
            print("  Problem analysis: PASS — problem is well-defined and ready for solving.")
            # Generate problem-only PDF
            print("  Generating problem PDF...")
            try:
                ok, final_tex = export_to_pdf(
                    problem_statement, "(Problem statement only — no solution yet.)", "",
                    log_dir, base_name, api_key,
                    provider=provider_name, model_name=model_name,
                )
                if ok and final_tex:
                    cached_tex = None  # Don't cache problem-only PDF
            except Exception as e:
                print(f"  PDF generation failed: {e}")
            return

        # Show the analysis
        print(f"\n  Problem analysis: {verdict}")
        print("  " + "-" * 50)
        # Print issues section from the analysis
        for line in result["analysis"].split("\n"):
            print(f"  {line}")
        print("  " + "-" * 50)

        if verdict == "FIXABLE" and result["fixed_problem"]:
            print("\n  Auto-fixed problem:")
            print("  " + "-" * 50)
            for line in result["fixed_problem"].split("\n"):
                print(f"    {line}")
            print("  " + "-" * 50)

            # Generate PDF with fixed problem for review
            print("  Generating PDF with fixed problem for review...")
            try:
                ok, final_tex = export_to_pdf(
                    result["fixed_problem"],
                    "(Problem statement only — auto-fixed, pending approval.)", "",
                    log_dir, base_name, api_key,
                    provider=provider_name, model_name=model_name,
                )
            except Exception as e:
                print(f"  PDF generation failed: {e}")

            # Ask user to accept or reject
            try:
                choice = input("  Accept fixed problem? (y/n/edit): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Keeping original problem.")
                return
            if choice in ("y", "yes"):
                problem_statement = result["fixed_problem"]
                cached_tex = None
                print("  Problem updated with fixes.")
            elif choice in ("e", "edit"):
                # Enter edit mode with the fixed problem
                problem_statement = result["fixed_problem"]
                cached_tex = None
                print("  Entering edit mode with fixed problem. Type /done when finished.")
                in_edit_mode = True  # noqa - captured by closure
            else:
                print("  Keeping original problem.")

        elif verdict == "NEEDS_INPUT":
            print("\n  The problem requires clarification before it can be used.")
            print("  Use /edit to interactively fix the problem with the agent.")

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
                print(f"  Problem set ({len(problem_statement)} chars). Use /run to solve.")
                # Prompt to save to .txt file
                try:
                    default_name = (os.path.splitext(os.path.basename(original_problem_path))[0]
                                    if original_problem_path else base_name or "problem")
                    user_fname = input(
                        f"  Save to file? Enter filename (default: {default_name}.txt, Enter to skip): "
                    ).strip()
                    if user_fname.lower() not in ("", "n", "no", "skip"):
                        fname = user_fname if user_fname else default_name
                        if not os.path.splitext(fname)[1]:
                            fname += ".txt"
                        problems_dir = os.path.join(os.path.dirname(script_dir), "problems")
                        os.makedirs(problems_dir, exist_ok=True)
                        save_path = os.path.join(problems_dir, fname)
                        if os.path.exists(save_path):
                            ow = input(f"  '{fname}' already exists. Overwrite? (y/N): ").strip().lower()
                            if ow != "y":
                                print("  Save cancelled.")
                            elif save_problem_to_file(problem_statement, save_path):
                                print(f"  ✓ Saved to {save_path}")
                                original_problem_path = save_path
                                base_name = os.path.splitext(fname)[0]
                                memory_file = os.path.join(log_dir, f"{base_name}.mem")
                        elif save_problem_to_file(problem_statement, save_path):
                            print(f"  ✓ Saved to {save_path}")
                            original_problem_path = save_path
                            base_name = os.path.splitext(fname)[0]
                            memory_file = os.path.join(log_dir, f"{base_name}.mem")
                except (EOFError, KeyboardInterrupt):
                    print("\n  Save skipped.")
                return True
            in_edit_mode = False
            return True
        # Normal turn: show reply, update history
        edit_history.append({"role": "user", "parts": [{"text": user_msg}]})
        edit_history.append({"role": "model", "parts": [{"text": reply}]})
        print(f"  Agent: {reply}")
        return False

    def save_comments_to_mem():
        """Save proof and verification comments to the memory file."""
        if not memory_file or not os.path.exists(memory_file):
            return
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                mem = json.load(f)
            mem["proof_comments"] = proof_comments
            mem["verify_comments"] = verify_comments
            # Remove old single-list key if present
            mem.pop("comments", None)
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(mem, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  Warning: could not save comments to mem: {e}")

    def save_tex_to_mem(tex_content: str):
        """Save cached LaTeX content to the memory file."""
        nonlocal cached_tex
        cached_tex = tex_content
        if not memory_file or not os.path.exists(memory_file):
            return
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                mem = json.load(f)
            mem["cached_tex"] = tex_content
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(mem, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  Warning: could not save tex to mem: {e}")

    def do_export_pdf(use_cached: bool = True, prov: str = None, mdl: str = None):
        """Export PDF, using cached tex if available. Updates cached_tex on success."""
        nonlocal cached_tex
        if not solution:
            print("  No solution yet. Run /run first.")
            return
        tex_to_use = cached_tex if use_cached else None
        ok, final_tex = export_to_pdf(
            problem_statement, solution, full_verification,
            log_dir, base_name, api_key,
            provider=prov or provider_name,
            model_name=mdl or model_name,
            cached_tex=tex_to_use,
        )
        if ok and final_tex:
            save_tex_to_mem(final_tex)
        elif not ok and tex_to_use:
            # Cached tex failed to compile, retry without cache
            print("  Cached LaTeX failed, regenerating...")
            cached_tex = None
            ok, final_tex = export_to_pdf(
                problem_statement, solution, full_verification,
                log_dir, base_name, api_key,
                provider=prov or provider_name,
                model_name=mdl or model_name,
            )
            if ok and final_tex:
                save_tex_to_mem(final_tex)

    def do_export(fmt: str = "pdf"):
        if not solution:
            print("  No solution yet. Run /run first.")
            return
        if fmt.lower() == "md" or fmt.lower() == "markdown":
            md_path = export_to_md(
                problem_statement, solution, full_verification,
                log_dir, base_name,
            )
            print(f"  ✓ Markdown exported: {md_path}")
        else:
            # /export always regenerates (no cache), since user explicitly requested
            do_export_pdf(use_cached=False)

    def do_run(streaming=True, show_thinking=True, interactive=True,
               use_provider=None, use_model=None):
        nonlocal solution, full_verification, provider_name, model_name, cached_tex

        # 如果提供了特定的provider和model，则使用它们
        current_provider = use_provider if use_provider is not None else provider_name
        current_model = use_model if use_model is not None else model_name

        # Merge other_prompts and proof comments for the agent
        # Each comment is sent as a separate prompt item for individual attention
        all_prompts = list(other_prompts)
        for i, c in enumerate(proof_comments):
            all_prompts.append(f"[Proof Comment {i + 1}]: {c}")

        # Build verification prompts from verify comments
        all_verify_prompts = []
        for i, c in enumerate(verify_comments):
            all_verify_prompts.append(f"[Verification Comment {i + 1}]: {c}")

        # Run agent in subprocess so Ctrl+C can terminate in-flight API calls
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=run_agent_worker,
            args=(
                problem_statement,
                all_prompts,
                all_verify_prompts,
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
                cached_tex = mem.get("cached_tex")  # subprocess may have saved tex
        # Auto-generate PDF with final solution + verification
        if problem_statement and solution:
            print("  Auto-generating PDF with final results...")
            # Invalidate cached tex since solution/verification likely changed
            cached_tex = None
            try:
                do_export_pdf(use_cached=False, prov=current_provider, mdl=current_model)
            except Exception as e:
                print(f"  PDF generation failed: {e}")

    # Auto-generate PDF if we loaded a problem with an existing solution
    if problem_statement and solution:
        print("  Auto-generating PDF for loaded solution...")
        try:
            do_export_pdf(use_cached=True)
        except Exception as e:
            print(f"  PDF generation failed: {e}")
    # (auto-analyze on load removed — use /analyze manually)

    # ── Prompt-toolkit session with Ctrl+V image paste ──────────────────────────
    _pt_session = None

    if HAS_PROMPT_TOOLKIT:
        _pt_kb = KeyBindings()

        @_pt_kb.add('c-v')
        def _ctrl_v_handler(event):
            """Ctrl+V: capture image from clipboard and queue for OCR."""
            img_path = get_clipboard_image()
            if not img_path:
                return  # No image in clipboard; ignore

            size_kb = max(1, os.path.getsize(img_path) // 1024)
            with _stash_lock:
                stash_item = {"status": "pending", "latex": None, "size_kb": size_kb}
                image_stash.append(stash_item)
                idx = len(image_stash)

            # Non-blocking: OCR in background thread
            def _do_ocr():
                try:
                    latex = ocr_image_to_latex(img_path, api_key, provider_name, model_name)
                    with _stash_lock:
                        if latex and latex.strip():
                            stash_item["latex"] = latex.strip()
                            stash_item["status"] = "ready"
                            preview = stash_item["latex"][:70]
                            ellipsis = "..." if len(stash_item["latex"]) > 70 else ""
                            print(f"\n  \u2713 Stash #{idx} ready ({len(stash_item['latex'])} chars): {preview}{ellipsis}")
                        else:
                            stash_item["status"] = "failed"
                            print(f"\n  \u2717 Stash #{idx}: no content recognized")
                except Exception as e:
                    with _stash_lock:
                        stash_item["status"] = "failed"
                    print(f"\n  \u2717 Stash #{idx} OCR failed: {e}")
                finally:
                    try:
                        os.unlink(img_path)
                    except Exception:
                        pass
                event.app.invalidate()

            _threading.Thread(target=_do_ocr, daemon=True).start()
            print(f"\n  \U0001f4f7 Stash #{idx}: captured {size_kb}KB, OCR-ing in background...")
            event.app.invalidate()

        _pt_commands = [
            "/run", "/r", "/load", "/problem", "/prompt", "/add", "/p",
            "/partial", "/paste", "/stash", "/st",
            "/comment", "/c", "/pcomment", "/vcomment",
            "/comments", "/del_comment", "/clear_comments",
            "/export", "/e", "/status", "/s", "/list", "/l", "/clear",
            "/analyze", "/edit", "/edit_problem", "/done", "/edit_existing", "/save_as",
            "/streaming", "/thinking", "/interactive", "/run_mode", "/quota",
            "/provider", "/model", "/providers",
            "/help", "/h", "/quit", "/q", "/exit",
        ]
        _pt_completer = WordCompleter(_pt_commands, sentence=True)
        _pt_session = PromptSession(
            key_bindings=_pt_kb,
            history=InMemoryHistory(),
            completer=_pt_completer,
        )

    def _get_input(prompt_str: str) -> str:
        if _pt_session:
            with _pt_patch_stdout():
                return _pt_session.prompt(prompt_str)
        return input(prompt_str)

    # Banner
    from datetime import datetime
    print()
    print("  IMO Interactive Agent v1.0")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
            provider_name, model_name,
            num_proof_comments=len(proof_comments),
            num_verify_comments=len(verify_comments),
            num_stash=len(image_stash),
            quota_exceeded=getattr(agent_module, 'TOKEN_QUOTA_EXCEEDED', False),
            quota_warning=getattr(agent_module, 'TOKEN_QUOTA_WARNING', False),
        )
        try:
            line = _get_input(f"  [{status}]\n  > ").strip()
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
                print("  /load [path]      Load memory file (.mem); no arg = pick from list")
                print("  /problem [path]   Load problem file; no arg = pick from list")
                print("  /partial [path]   Load LaTeX as partial solution; no arg = pick from list")
                print("  /paste            OCR image from clipboard → LaTeX (proof/verify/partial)")
                print("  /stash, /st       Manage image stash (Ctrl+V to add images)")
                print("  /analyze          Re-analyze current problem for issues")
                print("  /edit             Draft or refine problem with agent (no problem = direct chat)")
                print("  /edit_existing    Browse and edit an existing problem file")
                print("  /done             Save current draft as problem (in edit mode)")
                print("  /save_as <name>   Save edited problem to a new file")
                print("  /prompt <text>    Add prompt for next run")
                print("  /comment <text>   Add proof comment (bare text also works); no arg = multi-line")
                print("  /pcomment <text>  Add proof comment (alias for /comment)")
                print("  /vcomment <text>  Add verification comment; no arg = multi-line")
                print("  /comments         List & manage comments (edit/delete/view)")
                print("  /del_comment p|v <n>  Delete comment by type and number")
                print("  /clear_comments   Clear all comments")
                print("  /export, /e       Generate PDF (solution + verification)")
                print("  /export md        Generate Markdown file (fallback when PDF fails)")
                print("  /status, /s       Show state")
                print("  /list, /l         List memory files in log-dir")
                print("  /clear            Clear prompts and comments")
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
                    # Check if format is specified (e.g., /export md or /export pdf)
                    fmt = rest.strip().lower() if rest else "pdf"
                    do_export(fmt)
                except KeyboardInterrupt:
                    print("\n  Interrupted. Back to prompt.")
            elif cmd == "load":
                mem_path = None
                if rest:
                    mem_path = os.path.abspath(rest.strip())
                else:
                    mem_path = pick_file(log_dir, [".mem"], "memory")
                if mem_path:
                    load_from_memory(mem_path)
                    # Auto-generate PDF if solution exists after loading
                    if problem_statement and solution:
                        print("  Auto-generating PDF...")
                        try:
                            do_export_pdf(use_cached=True)
                        except Exception as e:
                            print(f"  PDF generation failed: {e}")
            elif cmd == "problem":
                prob_path = None
                if rest:
                    prob_path = os.path.abspath(rest.strip())
                else:
                    problems_dir = os.path.join(os.path.dirname(script_dir), "problems")
                    prob_path = pick_file(problems_dir, [".txt", ".md"], "problem")
                if prob_path:
                    load_from_problem(prob_path)
                    if problem_statement and solution:
                        # Has existing solution — auto-generate PDF
                        print("  Auto-generating PDF...")
                        try:
                            do_export_pdf(use_cached=True)
                        except Exception as e:
                            print(f"  PDF generation failed: {e}")
                    elif problem_statement and not solution:
                        print("  Problem loaded. Use /analyze to check it, or /run to solve.")
            elif cmd == "partial":
                # Load a LaTeX file as partial solution
                if not problem_statement:
                    print("  Load a problem first (/problem <path>), then use /partial.")
                    continue
                latex_path = None
                if rest:
                    latex_path = rest.strip()
                    if not os.path.isabs(latex_path) and not os.path.exists(latex_path):
                        # Try log_dir and common locations
                        for candidate_dir in [log_dir, os.path.join(script_dir, "..", "run_logs"), "."]:
                            candidate = os.path.join(candidate_dir, latex_path)
                            if os.path.exists(candidate):
                                latex_path = candidate
                                break
                    latex_path = os.path.abspath(latex_path)
                    if not os.path.exists(latex_path):
                        print(f"  File not found: {latex_path}")
                        continue
                else:
                    latex_path = pick_file(log_dir, [".tex", ".md", ".txt"], "partial solution")
                    if not latex_path:
                        continue
                try:
                    with open(latex_path, "r", encoding="utf-8") as f:
                        latex_content = f.read()
                except Exception as e:
                    print(f"  Error reading file: {e}")
                    continue
                if not latex_content.strip():
                    print("  File is empty.")
                    continue
                print(f"  Loaded {len(latex_content)} chars from {os.path.basename(latex_path)}")
                print(f"  Extracting partial solution using {provider_name}...")
                try:
                    partial_sol = extract_partial_solution(
                        problem_statement, latex_content, api_key,
                        provider=provider_name, model_name=model_name
                    )
                    if partial_sol and partial_sol.strip():
                        solution = partial_sol
                        # Save to memory so /run will pick it up as resume
                        if memory_file:
                            agent_module.save_memory(
                                memory_file,
                                problem_statement,
                                other_prompts,
                                0,  # iteration
                                30,  # max_iterations
                                solution,
                                "no",  # not verified yet
                                ""  # no verification yet
                            )
                        print(f"  Partial solution extracted ({len(partial_sol)} chars).")
                        # Show a preview
                        preview_lines = partial_sol.strip().split("\n")[:8]
                        for pl in preview_lines:
                            print(f"    {pl}")
                        if len(partial_sol.strip().split("\n")) > 8:
                            print("    ...")
                        # Auto-generate PDF (no cache since solution is new)
                        print("  Auto-generating PDF for partial solution...")
                        try:
                            do_export_pdf(use_cached=False)
                        except Exception as e:
                            print(f"  PDF generation failed: {e}")
                        print("  Use /run to continue proving from this starting point.")
                    else:
                        print("  Failed to extract partial solution (empty result).")
                except Exception as e:
                    print(f"  Error extracting partial solution: {e}")
            elif cmd == "paste":
                print("  Reading image from clipboard...")
                img_path = get_clipboard_image()
                if not img_path:
                    sys_name = platform.system()
                    print("  No image found in clipboard.")
                    if sys_name == "Darwin":
                        print("  Copy a screenshot first (Cmd+Ctrl+Shift+4 → area, or Cmd+Ctrl+Shift+3).")
                        print("  If this keeps failing: brew install pngpaste")
                    elif sys_name == "Linux":
                        print("  Requires xclip: sudo apt install xclip")
                    continue
                size_kb = max(1, os.path.getsize(img_path) // 1024)
                print(f"  Image captured ({size_kb} KB). Recognizing with {provider_name}...")
                try:
                    latex_text = ocr_image_to_latex(img_path, api_key, provider_name, model_name)
                except Exception as e:
                    print(f"  OCR failed: {e}")
                    latex_text = None
                finally:
                    try:
                        os.unlink(img_path)
                    except Exception:
                        pass
                if not latex_text or not latex_text.strip():
                    print("  No content recognized.")
                    continue
                # Show extracted content
                lines_ocr = latex_text.split("\n")
                print(f"\n  Extracted ({len(lines_ocr)} lines, {len(latex_text)} chars):")
                print("  " + "\u2500" * 60)
                for ocr_line in lines_ocr[:30]:
                    print(f"  {ocr_line}")
                if len(lines_ocr) > 30:
                    print(f"  ... ({len(lines_ocr) - 30} more lines)")
                print("  " + "\u2500" * 60)
                print("  Store as: (p) proof comment  (v) verify comment  (s) partial solution  (n) discard")
                try:
                    choice = input("  > ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\n  Discarded.")
                    continue
                if choice in ("p", "proof"):
                    proof_comments.append(latex_text)
                    save_comments_to_mem()
                    print(f"  +proof comment #{len(proof_comments)}")
                elif choice in ("v", "verify"):
                    verify_comments.append(latex_text)
                    save_comments_to_mem()
                    print(f"  +verify comment #{len(verify_comments)}")
                elif choice in ("s", "partial", "solution"):
                    solution = latex_text
                    if memory_file:
                        agent_module.save_memory(
                            memory_file, problem_statement, other_prompts,
                            0, 30, solution, "no", ""
                        )
                    print(f"  Partial solution set ({len(latex_text)} chars).")
                    try:
                        do_export_pdf(use_cached=False)
                    except Exception as e:
                        print(f"  PDF generation failed: {e}")
                else:
                    print("  Discarded.")
            elif cmd in ("stash", "st"):
                # Interactive stash manager
                with _stash_lock:
                    stash_copy = list(image_stash)
                if not stash_copy:
                    print("  No stash items. Press Ctrl+V to paste an image from clipboard.")
                    continue

                def _print_stash(items):
                    for i, item in enumerate(items):
                        status_icon = {"ready": "\u2713", "pending": "\u23f3", "failed": "\u2717"}.get(item["status"], "?")
                        size_str = f" {item['size_kb']}KB" if "size_kb" in item else ""
                        if item["status"] == "ready" and item["latex"]:
                            preview = item["latex"][:60].replace("\n", " ")
                            ellipsis = "..." if len(item["latex"]) > 60 else ""
                            print(f"    [{i+1}] {status_icon}{size_str} {preview}{ellipsis}")
                        else:
                            print(f"    [{i+1}] {status_icon}{size_str} ({item['status']})")

                print(f"  Image stash ({len(stash_copy)} items):")
                _print_stash(stash_copy)
                print("  \u2500\u2500\u2500\u2500\u2500")
                print("  Actions: p <n> = proof comment, vc <n> = verify comment, s <n> = partial solution")
                print("           d <n> = delete, show <n> = view full text, l = list, q = done")
                while True:
                    try:
                        act_line = input("  stash> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break
                    if not act_line or act_line.lower() in ("q", "quit", "done"):
                        break
                    parts_st = act_line.split()
                    if not parts_st:
                        continue
                    act_st = parts_st[0].lower()
                    if act_st in ("l", "list"):
                        with _stash_lock:
                            stash_copy = list(image_stash)
                        _print_stash(stash_copy)
                        continue
                    if len(parts_st) < 2 or not parts_st[1].isdigit():
                        print("  Usage: p|v|s|d <n>  or  l = list, q = done")
                        continue
                    si = int(parts_st[1]) - 1
                    with _stash_lock:
                        if si < 0 or si >= len(image_stash):
                            print(f"  Invalid stash number. Have {len(image_stash)} item(s).")
                            continue
                        item = image_stash[si]
                    if item["status"] == "pending":
                        print(f"  Stash #{si+1} is still being processed. Try again in a moment.")
                        continue
                    if item["status"] == "failed":
                        print(f"  Stash #{si+1} OCR failed. Cannot apply.")
                        continue
                    latex = item["latex"]
                    if act_st in ("show", "view"):
                        print(f"  [Stash #{si+1}]")
                        for stash_line in latex.split("\n"):
                            print(f"  {stash_line}")
                    elif act_st == "p":
                        proof_comments.append(latex)
                        save_comments_to_mem()
                        print(f"  +proof comment #{len(proof_comments)} from stash #{si+1}")
                    elif act_st == "vc":
                        verify_comments.append(latex)
                        save_comments_to_mem()
                        print(f"  +verify comment #{len(verify_comments)} from stash #{si+1}")
                    elif act_st == "s":
                        solution = latex
                        if memory_file:
                            agent_module.save_memory(
                                memory_file, problem_statement, other_prompts,
                                0, 30, solution, "no", ""
                            )
                        print(f"  Partial solution set from stash #{si+1} ({len(latex)} chars).")
                        try:
                            do_export_pdf(use_cached=False)
                        except Exception as e:
                            print(f"  PDF generation failed: {e}")
                    elif act_st == "d":
                        with _stash_lock:
                            if si < len(image_stash):
                                image_stash.pop(si)
                        print(f"  Stash #{si+1} deleted.")
                        with _stash_lock:
                            stash_copy = list(image_stash)
                        _print_stash(stash_copy)
                    else:
                        print("  Unknown action. Use p / vc / s / d / show / l / q.")
            elif cmd in ("prompt", "add", "p"):
                if rest:
                    other_prompts.append(rest)
                    print(f"  +prompt #{len(other_prompts)}")
                else:
                    print("  Usage: /prompt <instruction>")
            elif cmd in ("comment", "c", "pcomment", "vcomment"):
                # Determine target: proof (default) or verify
                if cmd == "vcomment":
                    target_list, label = verify_comments, "verify"
                else:
                    target_list, label = proof_comments, "proof"
                if rest:
                    target_list.append(rest)
                    save_comments_to_mem()
                    preview = rest[:60] + "..." if len(rest) > 60 else rest
                    print(f"  +{label} comment #{len(target_list)}: {preview}")
                else:
                    print(f"  Enter {label} comment (Ctrl-D to finish):")
                    lines = []
                    try:
                        while True:
                            l = input("  . ")
                            lines.append(l)
                    except EOFError:
                        pass
                    except KeyboardInterrupt:
                        print("\n  Cancelled.")
                        continue
                    if lines:
                        comment_text = "\n".join(lines)
                        target_list.append(comment_text)
                        save_comments_to_mem()
                        preview = lines[0][:50] + ("..." if len(lines[0]) > 50 or len(lines) > 1 else "")
                        print(f"\n  +{label} comment #{len(target_list)} ({len(lines)} lines): {preview}")
                    else:
                        print("  Empty comment, not added.")
            elif cmd in ("comments", "del_comment", "clear_comments"):
                # Unified interactive comment management
                def _print_comments():
                    if not proof_comments and not verify_comments:
                        print("  No comments.")
                        return False
                    if proof_comments:
                        print(f"  Proof comments ({len(proof_comments)}):")
                        for i, c in enumerate(proof_comments):
                            lines_c = c.split("\n")
                            first = lines_c[0][:70]
                            suffix = "..." if len(lines_c) > 1 or len(lines_c[0]) > 70 else ""
                            print(f"    [p{i + 1}] {first}{suffix}")
                    if verify_comments:
                        print(f"  Verification comments ({len(verify_comments)}):")
                        for i, c in enumerate(verify_comments):
                            lines_c = c.split("\n")
                            first = lines_c[0][:70]
                            suffix = "..." if len(lines_c) > 1 or len(lines_c[0]) > 70 else ""
                            print(f"    [v{i + 1}] {first}{suffix}")
                    return True

                # Handle /del_comment p|v <n> directly
                if cmd == "del_comment" and rest.strip():
                    parts_dc = rest.strip().split()
                    if len(parts_dc) == 2 and parts_dc[0] in ("p", "v") and parts_dc[1].isdigit():
                        ctype, cnum = parts_dc[0], int(parts_dc[1]) - 1
                        target = proof_comments if ctype == "p" else verify_comments
                        label = "proof" if ctype == "p" else "verify"
                        if 0 <= cnum < len(target):
                            removed = target.pop(cnum)
                            save_comments_to_mem()
                            preview = removed[:50] + "..." if len(removed) > 50 else removed
                            print(f"  Removed {label} comment #{cnum + 1}: {preview}")
                        else:
                            print(f"  Invalid {label} comment number. Have {len(target)} {label} comment(s).")
                        continue
                    else:
                        print("  Usage: /del_comment p|v <number>")
                        continue

                # Handle /clear_comments directly
                if cmd == "clear_comments":
                    proof_comments.clear()
                    verify_comments.clear()
                    save_comments_to_mem()
                    print("  All comments cleared.")
                    continue

                # /comments — interactive listing with edit/delete
                has = _print_comments()
                if not has:
                    print("  Use /comment (proof) or /vcomment (verify) to add.")
                    continue
                print("  ─────")
                print("  Actions: d p|v <n> = delete, e p|v <n> = edit, v p|v <n> = view full, q = done")
                while True:
                    try:
                        action = input("  comments> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break
                    if not action or action.lower() in ("q", "quit", "done"):
                        break
                    parts_a = action.split(None, 2)
                    if len(parts_a) < 3 and parts_a[0].lower() not in ("q", "quit", "done", "l", "list"):
                        if len(parts_a) >= 1 and parts_a[0].lower() in ("l", "list"):
                            _print_comments()
                            continue
                        print("  Usage: d|e|v p|v <n>  or  l = list, q = done")
                        continue
                    act = parts_a[0].lower()
                    ctype = parts_a[1].lower() if len(parts_a) > 1 else ""
                    cnum_s = parts_a[2] if len(parts_a) > 2 else ""
                    if ctype not in ("p", "v") or not cnum_s.isdigit():
                        print("  Usage: d|e|v p|v <n>")
                        continue
                    cnum = int(cnum_s) - 1
                    target = proof_comments if ctype == "p" else verify_comments
                    label = "proof" if ctype == "p" else "verify"
                    if cnum < 0 or cnum >= len(target):
                        print(f"  Invalid {label} comment number. Have {len(target)} {label} comment(s).")
                        continue
                    if act == "d":
                        removed = target.pop(cnum)
                        save_comments_to_mem()
                        preview = removed[:50] + "..." if len(removed) > 50 else removed
                        print(f"  Deleted {label} comment #{cnum + 1}: {preview}")
                        _print_comments()
                    elif act == "v":
                        print(f"  [{label} comment #{cnum + 1}]")
                        print(f"  {target[cnum]}")
                    elif act == "e":
                        old = target[cnum]
                        old_lines = old.split("\n")
                        print(f"  Editing {label} comment #{cnum + 1} ({len(old_lines)} line(s)):")
                        for ol in old_lines:
                            print(f"  | {ol}")
                        print(f"  Enter new text (Ctrl-D to finish, empty = cancel):")
                        new_lines = []
                        try:
                            while True:
                                l = input("  . ")
                                new_lines.append(l)
                        except EOFError:
                            pass
                        except KeyboardInterrupt:
                            print("\n  Edit cancelled.")
                            continue
                        if new_lines:
                            new_text = "\n".join(new_lines)
                            target[cnum] = new_text
                            save_comments_to_mem()
                            preview = new_lines[0][:50] + ("..." if len(new_lines[0]) > 50 or len(new_lines) > 1 else "")
                            print(f"\n  Updated {label} comment #{cnum + 1}: {preview}")
                        else:
                            print("  Edit cancelled (empty input).")
                    else:
                        print("  Unknown action. Use d = delete, e = edit, v = view, l = list, q = done")
            elif cmd in ("status", "s"):
                print(f"  Problem: {len(problem_statement)} chars")
                print(f"  Memory: {memory_file or '—'}")
                print(f"  Prompts: {len(other_prompts)}")
                print(f"  Proof comments: {len(proof_comments)}")
                print(f"  Verify comments: {len(verify_comments)}")
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
                proof_comments.clear()
                verify_comments.clear()
                in_edit_mode = False
                edit_history.clear()
                original_problem_path = None
                save_comments_to_mem()
                print("  Prompts and comments cleared.")
            elif cmd == "analyze":
                if not problem_statement:
                    print("  No problem loaded.")
                else:
                    try:
                        do_analyze_problem()
                    except KeyboardInterrupt:
                        print("\n  Analysis interrupted.")
            elif cmd in ("edit", "edit_problem"):
                # Multi-line direct input: user types/pastes the problem, Ctrl-D to finish
                if problem_statement:
                    print("  Current problem (shown for reference):")
                    for _l in problem_statement.split("\n")[:6]:
                        print(f"  | {_l}")
                    if problem_statement.count("\n") >= 6:
                        print("  | ...")
                    print()
                print("  Enter problem statement (Ctrl-D to finish):")
                edit_lines = []
                try:
                    while True:
                        edit_lines.append(input("  | "))
                except EOFError:
                    pass
                except KeyboardInterrupt:
                    print("\n  Cancelled.")
                    continue
                if edit_lines:
                    new_problem = "\n".join(edit_lines).strip()
                    if new_problem:
                        problem_statement = new_problem
                        cached_tex = None
                        edit_history = []
                        in_edit_mode = False
                        print(f"  Problem set ({len(problem_statement)} chars).")
                        # Prompt to save to file
                        try:
                            default_name = (os.path.splitext(os.path.basename(original_problem_path))[0]
                                            if original_problem_path else base_name or "problem")
                            user_fname = input(
                                f"  Save to file? Enter filename (default: {default_name}.txt, Enter to skip): "
                            ).strip()
                            if user_fname.lower() not in ("", "n", "no", "skip"):
                                fname = user_fname if user_fname else default_name
                                if not os.path.splitext(fname)[1]:
                                    fname += ".txt"
                                problems_dir = os.path.join(os.path.dirname(script_dir), "problems")
                                os.makedirs(problems_dir, exist_ok=True)
                                save_path = os.path.join(problems_dir, fname)
                                if os.path.exists(save_path):
                                    ow = input(f"  '{fname}' already exists. Overwrite? (y/N): ").strip().lower()
                                    if ow != "y":
                                        print("  Save cancelled.")
                                    elif save_problem_to_file(problem_statement, save_path):
                                        print(f"  ✓ Saved to {save_path}")
                                        original_problem_path = save_path
                                        base_name = os.path.splitext(fname)[0]
                                        memory_file = os.path.join(log_dir, f"{base_name}.mem")
                                elif save_problem_to_file(problem_statement, save_path):
                                    print(f"  ✓ Saved to {save_path}")
                                    original_problem_path = save_path
                                    base_name = os.path.splitext(fname)[0]
                                    memory_file = os.path.join(log_dir, f"{base_name}.mem")
                        except (EOFError, KeyboardInterrupt):
                            print("\n  Save skipped.")
                    else:
                        print("  Empty input, problem unchanged.")
                else:
                    print("  No input, problem unchanged.")
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
                    filename += ".txt"  # Default to plain text

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
                        print("    - kimi-k2-thinking (default, with thinking)")
                        print("    - kimi-k2-thinking-turbo (with thinking)")
                        print("    - kimi-k2.5 (high quality, temp=1 only)")
                        print("    - kimi-k2-turbo-preview")
                        print("    - kimi-latest")
                        print("    - moonshot-v1-128k")
                        print("    - moonshot-v1-32k")
                        print("    - moonshot-v1-8k")
                        print("  Note: Use kimi-k2-thinking series for thinking capability")
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

        # Bare input: no problem -> chat to draft; in_edit_mode -> chat; else -> add comment
        if line:
            if not problem_statement or in_edit_mode:
                try:
                    do_edit_problem(line)
                except KeyboardInterrupt:
                    print("\n  Interrupted.")
            else:
                proof_comments.append(line)
                save_comments_to_mem()
                preview = line[:60] + "..." if len(line) > 60 else line
                print(f"  +proof comment #{len(proof_comments)}: {preview}")

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
