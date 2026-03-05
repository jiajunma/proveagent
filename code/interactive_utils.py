"""File/memory utilities and status bar for the interactive IMO agent."""

import os
from typing import Optional


def list_files_by_ext(directory: str, extensions: list) -> list:
    """List files in directory matching given extensions, sorted by mtime (newest first)."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        [f for f in os.listdir(directory)
         if os.path.isfile(os.path.join(directory, f)) and
         any(f.endswith(ext) for ext in extensions)],
        key=lambda x: os.path.getmtime(os.path.join(directory, x)),
        reverse=True,
    )


def pick_file(directory: str, extensions: list, label: str, max_show: int = 20) -> Optional[str]:
    """Show a numbered list of files and let the user pick one.

    Returns the absolute path of the selected file, or None if cancelled.
    """
    files = list_files_by_ext(directory, extensions)
    if not files:
        print(f"  No {label} files found in {directory}")
        return None

    print(f"  Available {label} files:")
    for i, f in enumerate(files[:max_show]):
        fpath = os.path.join(directory, f)
        size = os.path.getsize(fpath)
        size_str = f"{size}B" if size < 1024 else f"{size // 1024}KB"
        print(f"    [{i + 1}] {f}  ({size_str})")
    if len(files) > max_show:
        print(f"    ... and {len(files) - max_show} more")

    try:
        sel = input(f"  Select {label} (number, or Enter to cancel): ").strip()
        if not sel:
            print("  Cancelled.")
            return None
        if not sel.isdigit() or int(sel) < 1 or int(sel) > min(len(files), max_show):
            print("  Invalid selection.")
            return None
        return os.path.abspath(os.path.join(directory, files[int(sel) - 1]))
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.")
        return None


def list_memory_files(log_dir: str) -> list:
    """List .mem files in log_dir sorted by mtime (newest first)."""
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


def render_status_bar(
    problem_loaded: bool,
    memory_file: Optional[str],
    solution: bool,
    prompts: int,
    edit_mode: bool = False,
    original_problem: Optional[str] = None,
    streaming: bool = True,
    thinking: bool = True,
    interactive: bool = True,
    provider: str = "gemini",
    model: str = None,
    num_proof_comments: int = 0,
    num_verify_comments: int = 0,
    num_stash: int = 0,
    quota_exceeded: bool = False,
    quota_warning: bool = False,
) -> str:
    """Render a compact status line like Claude Code."""
    parts = []
    parts.append("problem \u2713" if problem_loaded else "problem \u2014")
    if memory_file:
        parts.append(f"mem:{os.path.basename(memory_file)}")
    parts.append("solution \u2713" if solution else "solution \u2014")

    if provider:
        provider_info = provider
        if model:
            provider_info += f":{model.split('-')[0]}"
        parts.append(provider_info)

    if edit_mode:
        parts.append(f"editing:{os.path.basename(original_problem)}" if original_problem else "edit")

    mode_parts = []
    if streaming:
        mode_parts.append("stream")
    if thinking:
        mode_parts.append("think")
    if interactive:
        mode_parts.append("interact")
    if mode_parts:
        parts.append("+".join(mode_parts))

    if quota_exceeded:
        parts.append("\u26a0\ufe0fquota:exceeded")
    elif quota_warning:
        parts.append("\u26a0\ufe0fquota:low")

    if prompts:
        parts.append(f"+{prompts} prompt(s)")
    total_comments = num_proof_comments + num_verify_comments
    if total_comments:
        parts.append(f"+{num_proof_comments}p/{num_verify_comments}v comment(s)")
    if num_stash:
        parts.append(f"\U0001f4f7{num_stash} stash")
    return " | ".join(parts)
