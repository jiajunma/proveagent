"""LaTeX composition, compilation, PDF export, and problem analysis."""

import os
import platform
import subprocess
import tempfile
from datetime import datetime
from typing import Optional, Tuple

import requests

from interactive_prompts import (
    FAST_MODELS, TEX_PREAMBLE, TEX_COMPOSE_TEMPLATE, TEX_FIX_TEMPLATE,
    PARTIAL_SOLUTION_EXTRACT_TEMPLATE, PROBLEM_ANALYSIS_PROMPT,
)
from fast_model_client import call_fast_model

try:
    from res2md import memory_to_tex as _res2md_memory_to_tex, markdown_to_latex as _markdown_to_latex
    HAS_RES2MD = True
except ImportError:
    HAS_RES2MD = False


# ── Internal helpers ──────────────────────────────────────────────────────────

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


# ── LaTeX composition / fixing ────────────────────────────────────────────────

def compose_tex_with_fast_model(problem: str, solution: str, verification: str,
                                 api_key: str, provider: str = "gemini",
                                 model_name: str = None) -> str:
    """Compose a LaTeX document from problem/solution/verification."""
    prompt = TEX_COMPOSE_TEMPLATE.replace("<<<PROBLEM>>>", problem)
    prompt = prompt.replace("<<<SOLUTION>>>", solution or "(No solution yet)")
    prompt = prompt.replace("<<<VERIFICATION>>>", verification or "(No verification yet)")

    try:
        if provider.lower() == "kimi":
            out = call_fast_model(prompt, api_key, provider, model_name,
                                  enable_thinking=False, timeout=600)
        else:
            out = call_fast_model(prompt, api_key, provider, model_name)
    except Exception as e:
        raise RuntimeError(f"Fast model failed for {provider}: {e}") from e

    latex = _extract_latex(out)
    if "\\documentclass" in latex:
        return latex
    if "\\begin{document}" in latex and "\\end{document}" in latex:
        return TEX_PREAMBLE + "\n" + latex
    return TEX_PREAMBLE + "\n\\begin{document}\n\n" + latex + "\n\n\\end{document}\n"


def fix_tex_with_fast_model(latex: str, error: str, api_key: str,
                            provider: str = "gemini", model_name: str = None,
                            timeout: int = 300) -> str:
    """Fix LaTeX compilation errors using the fast model."""
    prompt = TEX_FIX_TEMPLATE.replace("<<<ERROR>>>", error).replace("<<<LATEX>>>", latex)
    if provider.lower() == "kimi":
        return _extract_latex(call_fast_model(prompt, api_key, provider, model_name,
                                              enable_thinking=False, timeout=600))
    return _extract_latex(call_fast_model(prompt, api_key, provider, model_name, timeout=timeout))


# Legacy aliases
def compose_tex_with_flash(problem: str, solution: str, verification: str,
                            api_key: str, timeout: int = 300) -> str:
    return compose_tex_with_fast_model(problem, solution, verification, api_key, "gemini", None)


def fix_tex_with_flash(latex: str, error: str, api_key: str, timeout: int = 300) -> str:
    return fix_tex_with_fast_model(latex, error, api_key, "gemini", None, timeout)


# ── Compilation / PDF ─────────────────────────────────────────────────────────

def _extract_error_context(log: str, context: int = 10) -> str:
    """Extract ±context lines around each pdflatex error line from the log."""
    lines = log.strip().split("\n")
    error_indices = [i for i, l in enumerate(lines) if l.startswith("!")]
    if not error_indices:
        # No '!' errors found; return last context lines as fallback
        return "\n".join(lines[-context:])
    seen: set = set()
    result = []
    for idx in error_indices:
        start = max(0, idx - context)
        end = min(len(lines), idx + context + 1)
        for i in range(start, end):
            if i not in seen:
                seen.add(i)
                result.append(lines[i])
        result.append("---")
    return "\n".join(result)


def compile_latex(tex_path: str, work_dir: str) -> Tuple[bool, str]:
    """Compile LaTeX. Returns (pdf_produced, log_output).

    pdf_produced is True if a PDF file was generated, even when pdflatex
    reported errors — pdflatex often recovers and produces usable output.
    """
    pdf_path = os.path.join(work_dir,
                            os.path.splitext(os.path.basename(tex_path))[0] + ".pdf")
    # Remove stale PDF so we can detect fresh production
    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except OSError:
            pass
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", os.path.basename(tex_path)],
            cwd=work_dir, capture_output=True, text=True, timeout=60,
        )
        err = (result.stderr or "") + (result.stdout or "")
        pdf_produced = os.path.exists(pdf_path)
        return pdf_produced, err
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


def export_to_md(problem: str, solution: str, verification: str,
                 output_dir: str, base_name: str) -> str:
    """Export solution to a Markdown file. Returns the file path."""
    md_path = os.path.join(output_dir, f"{base_name}.md")
    os.makedirs(output_dir, exist_ok=True)
    lines = [
        "# IMO Problem Solution\n",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
        "---\n",
        "\n## Problem Statement\n",
        problem or "*(No problem statement)*", "\n",
        "\n## Solution\n",
        solution or "*(No solution yet)*", "\n",
        "\n## Verification Report\n",
        verification or "*(No verification yet)*", "\n",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return md_path


def export_to_pdf(
    problem: str, solution: str, verification: str,
    output_dir: str, base_name: str, api_key: str,
    max_attempts: int = 5, provider: str = "gemini",
    model_name: str = None, cached_tex: str = None,
) -> Tuple[bool, Optional[str]]:
    """Export solution to PDF using the fast model for LaTeX composition.

    Returns (success, final_tex_content). Falls back to Markdown on failure.
    """
    tex_path = os.path.join(output_dir, f"{base_name}-temp.tex")
    pdf_path = os.path.join(output_dir, f"{base_name}-temp.pdf")
    os.makedirs(output_dir, exist_ok=True)

    for ext in ['.pdf', '.aux', '.log', '.out', '.toc', '.synctex.gz', '.fdb_latexmk', '.fls']:
        old = os.path.join(output_dir, f"{base_name}-temp{ext}")
        if os.path.exists(old):
            try:
                os.remove(old)
            except OSError:
                pass

    skip_fast_fix = False
    latex = None

    if cached_tex:
        print("  Using cached LaTeX content...")
        latex = cached_tex
        latex_model = None
    else:
        provider_lower = provider.lower()
        fast_model = FAST_MODELS.get(provider_lower)
        if provider_lower == "kimi":
            latex_model = fast_model or "kimi-k2.5"
            print(f"  Composing LaTeX using {provider} ({latex_model}, no thinking)...")
        else:
            latex_model = fast_model or model_name or "default"
            print(f"  Composing LaTeX using {provider}'s fast model ({latex_model})...")

        try:
            latex = compose_tex_with_fast_model(
                problem, solution, verification or "Verification pending.",
                api_key, provider, latex_model,
            )
        except Exception as e:
            print(f"  Fast model failed ({e}), using res2md fallback...")
            skip_fast_fix = True
            if HAS_RES2MD:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
                    tmp = f.name
                try:
                    _res2md_memory_to_tex({"problem_statement": problem, "solution": solution}, tmp)
                    with open(tmp, "r", encoding="utf-8") as f:
                        latex = f.read()
                    if verification:
                        latex = latex.replace(
                            "\\end{document}",
                            "\n\\section{Verification Report}\n\n"
                            + _markdown_to_latex(verification) + "\n\\end{document}",
                        )
                finally:
                    os.unlink(tmp)

    if not latex or not isinstance(latex, str):
        print("  Could not generate LaTeX content.")
        print("  Falling back to Markdown export...")
        md_path = export_to_md(problem, solution, verification, output_dir, base_name)
        print(f"  \u2713 Markdown: {md_path}")
        return False, None

    if "\\begin{document}" not in latex:
        latex = latex.rstrip() + "\n\n\\begin{document}\n\n\\end{document}\n"
    if "\\end{document}" not in latex:
        latex = latex.rstrip() + "\n\n\\end{document}\n"

    for attempt in range(max_attempts):
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex)
        pdf_produced, err = compile_latex(tex_path, output_dir)
        if pdf_produced:
            # Accept the PDF even if pdflatex reported errors — it may be usable
            has_errors = "Error" in err or "error" in err
            if has_errors:
                print(f"  \u2713 PDF (with some LaTeX errors): {pdf_path}")
            else:
                print(f"  \u2713 PDF: {pdf_path}")
            open_pdf(pdf_path)
            return True, latex
        err_snippet = _extract_error_context(err)
        print(f"  Compile attempt {attempt + 1}/{max_attempts}: no PDF produced")
        if attempt < max_attempts - 1 and not skip_fast_fix and latex_model:
            try:
                latex = fix_tex_with_fast_model(latex, err_snippet, api_key, provider, latex_model)
            except Exception as e:
                print(f"  Fast model fix failed: {e}")
                break

    print("  Could not produce PDF. Falling back to Markdown export...")
    md_path = export_to_md(problem, solution, verification, output_dir, base_name)
    print(f"  \u2713 Markdown: {md_path}")
    return False, None


# ── Problem utilities ─────────────────────────────────────────────────────────

def extract_partial_solution(problem: str, latex_content: str, api_key: str,
                              provider: str = "gemini", model_name: str = None) -> str:
    """Extract and organize a partial solution from a LaTeX document."""
    prompt = PARTIAL_SOLUTION_EXTRACT_TEMPLATE.replace("<<<PROBLEM>>>", problem)
    prompt = prompt.replace("<<<LATEX>>>", latex_content)
    return call_fast_model(prompt, api_key, provider, model_name,
                           enable_thinking=True, timeout=600)


def analyze_problem(problem: str, api_key: str,
                    provider: str = "gemini", model_name: str = None) -> dict:
    """Analyze a problem statement for quality and well-definedness.

    Returns dict with keys: verdict, analysis, fixed_problem, issues.
    verdict is one of "PASS", "FIXABLE", "NEEDS_INPUT", "ERROR".
    """
    prompt = PROBLEM_ANALYSIS_PROMPT.replace("<<<PROBLEM>>>", problem)
    try:
        result = call_fast_model(prompt, api_key, provider, model_name,
                                 enable_thinking=True, timeout=300)
    except Exception as e:
        return {"verdict": "ERROR", "analysis": str(e), "fixed_problem": None, "issues": []}

    result_upper = result.upper()
    if "VERDICT" in result_upper and "PASS" in result_upper.split("VERDICT")[1][:50]:
        verdict = "PASS"
    elif "FIXABLE" in result_upper:
        verdict = "FIXABLE"
    elif "NEEDS_INPUT" in result_upper:
        verdict = "NEEDS_INPUT"
    else:
        verdict = "PASS"

    fixed_problem = None
    if verdict == "FIXABLE":
        for marker in ["**Fixed Problem**", "## Fixed Problem", "### Fixed Problem",
                       "Fixed Problem:", "**Fixed Problem:**"]:
            if marker in result:
                fixed_part = result.split(marker, 1)[1].strip()
                if fixed_part.startswith("```"):
                    lines = fixed_part.split("\n")[1:]
                    end_idx = next((i for i, l in enumerate(lines) if l.strip() == "```"), len(lines))
                    fixed_part = "\n".join(lines[:end_idx])
                fixed_problem = fixed_part.strip()
                break

    return {"verdict": verdict, "analysis": result, "fixed_problem": fixed_problem, "issues": []}
