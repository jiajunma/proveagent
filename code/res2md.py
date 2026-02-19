"""
Parse result/memory files and optionally convert to LaTeX.
Supports:
  - JSONL files (prints last JSON object)
  - Memory files with --tex: generates LaTeX from problem_statement and solution
"""

import argparse
import json
import re
import sys


# LaTeX document preamble and packages
TEX_PREAMBLE = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{geometry}
\geometry{a4paper, margin=1in}

\newtheorem{lemma}{Lemma}
\newtheorem*{lemma*}{Lemma}
\renewcommand{\qedsymbol}{$\blacksquare$}

"""


def markdown_to_latex(text: str) -> str:
    """
    Convert agent output (markdown with TeX math) to LaTeX.
    Handles: ### sections ###, **bold**, $$ display math $$, lists.
    """
    if not text:
        return ""

    # Replace $$...$$ (display math, possibly multiline) with \[...\]
    def replace_display_math(m):
        content = m.group(1).strip()
        # In JSON, \\ becomes \ so we may have \frac, \sum etc - keep as is
        return "\\[\n" + content + "\n\\]"

    text = re.sub(r"\$\$(.*?)\$\$", replace_display_math, text, flags=re.DOTALL)

    # Split into lines for structural conversion
    lines = text.split("\n")
    result = []
    in_itemize = False
    in_enumerate = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Section: ### X ### or ### X (but not ####)
        section_match = re.match(r"^###(?!#)\s*(.+?)\s*(?:###)?\s*$", line.strip())
        if section_match:
            if in_itemize:
                result.append("\\end{itemize}\n")
                in_itemize = False
            if in_enumerate:
                result.append("\\end{enumerate}\n")
                in_enumerate = False
            title = section_match.group(1).strip()
            result.append(f"\\section{{{title}}}\n\n")
            i += 1
            continue

        # Subsection: ## X ## or ## X (but not ### or ####)
        subsection_match2 = re.match(r"^##(?!#)\s*(.+?)\s*(?:##)?\s*$", line.strip())
        if subsection_match2:
            if in_itemize:
                result.append("\\end{itemize}\n")
                in_itemize = False
            if in_enumerate:
                result.append("\\end{enumerate}\n")
                in_enumerate = False
            title = subsection_match2.group(1).strip()
            result.append(f"\\subsection{{{title}}}\n\n")
            i += 1
            continue

        # Subsubsection: #### X #### or #### X
        subsubsection_match = re.match(r"^####\s*(.+?)\s*(?:####)?\s*$", line.strip())
        if subsubsection_match:
            if in_itemize:
                result.append("\\end{itemize}\n")
                in_itemize = False
            if in_enumerate:
                result.append("\\end{enumerate}\n")
                in_enumerate = False
            title = subsubsection_match.group(1).strip()
            result.append(f"\\subsubsection{{{title}}}\n\n")
            i += 1
            continue

        # Subsection: **a. Verdict** or **b. Method Sketch**
        subsection_match = re.match(r"^\*\*(.+?)\*\*\s*$", line.strip())
        if subsection_match:
            if in_itemize:
                result.append("\\end{itemize}\n")
                in_itemize = False
            if in_enumerate:
                result.append("\\end{enumerate}\n")
                in_enumerate = False
            title = subsection_match.group(1).strip()
            result.append(f"\\subsection*{{{title}}}\n\n")
            i += 1
            continue

        # Bullet list: * item or - item (possibly indented)
        bullet_match = re.match(r"^\s*[\*\-]\s+(.+)$", line)
        if bullet_match:
            if not in_itemize:
                result.append("\\begin{itemize}\n")
                in_itemize = True
            item_content = _convert_inline_bold(bullet_match.group(1))
            result.append(f"  \\item {item_content}\n")
            i += 1
            continue

        # Numbered list: "1. item" or "2. item"
        enum_match = re.match(r"^(\d+)\.\s+(.+)$", line)
        if enum_match and not line.strip().startswith("$"):
            if in_itemize:
                result.append("\\end{itemize}\n")
                in_itemize = False
            if not in_enumerate:
                result.append("\\begin{enumerate}\n")
                in_enumerate = True
            item_content = _convert_inline_bold(enum_match.group(2))
            result.append(f"  \\item {item_content}\n")
            i += 1
            continue

        # Closing list contexts on blank or new structural line
        if line.strip() == "":
            if in_itemize:
                result.append("\\end{itemize}\n")
                in_itemize = False
            if in_enumerate:
                result.append("\\end{enumerate}\n")
                in_enumerate = False
            result.append("\n")
            i += 1
            continue

        # Regular paragraph: convert **bold** to \textbf{}
        converted = _convert_inline_bold(line)
        result.append(converted + "\n")
        i += 1

    if in_itemize:
        result.append("\\end{itemize}\n")
    if in_enumerate:
        result.append("\\end{enumerate}\n")

    return "".join(result)


def _convert_inline_bold(text: str) -> str:
    """Convert **bold** to \textbf{bold}, avoiding math mode."""
    # Match **...** but not inside $...$
    parts = []
    remaining = text
    while True:
        # Find next ** that is not inside $
        dollar_count = 0
        idx = 0
        while idx < len(remaining):
            if remaining[idx] == "$" and (idx == 0 or remaining[idx - 1] != "\\"):
                dollar_count = 1 - dollar_count
                idx += 1
                continue
            if dollar_count == 0 and remaining[idx : idx + 2] == "**":
                break
            idx += 1
        if idx >= len(remaining):
            parts.append(remaining)
            break
        # Found ** at idx
        parts.append(remaining[:idx])
        start = idx + 2
        # Find closing **
        end = remaining.find("**", start)
        if end == -1:
            parts.append(remaining[idx:])
            break
        inner = remaining[start:end]
        parts.append(r"\textbf{" + inner + "}")
        remaining = remaining[end + 2 :]
    return "".join(parts)


def problem_statement_to_latex(problem: str) -> str:
    """Convert problem statement to LaTeX (minimal conversion, mostly already has $ $ math)."""
    # Remove *** Problem Statement *** header if present
    text = re.sub(r"^\*\*\* Problem Statement \*\*\*\s*\n*", "", problem)
    # Replace $$ with \[ \] for display math
    text = re.sub(r"\$\$(.*?)\$\$", r"\\[\1\\]", text, flags=re.DOTALL)
    # Convert **bold** to \textbf{bold}
    text = _convert_inline_bold(text)
    # Wrap in section
    return "\\section*{Problem Statement}\n\n" + text.strip() + "\n\n"


def memory_to_tex(memory: dict, output_path: str) -> None:
    """Generate LaTeX file from agent memory (problem_statement + solution)."""
    problem = memory.get("problem_statement", "")
    solution = memory.get("solution", "")

    problem_latex = problem_statement_to_latex(problem)
    solution_latex = markdown_to_latex(solution)

    content = (
        TEX_PREAMBLE
        + "\\begin{document}\n\n"
        + problem_latex
        + solution_latex
        + "\n\\end{document}\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Parse result/memory files, optionally convert to LaTeX")
    parser.add_argument("result_file", help="Path to result file (JSON, JSONL, or memory JSON)")
    parser.add_argument("--tex", "-t", type=str, help="Output path for generated LaTeX file (requires memory-format input)")

    args = parser.parse_args()

    with open(args.result_file, "r", encoding="utf-8") as f:
        raw = f.read()

    if args.tex:
        # Expect memory-format: single JSON object with problem_statement and solution
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Maybe JSONL - try last line
            lines = raw.strip().split("\n")
            if lines:
                data = json.loads(lines[-1])
            else:
                print("Error: No valid JSON found in file", file=sys.stderr)
                sys.exit(1)

        if "solution" not in data and "problem_statement" not in data:
            print("Error: File does not appear to be a memory file (missing solution/problem_statement)", file=sys.stderr)
            sys.exit(1)

        memory_to_tex(data, args.tex)
        print(f"LaTeX written to: {args.tex}")
    else:
        # Original behavior: print last JSON (for JSONL) or whole JSON
        try:
            data = json.loads(raw)
            print(json.dumps(data, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            lines = raw.strip().split("\n")
            if not lines:
                print("No data found in the file", file=sys.stderr)
                sys.exit(1)
            data = json.loads(lines[-1].strip())
            print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
