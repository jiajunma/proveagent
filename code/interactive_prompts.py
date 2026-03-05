"""Prompt templates and constants for the interactive IMO agent."""

# Fast/cheap models used for LaTeX composition, problem analysis, OCR, etc.
FAST_MODELS = {
    "gemini": "gemini-2.5-flash-lite",
    "openai": "gpt-4o-mini",
    "kimi": "kimi-k2.5",  # kimi-k2.5 without thinking for LaTeX tasks
}

# Legacy constants
FLASH_MODEL = FAST_MODELS["gemini"]
FLASH_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{FLASH_MODEL}:generateContent"

TEX_PREAMBLE = r"""\documentclass{article}
\usepackage{fontspec}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{geometry}
\geometry{a4paper, margin=1in}

"""

TEX_COMPOSE_TEMPLATE = """You are a LaTeX expert. Convert the following mathematical solution and verification report into a single, valid LaTeX document.

Requirements:
- Use XeLaTeX-compatible packages: use \\usepackage{fontspec} instead of inputenc/fontenc
- Do NOT use \\usepackage[utf8]{inputenc} or \\usepackage[T1]{fontenc}
- Preserve all mathematical content exactly; use $...$ for inline math, \\[...\\] for display math
- Convert markdown **bold** to \\textbf{}, ### sections to \\section{}, etc.
- Output ONLY the complete LaTeX source, no explanations
- Keep it concise but complete

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

PARTIAL_SOLUTION_EXTRACT_TEMPLATE = """You are an expert mathematician. You are given a problem statement and a LaTeX document that contains a partial or draft solution (possibly incomplete, informal, or mixed with scratch work).

Your task is to extract and reorganize the mathematical content into a clean **partial solution** that can serve as a starting point for further rigorous proof work.

### Output Format ###

Your response MUST follow this exact structure:

**1. Summary**

*   **a. Verdict:** State clearly: "This is a partial solution." Then list the main rigorous conclusions that have been established so far.
*   **b. Method Sketch:** Describe the overall strategy attempted so far, including:
    - What has been proven rigorously
    - What key lemmas or intermediate results are established
    - What remains to be proven or what gaps exist
    - Any promising directions or approaches identified but not yet completed

**2. Detailed Solution**

Present the rigorous parts of the solution in a clean, step-by-step format:
- Include all definitions, lemmas, and proofs that are mathematically sound
- Clearly mark where the proof is incomplete with comments like "[TODO: ...]" or "[Gap: ...]"
- Preserve all correct mathematical reasoning from the original
- Remove scratch work, dead ends, and informal notes that don't contribute to the proof
- Use TeX for all mathematics: $...$ for inline, \\[...\\] for display

### Important Guidelines ###
- Be faithful to the original content — do NOT invent new proofs or fill gaps yourself
- If the original contains errors, note them but include the surrounding correct work
- Organize the content logically even if the original is disorganized
- Keep all established results, even if the overall proof is incomplete

=== PROBLEM STATEMENT ===
<<<PROBLEM>>>

=== LATEX DOCUMENT ===
<<<LATEX>>>
"""

PROBLEM_ANALYSIS_PROMPT = """You are an expert mathematician. Analyze the following mathematical problem statement for quality, completeness, and well-definedness.

Check for the following issues:

1. **Undefined terms**: Are all mathematical objects, sets, or structures clearly defined? (e.g., does it say "for all $n$" without specifying $n \\in \\mathbb{Z}^+$?)
2. **Ambiguity**: Is the problem statement unambiguous? Could it be interpreted in multiple ways?
3. **Missing constraints**: Are there necessary constraints or conditions that are missing? (e.g., boundedness, finiteness, positivity)
4. **Goal clarity**: Is it clear what needs to be proved, found, or computed?
5. **Mathematical correctness**: Does the statement make mathematical sense? Are there obvious contradictions?
6. **Notation consistency**: Is mathematical notation used consistently?
7. **Self-containedness**: Can the problem be understood without external references?

### Output Format ###

**Verdict**: One of:
- "PASS" — the problem is well-defined, complete, and ready for solving.
- "FIXABLE" — there are issues but they can be fixed automatically (list the fixes).
- "NEEDS_INPUT" — there are issues that require human clarification (list the questions).

**Issues** (if any): A numbered list of issues found, each with:
- Category (from the list above)
- Description of the issue
- Suggested fix (for FIXABLE issues) or question to ask the user (for NEEDS_INPUT)

**Fixed Problem** (only if verdict is FIXABLE): Output the corrected problem statement with all fixes applied. Preserve the original structure and style as much as possible.

=== PROBLEM STATEMENT ===
<<<PROBLEM>>>
"""

EDIT_PROBLEM_SYSTEM = """You help the user formulate or refine a mathematical problem for an IMO-style solver.

- Ask clarifying questions. Suggest structure (hypotheses, goal, constraints).
- Use TeX for math: $n$, $\\mathbb{R}$, $$display math$$
- When the user indicates they're done (e.g. "done", "that's it"), output ONLY the final problem statement, nothing else.
- The final problem should be self-contained, clear, and suitable for rigorous proof.
- Be concise."""

IMAGE_OCR_PROMPT = """You are an expert mathematician and LaTeX typesetter. Extract ALL content from the image and convert it to clean LaTeX.

Rules:
- Convert all mathematics to proper LaTeX: $...$ for inline, \\[...\\] for display
- Preserve the logical structure (problem, lemma, proof steps, remarks, etc.)
- For handwritten content: transcribe carefully, mark illegible parts as [?]
- Do NOT add \\documentclass, \\begin{document} — output body content only
- Do NOT add explanatory commentary — output ONLY the extracted content

Output the LaTeX content now:"""
