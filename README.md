# Math Prover Agent

An AI agent for solving IMO and research-level mathematics problems. Built on top of [lyang36/IMO25](https://github.com/lyang36/IMO25) (Huang & Yang, 2025), which demonstrated that Gemini 2.5 Pro can achieve gold-medal performance on IMO 2025.

**Author:** Ma Jia-Jun

```
MIT License
Copyright (c) 2025 Lin Yang, Yichen Huang
Copyright (c) 2026 Ma Jia-Jun
```

---

## What's New in This Fork

- **`code/interactive_agent.py`**: An interactive Claude Code–style REPL for solving math problems with slash commands, multi-provider support, and PDF export.
- **Kimi API support**: Added `code/agent_kimi.py` and `KimiProvider` in `code/model_providers.py`, supporting `kimi-k2-thinking` and other Kimi/Moonshot models.
- **Beyond IMO**: The agent works on any competition or research-level math problem — not just IMO. Drop any problem statement into a `.txt` file and run.

---

## Interactive Agent (`interactive_agent.py`)

A conversational REPL for running the solver interactively. Load a problem, run the agent, add hints, export results to PDF — all from one session.

### Prerequisites

- Python 3.7+
- At least one API key set as an environment variable:
  - `GOOGLE_API_KEY` — Google Gemini
  - `OPENAI_API_KEY` — OpenAI
  - `KIMI_API_KEY` — Kimi / Moonshot
- For PDF export: `pdflatex` (e.g. from TeX Live or MacTeX)
- Python packages:
  ```bash
  pip install -r requirements.txt
  ```

### Starting the agent

```bash
# Start with no problem loaded (interactive mode)
python code/interactive_agent.py

# Load a problem file at startup
python code/interactive_agent.py problems/imo01.txt

# Resume from a previous session (.mem file)
python code/interactive_agent.py run_logs/imo01.mem

# Explicitly specify problem or memory file
python code/interactive_agent.py --problem problems/imo01.txt
python code/interactive_agent.py --mem run_logs/imo01.mem

# Select API provider and model
python code/interactive_agent.py --provider kimi --model kimi-k2-thinking problems/imo01.txt
python code/interactive_agent.py --provider gemini --model gemini-2.5-pro problems/imo01.txt

# Disable streaming / thinking display
python code/interactive_agent.py --no-streaming --no-thinking problems/imo01.txt

# Use a custom log directory
python code/interactive_agent.py --log-dir /tmp/my_logs problems/imo01.txt

# List available memory files
python code/interactive_agent.py --list-mem

# List detected API providers
python code/interactive_agent.py --list-providers
```

**All CLI options:**

| Option | Short | Description |
|--------|-------|-------------|
| `path` | | Problem file or `.mem` memory file to load at startup |
| `--problem FILE` | `-f` | Explicitly load a problem file |
| `--mem FILE` | | Explicitly load a `.mem` memory file to resume |
| `--provider NAME` | `-p` | API provider: `gemini`, `openai`, `kimi` |
| `--model NAME` | `-m` | Model name for the selected provider |
| `--log-dir DIR` | `-d` | Directory for logs and memory files (default: `run_logs`) |
| `--no-streaming` | | Disable streaming output |
| `--no-thinking` | | Hide thinking/reasoning process |
| `--no-interactive` | | Disable interactive agent mode |
| `--list-mem` | | List available `.mem` files and exit |
| `--list-providers` | | List detected API providers and exit |

### Slash commands (inside the REPL)

Once running, use slash commands to control the session:

| Command | Short | Description |
|---------|-------|-------------|
| `/run` | `/r` | Run the agent on the current problem |
| `/problem <path>` | | Load a problem file |
| `/load <path>` | | Load a `.mem` memory file to resume a session |
| `/edit` | | Draft or refine the problem statement interactively with the agent |
| `/edit_existing` | | Browse and edit an existing problem file |
| `/done` | | Save the current draft as the problem (while in edit mode) |
| `/save_as <name>` | | Save the edited problem to a new file |
| `/prompt <text>` | `/p` | Add an extra hint/prompt for the next run |
| `/export` | `/e` | Export solution to PDF (requires `pdflatex`) |
| `/export md` | | Export solution to Markdown (fallback when PDF fails) |
| `/status` | `/s` | Show current session state |
| `/list` | `/l` | List memory files in the log directory |
| `/clear` | | Clear all added prompts |
| `/streaming on\|off` | | Enable or disable streaming output |
| `/thinking on\|off` | | Show or hide the model's thinking process |
| `/interactive on\|off` | | Enable or disable interactive agent mode |
| `/run_mode` | | Show current streaming/thinking/interactive settings |
| `/quota` | | Check API token quota status |
| `/provider <name>` | | Switch provider mid-session (`gemini`, `openai`, `kimi`) |
| `/model <name>` | | Set a specific model for the current provider |
| `/providers` | | List all detected API providers |
| `/help` | `/h` | Show command reference |
| `/quit` | `/q` | Exit |

Bare input (without a `/`) adds an extra prompt hint for the next `/run`.

### Typical workflow

```
$ python code/interactive_agent.py --provider kimi problems/imo01.txt

  IMO Interactive Agent v1.0
  Type /help for commands.

  [prob:imo01 | kimi/kimi-k2-thinking | stream:on think:on]
  > /run
  ... (agent solves the problem with streaming output) ...

  > Here's an alternative approach using generating functions
  > /run
  ... (re-runs with the extra hint) ...

  > /export
  ... (composes LaTeX, compiles PDF, opens it) ...

  > /quit
```

### Memory files

After each run the session state is saved to `<log-dir>/<base-name>.mem`. This includes the problem statement, all extra prompts, the solution, and the verification report. Use `/load` or `--mem` to resume.

---

## Original Project Components

Inherited from [lyang36/IMO25](https://github.com/lyang36/IMO25):

- **`code/agent.py`**: Single Gemini agent for IMO problem solving
- **`code/agent_oai.py`**: OpenAI variant (same CLI as `agent.py`)
- **`code/agent_xai.py`**: XAI Grok-4 variant (same CLI as `agent.py`)
- **`code/run_parallel.py`**: Parallel multi-agent runner
- **`code/res2md.py`**: Utility to parse JSONL result files

### Single agent (original)

```bash
python code/agent.py problems/imo01.txt --log run_logs/imo01.log
python code/agent_oai.py problems/imo01.txt --log run_logs/imo01_oai.log
```

### Parallel runner (original)

```bash
# Run 20 parallel agents with 5-minute timeout each
python code/run_parallel.py problems/imo01.txt -n 20 -t 300

# Exit immediately when the first solution is found
python code/run_parallel.py problems/imo01.txt -n 10 -e

# Use Kimi agent in parallel
python code/run_parallel.py problems/imo01.txt -n 10 -a code/agent_kimi.py
```

---

## Run Logs

- `run_logs/`: Gemini 2.5 Pro runs (original + interactive sessions)
- `run_logs_gpt5/`: OpenAI GPT-5 runs
- `run_logs_grok4/`: XAI Grok-4 runs

---

## License

MIT License — Copyright (c) 2025 Lin Yang, Yichen Huang; Copyright (c) 2026 Ma Jia-Jun

## Citation

```bibtex
@article{huang2025gemini,
  title={Gemini 2.5 Pro Capable of Winning Gold at IMO 2025},
  author={Huang, Yichen and Yang, Lin F},
  journal={arXiv preprint arXiv:2507.15855},
  year={2025}
}
```
