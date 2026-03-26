"""
MCP Server for IMO Proof Agent (stdio-based).

Exposes the proof agent as an OpenClaw-compatible MCP server with:
- start_proof   : launch a proof job (returns job_id immediately)
- get_status    : poll job progress + log tail + current solution
- list_proofs   : list all saved .mem files (resumable sessions)
- cancel_proof  : kill a running job
- export_pdf    : compile current best solution to PDF
"""

import json
import os
import sys
import asyncio
import threading
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

# Use standard MCP SDK
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from mcp.server.models import InitializationOptions

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(CODE_DIR)
LOG_DIR = os.path.join(REPO_DIR, "run_logs")
HINTS_DIR = os.path.join(REPO_DIR, "problems")
JOBS_FILE = os.path.join(LOG_DIR, "_mcp_jobs.json")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(HINTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Hints management (shared between MCP and agent)
# ---------------------------------------------------------------------------
def _get_hints_file(base_name: str) -> str:
    """Get the hints file path for a problem."""
    return os.path.join(HINTS_DIR, f"{base_name}_hints.txt")


def _add_hint_to_file(base_name: str, hint: str) -> str:
    """Add a hint to the hints file."""
    hints_file = _get_hints_file(base_name)
    with open(hints_file, "a", encoding="utf-8") as f:
        f.write(hint + "\n")
    return hints_file


def _read_hints(base_name: str) -> list:
    """Read all hints from the hints file."""
    hints_file = _get_hints_file(base_name)
    if not os.path.exists(hints_file):
        return []
    with open(hints_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _clear_hints(base_name: str) -> str:
    """Clear all hints for a problem."""
    hints_file = _get_hints_file(base_name)
    if os.path.exists(hints_file):
        os.remove(hints_file)
    return hints_file

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------
app = Server("imo-proof-agent")

# ---------------------------------------------------------------------------
# Job registry  (persisted to JOBS_FILE so server restarts keep history)
# ---------------------------------------------------------------------------
_registry_lock = threading.Lock()


def _load_registry() -> dict:
    try:
        if os.path.exists(JOBS_FILE):
            with open(JOBS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_registry(reg: dict):
    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)


def _update_job(job_id: str, **kwargs):
    with _registry_lock:
        reg = _load_registry()
        if job_id in reg:
            reg[job_id].update(kwargs)
            _save_registry(reg)


def _get_job(job_id: str) -> Optional[dict]:
    with _registry_lock:
        return _load_registry().get(job_id)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_job(job_id: str, job: dict):
    """Run proof agent as a subprocess, tailing stdout to a log file."""
    log_path = job["log_path"]
    mem_path = job["mem_path"]
    base_name = job["base_name"]
    problem_text = job["problem_text"]
    resume = job["resume"]
    provider = job["provider"]
    model = job.get("model") or ""

    # Build command
    cmd = [
        sys.executable,
        os.path.join(CODE_DIR, "agent_worker_subprocess.py"),
        "--problem", problem_text,
        "--mem", mem_path,
        "--log-dir", LOG_DIR,
        "--base-name", base_name,
        "--provider", provider,
    ]
    if model:
        cmd += ["--model", model]
    if resume:
        cmd += ["--resume"]

    _update_job(job_id, status="running", started_at=datetime.now().isoformat(),
                pid=None, error=None)

    try:
        with open(log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONPATH": CODE_DIR},
            )
        _update_job(job_id, pid=proc.pid)
        returncode = proc.wait()

        if returncode == 0:
            _update_job(job_id, status="done", finished_at=datetime.now().isoformat())
        else:
            _update_job(job_id, status="failed", finished_at=datetime.now().isoformat(),
                        error=f"Process exited with code {returncode}")
    except Exception as e:
        _update_job(job_id, status="failed", finished_at=datetime.now().isoformat(),
                    error=str(e))


# ---------------------------------------------------------------------------
# MCP Tool Implementations
# ---------------------------------------------------------------------------

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="start_proof",
            description="Start a proof attempt for a math problem. Returns job_id for tracking.",
            inputSchema={
                "type": "object",
                "properties": {
                    "problem_text": {
                        "type": "string",
                        "description": "The full problem statement"
                    },
                    "job_name": {
                        "type": "string",
                        "description": "Human-readable name (used as base filename). Auto-generated if empty."
                    },
                    "resume_from": {
                        "type": "string",
                        "description": "job_id of a previous session to resume from its .mem file."
                    },
                    "provider": {
                        "type": "string",
                        "description": "Model provider: 'gemini' (default), 'openai', or 'kimi'.",
                        "default": "gemini"
                    },
                    "model": {
                        "type": "string",
                        "description": "Specific model name override (e.g. 'gemini-2.5-pro')."
                    }
                },
                "required": ["problem_text"]
            }
        ),
        Tool(
            name="get_status",
            description="Get the current status of a proof job.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID returned by start_proof"
                    },
                    "log_lines": {
                        "type": "number",
                        "description": "How many trailing lines of the agent log to include.",
                        "default": 30
                    }
                },
                "required": ["job_id"]
            }
        ),
        Tool(
            name="list_proofs",
            description="List all proof sessions (both running and completed).",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="cancel_proof",
            description="Cancel a running proof job.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID to cancel"
                    }
                },
                "required": ["job_id"]
            }
        ),
        Tool(
            name="export_pdf",
            description="Compile the current best solution in a proof job to a PDF.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID whose .mem file contains the solution"
                    }
                },
                "required": ["job_id"]
            }
        ),
        Tool(
            name="add_hint",
            description="Add a hint to a running proof job. The agent will read this hint on its next iteration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID to add the hint to"
                    },
                    "hint": {
                        "type": "string",
                        "description": "The hint text (e.g., 'Try using induction', 'Consider case analysis', etc.)"
                    }
                },
                "required": ["job_id", "hint"]
            }
        ),
        Tool(
            name="get_solution",
            description="Get the current solution from a proof job.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID to get the solution from"
                    }
                },
                "required": ["job_id"]
            }
        ),
        Tool(
            name="list_hints",
            description="List all pending hints for a proof job.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID to list hints for"
                    }
                },
                "required": ["job_id"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "start_proof":
        result = await _start_proof(
            problem_text=arguments.get("problem_text", ""),
            job_name=arguments.get("job_name", ""),
            resume_from=arguments.get("resume_from", ""),
            provider=arguments.get("provider", "gemini"),
            model=arguments.get("model", "")
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
    
    elif name == "get_status":
        result = await _get_status(
            job_id=arguments.get("job_id", ""),
            log_lines=arguments.get("log_lines", 30)
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
    
    elif name == "list_proofs":
        result = await _list_proofs()
        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
    
    elif name == "cancel_proof":
        result = await _cancel_proof(job_id=arguments.get("job_id", ""))
        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
    
    elif name == "export_pdf":
        result = await _export_pdf(job_id=arguments.get("job_id", ""))
        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
    
    elif name == "add_hint":
        result = await _add_hint(
            job_id=arguments.get("job_id", ""),
            hint=arguments.get("hint", "")
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
    
    elif name == "get_solution":
        result = await _get_solution(job_id=arguments.get("job_id", ""))
        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
    
    elif name == "list_hints":
        result = await _list_hints(job_id=arguments.get("job_id", ""))
        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
    
    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def _start_proof(
    problem_text: str,
    job_name: str = "",
    resume_from: str = "",
    provider: str = "gemini",
    model: str = "",
) -> dict:
    """Start a proof job."""
    
    # Determine base name & paths
    if not job_name:
        job_name = "proof_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = job_name.replace(" ", "_")
    mem_path = os.path.join(LOG_DIR, f"{base_name}.mem")
    log_path = os.path.join(LOG_DIR, f"{base_name}.log")

    resume = False
    if resume_from:
        prev = _get_job(resume_from)
        if prev and os.path.exists(prev["mem_path"]):
            mem_path = prev["mem_path"]
            base_name = prev["base_name"]
            log_path = prev["log_path"]
            resume = True
        else:
            return {"error": f"Cannot resume: job '{resume_from}' not found or has no .mem file."}

    job_id = str(uuid.uuid4())[:8]
    job = {
        "job_id": job_id,
        "job_name": job_name,
        "base_name": base_name,
        "problem_text": problem_text,
        "mem_path": mem_path,
        "log_path": log_path,
        "provider": provider,
        "model": model,
        "resume": resume,
        "status": "starting",
        "pid": None,
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "finished_at": None,
        "error": None,
    }

    with _registry_lock:
        reg = _load_registry()
        reg[job_id] = job
        _save_registry(reg)

    t = threading.Thread(target=_run_job, args=(job_id, job), daemon=True)
    t.start()

    return {
        "job_id": job_id,
        "status": "starting",
        "mem_path": mem_path,
        "log_path": log_path,
        "message": f"Proof job '{job_name}' started. Use get_status('{job_id}') to track progress.",
    }


async def _get_status(job_id: str, log_lines: int = 30) -> dict:
    """Get job status."""
    
    job = _get_job(job_id)
    if not job:
        return {"error": f"Job '{job_id}' not found."}

    result = {
        "job_id": job_id,
        "job_name": job["job_name"],
        "status": job["status"],
        "provider": job["provider"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "finished_at": job.get("finished_at"),
        "error": job.get("error"),
        "mem_path": job["mem_path"],
    }

    # Check if process is still alive
    pid = job.get("pid")
    if pid and job["status"] == "running":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            result["status"] = "done (process ended)"

    # Load mem file for current solution preview
    mem_path = job["mem_path"]
    if os.path.exists(mem_path):
        try:
            with open(mem_path, "r", encoding="utf-8") as f:
                mem = json.load(f)
            result["iterations"] = mem.get("current_iteration", 0)
            result["max_runs"] = mem.get("max_runs", "?")
            sol = mem.get("solution") or ""
            result["solution_preview"] = sol[:500] + ("..." if len(sol) > 500 else "")
            result["has_verification"] = bool(mem.get("verify"))
            result["last_saved"] = mem.get("timestamp", "")
        except Exception as e:
            result["mem_error"] = str(e)

    # Tail the log file
    log_path = job["log_path"]
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            result["log_tail"] = "".join(lines[-log_lines:])
            result["log_total_lines"] = len(lines)
        except Exception as e:
            result["log_error"] = str(e)

    return result


async def _list_proofs() -> list:
    """List all proof sessions."""
    
    with _registry_lock:
        reg = _load_registry()

    results = []
    known_mems = set()

    # Registered jobs
    for job_id, job in reg.items():
        entry = {
            "job_id": job_id,
            "job_name": job["job_name"],
            "status": job["status"],
            "provider": job.get("provider", "?"),
            "created_at": job["created_at"],
            "finished_at": job.get("finished_at"),
            "mem_path": job["mem_path"],
            "mem_exists": os.path.exists(job["mem_path"]),
        }
        if entry["mem_exists"]:
            try:
                with open(job["mem_path"], "r", encoding="utf-8") as f:
                    m = json.load(f)
                entry["iterations"] = m.get("current_iteration", 0)
                entry["has_solution"] = bool(m.get("solution"))
            except Exception:
                pass
        known_mems.add(job["mem_path"])
        results.append(entry)

    # Orphan .mem files
    for mem_file in Path(LOG_DIR).glob("*.mem"):
        if str(mem_file) not in known_mems:
            try:
                with open(mem_file, "r", encoding="utf-8") as f:
                    m = json.load(f)
                results.append({
                    "job_id": None,
                    "job_name": mem_file.stem,
                    "status": "orphan (interactive session)",
                    "mem_path": str(mem_file),
                    "mem_exists": True,
                    "iterations": m.get("current_iteration", 0),
                    "has_solution": bool(m.get("solution")),
                    "created_at": m.get("timestamp", ""),
                })
            except Exception:
                pass

    results.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return results


async def _cancel_proof(job_id: str) -> dict:
    """Cancel a running proof job."""
    
    job = _get_job(job_id)
    if not job:
        return {"error": f"Job '{job_id}' not found."}

    if job["status"] not in ("running", "starting"):
        return {"message": f"Job '{job_id}' is already {job['status']}."}

    pid = job.get("pid")
    if pid:
        try:
            import signal
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass
        except Exception as e:
            return {"error": f"Failed to kill process {pid}: {e}"}

    _update_job(job_id, status="cancelled", finished_at=datetime.now().isoformat())
    return {"message": f"Job '{job_id}' cancelled. Memory file preserved for resume."}


async def _export_pdf(job_id: str) -> dict:
    """Export solution to PDF."""
    
    job = _get_job(job_id)
    if not job:
        # Try treating job_id as a mem file stem
        candidate = os.path.join(LOG_DIR, f"{job_id}.mem")
        if os.path.exists(candidate):
            mem_path = candidate
            base_name = job_id
        else:
            return {"error": f"Job '{job_id}' not found."}
    else:
        mem_path = job["mem_path"]
        base_name = job["base_name"]

    if not os.path.exists(mem_path):
        return {"error": "No .mem file found — the agent hasn't saved any progress yet."}

    try:
        with open(mem_path, "r", encoding="utf-8") as f:
            mem = json.load(f)
    except Exception as e:
        return {"error": f"Could not read .mem file: {e}"}

    solution = mem.get("solution") or ""
    if not solution:
        return {"error": "No solution in memory yet."}

    problem = mem.get("problem_statement") or ""
    verify = mem.get("verify") or ""

    # Run export in subprocess
    export_script = os.path.join(CODE_DIR, "export_pdf_subprocess.py")
    cmd = [
        sys.executable, export_script,
        "--problem", problem,
        "--solution", solution,
        "--verify", verify,
        "--log-dir", LOG_DIR,
        "--base-name", base_name,
        "--provider", (job or {}).get("provider", "gemini"),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "PYTHONPATH": CODE_DIR},
        )
        if result.returncode == 0:
            pdf_path = os.path.join(LOG_DIR, f"{base_name}-temp.pdf")
            return {
                "success": True,
                "pdf_path": pdf_path,
                "stdout": result.stdout[-500:],
            }
        else:
            return {
                "success": False,
                "error": result.stderr[-500:] or result.stdout[-500:],
            }
    except subprocess.TimeoutExpired:
        return {"error": "PDF export timed out after 5 minutes."}
    except Exception as e:
        return {"error": str(e)}


async def _add_hint(job_id: str, hint: str) -> dict:
    """Add a hint to a proof job."""
    
    job = _get_job(job_id)
    if not job:
        return {"error": f"Job '{job_id}' not found."}
    
    base_name = job["base_name"]
    hints_file = _add_hint_to_file(base_name, hint)
    
    return {
        "success": True,
        "job_id": job_id,
        "hint": hint,
        "hints_file": hints_file,
        "message": f"Hint added. Agent will read it on next iteration."
    }


async def _get_solution(job_id: str) -> dict:
    """Get the current solution from a proof job."""
    
    job = _get_job(job_id)
    if not job:
        # Try treating job_id as a mem file stem
        candidate = os.path.join(LOG_DIR, f"{job_id}.mem")
        if os.path.exists(candidate):
            mem_path = candidate
            base_name = job_id
        else:
            return {"error": f"Job '{job_id}' not found."}
    else:
        mem_path = job["mem_path"]
        base_name = job["base_name"]
    
    if not os.path.exists(mem_path):
        return {"error": "No .mem file found.", "has_solution": False}
    
    try:
        with open(mem_path, "r", encoding="utf-8") as f:
            mem = json.load(f)
        
        solution = mem.get("solution", "")
        return {
            "job_id": job_id,
            "base_name": base_name,
            "has_solution": bool(solution),
            "solution": solution,
            "iterations": mem.get("current_iteration", 0),
            "timestamp": mem.get("timestamp", "")
        }
    except Exception as e:
        return {"error": f"Could not read .mem file: {e}"}


async def _list_hints(job_id: str) -> dict:
    """List all hints for a proof job."""
    
    job = _get_job(job_id)
    if not job:
        # Try treating job_id as a mem file stem
        candidate = os.path.join(LOG_DIR, f"{job_id}.mem")
        if os.path.exists(candidate):
            base_name = job_id
        else:
            return {"error": f"Job '{job_id}' not found."}
    else:
        base_name = job["base_name"]
    
    hints = _read_hints(base_name)
    hints_file = _get_hints_file(base_name)
    
    return {
        "job_id": job_id,
        "base_name": base_name,
        "hints_file": hints_file,
        "hints": hints,
        "count": len(hints)
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def main():
    """Run the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="imo-proof-agent",
                server_version="1.0.0",
                capabilities={}
            )
        )


if __name__ == "__main__":
    asyncio.run(main())