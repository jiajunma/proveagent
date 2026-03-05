"""Subprocess worker for running the proof agent in an isolated process."""

import json
import os
import sys
from typing import Optional

from latex_pipeline import export_to_pdf


def run_agent_worker(
    problem_statement: str,
    other_prompts: list,
    verify_prompts: list,
    memory_file: Optional[str],
    resume: bool,
    log_dir: str,
    base_name: str,
    result_queue,
    streaming: bool = True,
    show_thinking: bool = True,
    interactive: bool = True,
    provider_name: str = "gemini",
    model_name: str = None,
) -> None:
    """Run the proof agent in a subprocess; put (status, result) in result_queue.

    Designed to be the target of multiprocessing.Process so that terminating
    the process aborts any in-flight API calls.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        if provider_name.lower() in ["openai", "gpt"]:
            import agent_oai as ag
            api_key = os.environ.get("OPENAI_API_KEY", "")
        elif provider_name.lower() in ["kimi", "moonshot"]:
            print("  Using Kimi API")
            import agent_kimi as ag
            api_key = os.environ.get("KIMI_API_KEY", "")
        else:
            import agent as ag
            api_key = os.environ.get("GOOGLE_API_KEY", "")
    except ImportError as e:
        print(f"  Error importing agent module for {provider_name}: {e}")
        print("  Falling back to default agent module")
        import agent as ag
        api_key = os.environ.get("GOOGLE_API_KEY", "")

    def on_result(prob, sol, verif, _iter):
        print(f"  [iter {_iter}] Exporting PDF...")
        ok, final_tex = export_to_pdf(prob, sol, verif, log_dir, base_name, api_key,
                                      provider=provider_name, model_name=model_name)
        if ok and final_tex and memory_file:
            try:
                if os.path.exists(memory_file):
                    with open(memory_file, "r", encoding="utf-8") as f:
                        mem_data = json.load(f)
                    mem_data["cached_tex"] = final_tex
                    with open(memory_file, "w", encoding="utf-8") as f:
                        json.dump(mem_data, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

    log_path = os.path.join(log_dir, f"{base_name}_interactive.prooflog")
    ag.set_log_file(log_path)

    try:
        if model_name:
            if hasattr(ag, "set_model"):
                ag.set_model(model_name)
            elif hasattr(ag, "MODEL_NAME"):
                ag.MODEL_NAME = model_name
                print(f"  Set model to {model_name}")

        sol = ag.agent(
            problem_statement,
            other_prompts,
            verify_prompts=verify_prompts if verify_prompts else None,
            memory_file=memory_file,
            resume_from_memory=resume,
            on_iteration_result=on_result,
            streaming=streaming,
            show_thinking=show_thinking,
            interactive=interactive,
        )
        result_queue.put(("ok", sol))
    except Exception as e:
        result_queue.put(("error", str(e)))
    finally:
        ag.close_log_file()
