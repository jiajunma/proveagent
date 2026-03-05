"""Fast/cheap LLM client for interactive agent tasks (LaTeX, problem editing, etc.)."""

import json
import requests
from interactive_prompts import FAST_MODELS


def call_fast_model(prompt: str, api_key: str, provider: str = "gemini",
                    model_name: str = None, enable_thinking: bool = True,
                    timeout: int = 300) -> str:
    """Call the fast/cheap model for the specified provider."""
    provider = provider.lower()

    if provider == "gemini":
        model = model_name or FAST_MODELS.get("gemini", "gemini-2.5-flash-lite")
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
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        resp = requests.post("https://api.openai.com/v1/chat/completions",
                             headers=headers, data=json.dumps(payload), timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    elif provider == "kimi":
        model = model_name or FAST_MODELS.get("kimi", "kimi-k2.5")
        temp = 1.0 if model == "kimi-k2.5" else 0.3
        latex_task = any(m in prompt for m in ["LaTeX", "latex", "\\documentclass", "\\begin{document}"])
        kimi_timeout = 600 if latex_task else timeout
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
        }
        if not enable_thinking and "thinking" in model:
            payload["extra_body"] = {"enable_thinking": False}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        resp = requests.post("https://api.moonshot.cn/v1/chat/completions",
                             headers=headers, data=json.dumps(payload), timeout=kimi_timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    else:
        return call_fast_model(prompt, api_key, "gemini", model_name, enable_thinking, timeout)


def call_fast_model_chat(system_prompt: str, contents: list, api_key: str,
                         provider: str = "gemini", model_name: str = None,
                         enable_thinking: bool = True, timeout: int = 300) -> str:
    """Multi-turn chat with fast model.

    contents format for gemini: [{"role":"user","parts":[{"text":"..."}]}, ...]
    contents format for openai/kimi: [{"role":"user","content":"..."}, ...]
    """
    provider = provider.lower()

    if provider == "gemini":
        model = model_name or FAST_MODELS.get("gemini", "gemini-2.5-flash-lite")
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

    elif provider in ("openai", "kimi"):
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        for item in contents:
            role = item.get("role", "user")
            if "parts" in item and item["parts"]:
                content = item["parts"][0].get("text", "")
            else:
                content = item.get("content", "")
            messages.append({"role": role, "content": content})

        if provider == "openai":
            model = model_name or FAST_MODELS.get("openai", "gpt-4o-mini")
            payload = {"model": model, "messages": messages, "temperature": 0.3}
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            resp = requests.post("https://api.openai.com/v1/chat/completions",
                                 headers=headers, data=json.dumps(payload), timeout=timeout)
        else:  # kimi
            model = model_name or FAST_MODELS.get("kimi", "kimi-k2.5")
            temp = 1.0 if model == "kimi-k2.5" else 0.3
            latex_task = any(
                any(m in msg.get("content", "") for m in ["LaTeX", "latex", "\\documentclass"])
                for msg in messages
            )
            kimi_timeout = 600 if latex_task else timeout
            payload = {"model": model, "messages": messages, "temperature": temp}
            if not enable_thinking and "thinking" in model:
                payload["extra_body"] = {"enable_thinking": False}
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            resp = requests.post("https://api.moonshot.cn/v1/chat/completions",
                                 headers=headers, data=json.dumps(payload), timeout=kimi_timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    else:
        return call_fast_model_chat(system_prompt, contents, api_key, "gemini",
                                    model_name, enable_thinking, timeout)


# Legacy aliases
def call_flash(prompt: str, api_key: str, timeout: int = 300) -> str:
    return call_fast_model(prompt, api_key, "gemini", None, True, timeout)


def call_flash_chat(system_prompt: str, contents: list, api_key: str, timeout: int = 300) -> str:
    return call_fast_model_chat(system_prompt, contents, api_key, "gemini", None, True, timeout)
