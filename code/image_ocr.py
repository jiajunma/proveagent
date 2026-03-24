"""Clipboard image capture and multimodal OCR for the interactive IMO agent."""

import base64
import json
import os
import platform
import subprocess
import tempfile
from typing import Optional

import requests

from interactive_prompts import FAST_MODELS, IMAGE_OCR_PROMPT
from model_providers import _post_with_fallback


def get_clipboard_image() -> Optional[str]:
    """Save clipboard image to a temp PNG file. Returns path or None if no image."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()

    system = platform.system()
    success = False
    try:
        if system == "Darwin":
            try:
                r = subprocess.run(["pngpaste", tmp_path], capture_output=True, timeout=10)
                if r.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 100:
                    success = True
            except FileNotFoundError:
                pass
            if not success:
                script = (
                    "try\n"
                    "    set imgData to (the clipboard as \u00abclass PNGf\u00bb)\n"
                    f"    set fileRef to open for access POSIX file \"{tmp_path}\" with write permission\n"
                    "    write imgData to fileRef\n"
                    "    close access fileRef\n"
                    "on error\n"
                    "    return \"error\"\n"
                    "end try"
                )
                r = subprocess.run(["osascript", "-e", script],
                                   capture_output=True, text=True, timeout=15)
                if r.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 100:
                    success = True
        elif system == "Linux":
            for _tool, args in [
                ("xclip", ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"]),
                ("xsel",  ["xsel", "--clipboard", "--output"]),
            ]:
                try:
                    r = subprocess.run(args, capture_output=True, timeout=10)
                    if r.returncode == 0 and r.stdout and len(r.stdout) > 100:
                        with open(tmp_path, "wb") as f:
                            f.write(r.stdout)
                        success = True
                        break
                except FileNotFoundError:
                    continue
        elif system == "Windows":
            ps = (
                "Add-Type -Assembly System.Windows.Forms; "
                "Add-Type -Assembly System.Drawing; "
                "$img = [System.Windows.Forms.Clipboard]::GetImage(); "
                f"if ($img) {{ $img.Save('{tmp_path}') }}"
            )
            r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                               capture_output=True, timeout=15)
            if r.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 100:
                success = True
    except Exception:
        pass

    if success:
        return tmp_path
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
    return None


def ocr_image_to_latex(image_path: str, api_key: str,
                        provider: str = "gemini", model_name: str = None) -> str:
    """Send image to a multimodal LLM and extract LaTeX content."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    provider_lower = provider.lower()

    if provider_lower == "gemini":
        model = model_name or FAST_MODELS.get("gemini", "gemini-2.5-flash-lite")
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [
                {"inlineData": {"mimeType": "image/png", "data": image_b64}},
                {"text": IMAGE_OCR_PROMPT},
            ]}],
            "generationConfig": {"temperature": 0.0},
        }
        headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}
        resp = _post_with_fallback(api_url, headers=headers, data=json.dumps(payload), timeout=120)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    elif provider_lower in ("openai", "gpt"):
        model = model_name or "gpt-4o"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": IMAGE_OCR_PROMPT},
            ]}],
            "temperature": 0.0,
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        resp = requests.post("https://api.openai.com/v1/chat/completions",
                             headers=headers, data=json.dumps(payload), timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    elif provider_lower in ("kimi", "moonshot"):
        kimi_key = os.environ.get("KIMI_API_KEY", api_key)
        model = model_name or "moonshot-v1-32k-vision-preview"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": IMAGE_OCR_PROMPT},
            ]}],
            "temperature": 0.3,
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {kimi_key}"}
        resp = requests.post("https://api.moonshot.cn/v1/chat/completions",
                             headers=headers, data=json.dumps(payload), timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    else:
        return ocr_image_to_latex(image_path, api_key, "gemini", model_name)
