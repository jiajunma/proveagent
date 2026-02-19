"""
模型提供商抽象层
支持多种API提供商，如Google Gemini、OpenAI和Kimi
"""

import os
import sys
import json
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple

# 全局常量
DEFAULT_TIMEOUT = 7200  # API请求超时时间（秒）

class ModelProvider(ABC):
    """模型提供商的抽象基类"""

    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key
        self.model_name = model_name
        self.streaming_supported = False

    @abstractmethod
    def get_api_key(self) -> str:
        """获取API密钥"""
        pass

    @abstractmethod
    def build_request_payload(self, system_prompt: str,
                             question_prompt: str,
                             other_prompts: List[str] = None,
                             enable_thinking: bool = True,
                             streaming: bool = True) -> Dict:
        """构建API请求负载"""
        pass

    @abstractmethod
    def send_api_request(self, payload: Dict, streaming: bool = False,
                         show_thinking: bool = False) -> Dict:
        """发送API请求并返回响应"""
        pass

    @abstractmethod
    def extract_text_from_response(self, response_data: Dict) -> Tuple[str, str]:
        """从API响应中提取生成的文本和思考过程"""
        pass

    @abstractmethod
    def check_capabilities(self) -> bool:
        """检查模型能力（如是否支持流式输出）"""
        pass

    def get_name(self) -> str:
        """获取提供商名称"""
        return self.__class__.__name__.replace('Provider', '')


class GeminiProvider(ModelProvider):
    """Google Gemini API提供商实现"""

    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-pro"):
        super().__init__(api_key, model_name)
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    def get_api_key(self) -> str:
        """获取Google API密钥"""
        if self.api_key:
            return self.api_key

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable not set.")
            print("Please set the variable, e.g., 'export GOOGLE_API_KEY=\"your_api_key\"'")
            sys.exit(1)
        self.api_key = api_key
        return api_key

    def build_request_payload(self, system_prompt: str,
                             question_prompt: str,
                             other_prompts: List[str] = None,
                             enable_thinking: bool = True,
                             streaming: bool = True) -> Dict:
        """构建Gemini API请求负载"""
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": question_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 1.0,
            },
        }

        # 添加系统指令
        if system_prompt.strip():
            payload["systemInstruction"] = {
                "role": "system",
                "parts": [
                    {
                        "text": system_prompt
                    }
                ]
            }

        # 添加思考配置
        if enable_thinking and self.streaming_supported:
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": 32768
            }

        # 添加流式输出配置
        if streaming and self.streaming_supported:
            if enable_thinking:
                payload["generationConfig"]["streamingBehavior"] = "THINKING_AND_TEXT"
            else:
                payload["generationConfig"]["streamingBehavior"] = "TEXT_ONLY"

        # 添加其他提示
        if other_prompts:
            for prompt in other_prompts:
                payload["contents"].append({
                    "role": "user",
                    "parts": [{"text": prompt}]
                })

        return payload

    def get_api_url(self, streaming=False):
        """获取API URL"""
        base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}"

        if streaming and self.streaming_supported:
            return f"{base_url}:generateContent?alt=sse"

        return f"{base_url}:generateContent"

    def send_api_request(self, payload: Dict, streaming: bool = False,
                         show_thinking: bool = False) -> Dict:
        """发送Gemini API请求并返回响应"""
        api_key = self.get_api_key()
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": api_key
        }

        # 非流式请求
        if not streaming:
            api_url = self.get_api_url(streaming=False)
            try:
                response = requests.post(api_url, headers=headers,
                                        data=json.dumps(payload),
                                        timeout=DEFAULT_TIMEOUT)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error during Gemini API request: {e}")
                if hasattr(e, 'response') and e.response.status_code == 400:
                    print(f"Raw API Response: {e.response.text[:200]}")
                raise e

        # 流式请求处理
        else:
            api_url = self.get_api_url(streaming=True)
            try:
                accumulated_text = ""
                thinking_text = ""
                thinking_active = False

                with requests.Session() as session:
                    with session.post(api_url, headers=headers,
                                     data=json.dumps(payload),
                                     stream=True) as response:
                        response.raise_for_status()

                        for line in response.iter_lines():
                            if not line:
                                continue

                            line_text = line.decode('utf-8')
                            if not line_text.startswith("data: "):
                                continue

                            data = line_text[6:]  # 跳过 "data: " 前缀

                            if data == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data)

                                # 处理思考内容
                                if "thinking" in chunk:
                                    if show_thinking:
                                        thinking_text += chunk["thinking"]
                                        if not thinking_active:
                                            thinking_active = True
                                            print("\n--- Thinking Process ---")
                                        print(chunk["thinking"], end="", flush=True)
                                    continue

                                # 思考结束标志
                                if thinking_active and "candidates" in chunk:
                                    thinking_active = False
                                    print("\n--- End of Thinking ---\n")

                                # 提取生成的文本
                                if "candidates" in chunk and chunk["candidates"]:
                                    candidate = chunk["candidates"][0]
                                    if "content" in candidate and "parts" in candidate["content"]:
                                        for part in candidate["content"]["parts"]:
                                            if "text" in part:
                                                text_chunk = part["text"]
                                                accumulated_text += text_chunk
                                                # 打印流式输出
                                                print(text_chunk, end="", flush=True)
                            except json.JSONDecodeError:
                                print(f"Error parsing chunk: {data[:100]}")
                                continue

                # 构建类似于非流式响应的对象
                return {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": accumulated_text}]
                            }
                        }
                    ],
                    "thinking_process": thinking_text
                }

            except requests.exceptions.RequestException as e:
                print(f"Error during streaming Gemini API request: {e}")
                if hasattr(e, 'response') and e.response.status_code == 400:
                    print(f"Raw API Response: {e.response.text[:200]}")
                # 回退到非流式请求
                return self.send_api_request(payload, streaming=False, show_thinking=False)

    def extract_text_from_response(self, response_data: Dict) -> Tuple[str, str]:
        """从Gemini API响应中提取文本和思考过程"""
        try:
            text = response_data['candidates'][0]['content']['parts'][0]['text']
            thinking = response_data.get('thinking_process', '')
            return text, thinking
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error extracting text from Gemini API response: {e}")
            print(json.dumps(response_data, indent=2)[:500])
            return "", ""

    def check_capabilities(self) -> bool:
        """检查Gemini模型是否支持流式输出和思考过程"""
        api_key = self.get_api_key()
        test_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?alt=sse"
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": api_key
        }

        minimal_payload = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 10,
                "streamingBehavior": "TEXT_ONLY"
            }
        }

        try:
            response = requests.post(test_url, headers=headers,
                                    data=json.dumps(minimal_payload),
                                    stream=True, timeout=5)
            if response.status_code == 200:
                self.streaming_supported = True
                print(f"Model {self.model_name} supports streaming")
                return True
        except Exception:
            pass

        self.streaming_supported = False
        print(f"Model {self.model_name} does not support streaming")
        return False


class OpenAIProvider(ModelProvider):
    """OpenAI API提供商实现"""

    def __init__(self, api_key: str = None, model_name: str = "gpt-4o"):
        super().__init__(api_key, model_name)
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def get_api_key(self) -> str:
        """获取OpenAI API密钥"""
        if self.api_key:
            return self.api_key

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set.")
            print("Please set the variable, e.g., 'export OPENAI_API_KEY=\"your_api_key\"'")
            sys.exit(1)
        self.api_key = api_key
        return api_key

    def build_request_payload(self, system_prompt: str,
                             question_prompt: str,
                             other_prompts: List[str] = None,
                             enable_thinking: bool = True,
                             streaming: bool = True) -> Dict:
        """构建OpenAI API请求负载"""
        messages = []

        # 添加系统提示
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        # 添加主要问题
        messages.append({"role": "user", "content": question_prompt})

        # 添加其他提示
        if other_prompts:
            for prompt in other_prompts:
                messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
        }

        # 添加流式输出配置
        if streaming and self.streaming_supported:
            payload["stream"] = True

        # OpenAI不支持思考过程的显示，所以忽略enable_thinking参数

        return payload

    def send_api_request(self, payload: Dict, streaming: bool = False,
                         show_thinking: bool = False) -> Dict:
        """发送OpenAI API请求并返回响应"""
        api_key = self.get_api_key()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # 流式请求仅在支持时启用
        use_streaming = streaming and self.streaming_supported

        if use_streaming:
            payload["stream"] = True

            try:
                accumulated_text = ""

                with requests.Session() as session:
                    with session.post(self.api_url, headers=headers,
                                     json=payload,
                                     stream=True) as response:
                        response.raise_for_status()

                        for line in response.iter_lines():
                            if not line:
                                continue

                            line_text = line.decode('utf-8')
                            if not line_text.startswith("data: "):
                                continue

                            data = line_text[6:]  # 跳过 "data: " 前缀

                            if data == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data)

                                # 提取文本增量
                                if "choices" in chunk and chunk["choices"]:
                                    choice = chunk["choices"][0]
                                    if "delta" in choice and "content" in choice["delta"]:
                                        text_chunk = choice["delta"]["content"]
                                        accumulated_text += text_chunk
                                        # 打印流式输出
                                        print(text_chunk, end="", flush=True)
                            except json.JSONDecodeError:
                                continue

                # 构建类似于非流式响应的对象
                return {
                    "choices": [
                        {
                            "message": {
                                "content": accumulated_text
                            }
                        }
                    ]
                }

            except requests.exceptions.RequestException as e:
                print(f"Error during streaming OpenAI API request: {e}")
                # 出错时回退到非流式请求
                payload["stream"] = False

        # 非流式请求
        try:
            if "stream" in payload:
                payload.pop("stream")

            response = requests.post(self.api_url, headers=headers,
                                    json=payload,
                                    timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error during OpenAI API request: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Status code: {e.response.status_code}")
                print(f"Raw API Response: {e.response.text[:200]}")
            raise e

    def extract_text_from_response(self, response_data: Dict) -> Tuple[str, str]:
        """从OpenAI API响应中提取文本"""
        try:
            text = response_data["choices"][0]["message"]["content"]
            # OpenAI不支持思考过程
            return text, ""
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error extracting text from OpenAI API response: {e}")
            print(json.dumps(response_data, indent=2)[:500])
            return "", ""

    def check_capabilities(self) -> bool:
        """检查OpenAI模型是否支持流式输出"""
        # OpenAI支持流式输出但不支持思考过程显示
        self.streaming_supported = True
        return True


class KimiProvider(ModelProvider):
    """Kimi API提供商实现"""

    # 支持思考能力的模型列表
    THINKING_MODELS = [
        "kimi-k1.5",
        "kimi-k1.5-preview",
        "kimi-k1.5-long-context",
    ]

    # 所有可用模型列表
    AVAILABLE_MODELS = [
        "kimi-k1.5",
        "kimi-k1.5-preview", 
        "kimi-k1.5-long-context",
        "moonshot-v1-128k",
        "moonshot-v1-32k",
        "moonshot-v1-8k",
    ]

    def __init__(self, api_key: str = None, model_name: str = "kimi-k1.5"):
        super().__init__(api_key, model_name)
        self.api_url = "https://api.moonshot.cn/v1/chat/completions"
        self.supports_thinking = model_name in self.THINKING_MODELS

    def get_api_key(self) -> str:
        """获取Kimi API密钥"""
        if self.api_key:
            return self.api_key

        api_key = os.getenv("KIMI_API_KEY")
        if not api_key:
            print("Error: KIMI_API_KEY environment variable not set.")
            print("Please set the variable, e.g., 'export KIMI_API_KEY=\"your_api_key\"'")
            sys.exit(1)
        self.api_key = api_key
        return api_key

    def build_request_payload(self, system_prompt: str,
                             question_prompt: str,
                             other_prompts: List[str] = None,
                             enable_thinking: bool = True,
                             streaming: bool = True) -> Dict:
        """构建Kimi API请求负载"""
        messages = []

        # 添加系统提示
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        # 添加主要问题
        messages.append({"role": "user", "content": question_prompt})

        # 添加其他提示
        if other_prompts:
            for prompt in other_prompts:
                messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.3 if enable_thinking and self.supports_thinking else 0.1,
        }

        # 为支持思考的模型添加思考配置
        if enable_thinking and self.supports_thinking:
            # kimi-k1.5 系列支持 reasoning_content
            payload["extra_body"] = {
                "enable_thinking": True,
                "thinking_budget": 32768
            }

        # 添加流式输出配置
        if streaming and self.streaming_supported:
            payload["stream"] = True

        return payload

    def send_api_request(self, payload: Dict, streaming: bool = False,
                         show_thinking: bool = False) -> Dict:
        """发送Kimi API请求并返回响应"""
        api_key = self.get_api_key()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # 流式请求仅在支持时启用
        use_streaming = streaming and self.streaming_supported

        if use_streaming:
            payload["stream"] = True

            try:
                accumulated_text = ""
                thinking_text = ""
                thinking_active = False

                with requests.Session() as session:
                    with session.post(self.api_url, headers=headers,
                                     json=payload,
                                     stream=True) as response:
                        response.raise_for_status()

                        for line in response.iter_lines():
                            if not line:
                                continue

                            line_text = line.decode('utf-8')
                            if not line_text.startswith("data: "):
                                continue

                            data = line_text[6:]  # 跳过 "data: " 前缀

                            if data == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data)

                                # 处理思考内容 (kimi-k1.5 系列)
                                if "choices" in chunk and chunk["choices"]:
                                    choice = chunk["choices"][0]
                                    delta = choice.get("delta", {})
                                    
                                    # 提取思考过程
                                    if "reasoning_content" in delta and delta["reasoning_content"]:
                                        if show_thinking:
                                            reasoning_chunk = delta["reasoning_content"]
                                            thinking_text += reasoning_chunk
                                            if not thinking_active:
                                                thinking_active = True
                                                print("\n--- Thinking Process ---")
                                            print(reasoning_chunk, end="", flush=True)
                                    
                                    # 思考结束，打印分隔线
                                    if thinking_active and "content" in delta and delta["content"]:
                                        thinking_active = False
                                        print("\n--- End of Thinking ---\n")
                                    
                                    # 提取文本增量
                                    if "content" in delta and delta["content"]:
                                        text_chunk = delta["content"]
                                        accumulated_text += text_chunk
                                        # 打印流式输出
                                        print(text_chunk, end="", flush=True)
                            except json.JSONDecodeError:
                                continue

                # 构建类似于非流式响应的对象
                result = {
                    "choices": [
                        {
                            "message": {
                                "content": accumulated_text
                            }
                        }
                    ]
                }
                if thinking_text:
                    result["thinking_process"] = thinking_text
                return result

            except requests.exceptions.RequestException as e:
                print(f"Error during streaming Kimi API request: {e}")
                # 出错时回退到非流式请求
                payload["stream"] = False

        # 非流式请求
        try:
            if "stream" in payload:
                payload.pop("stream")

            response = requests.post(self.api_url, headers=headers,
                                    json=payload,
                                    timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            result = response.json()
            
            # 对于支持思考的模型，尝试提取思考过程
            if self.supports_thinking and "choices" in result and result["choices"]:
                message = result["choices"][0].get("message", {})
                if "reasoning_content" in message:
                    result["thinking_process"] = message["reasoning_content"]
            
            return result
        except requests.exceptions.RequestException as e:
            print(f"Error during Kimi API request: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Status code: {e.response.status_code}")
                print(f"Raw API Response: {e.response.text[:200]}")
            raise e

    def extract_text_from_response(self, response_data: Dict) -> Tuple[str, str]:
        """从Kimi API响应中提取文本和思考过程"""
        try:
            text = response_data["choices"][0]["message"]["content"]
            # 提取思考过程（如果有）
            thinking = response_data.get("thinking_process", "")
            if not thinking and "choices" in response_data and response_data["choices"]:
                message = response_data["choices"][0].get("message", {})
                thinking = message.get("reasoning_content", "")
            return text, thinking
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error extracting text from Kimi API response: {e}")
            print(json.dumps(response_data, indent=2)[:500])
            return "", ""

    def check_capabilities(self) -> bool:
        """检查Kimi模型是否支持流式输出"""
        # Kimi支持流式输出
        self.streaming_supported = True
        return True


# 工厂函数来创建适当的提供商实例
def create_provider(provider_name: str = None, api_key: str = None, model_name: str = None) -> ModelProvider:
    """
    创建模型提供商实例

    参数:
        provider_name: 提供商名称（'gemini'、'openai'、'kimi'）
        api_key: API密钥（可选）
        model_name: 模型名称（可选）

    返回:
        ModelProvider的具体实现实例
    """
    if not provider_name:
        # 检查环境变量获取默认提供商
        provider_name = os.getenv("DEFAULT_MODEL_PROVIDER", "gemini").lower()

    # 规范化提供商名称
    provider_name = provider_name.lower()

    if provider_name in ["gemini", "google", "gemini-api"]:
        return GeminiProvider(api_key, model_name or "gemini-2.5-pro")
    elif provider_name in ["openai", "gpt", "openai-api"]:
        return OpenAIProvider(api_key, model_name or "gpt-4o")
    elif provider_name in ["kimi", "moonshot", "kimi-api"]:
        # 默认使用支持思考的 kimi-k1.5 模型
        return KimiProvider(api_key, model_name or "kimi-k1.5")
    else:
        print(f"Unknown provider: {provider_name}, using Gemini as default")
        return GeminiProvider(api_key, model_name or "gemini-2.5-pro")


# 获取可用的提供商列表
def get_available_providers() -> List[str]:
    """获取可用的模型提供商列表"""
    providers = []

    if os.getenv("GOOGLE_API_KEY"):
        providers.append("gemini")

    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")

    if os.getenv("KIMI_API_KEY"):
        providers.append("kimi")

    return providers