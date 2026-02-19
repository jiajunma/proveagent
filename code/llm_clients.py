"""
LLM Client implementations using model_providers.

This module provides concrete implementations of the LLMClient interface
for different API providers.
"""

import os
import sys
from typing import Optional, List

# Import the abstract interface and utilities
try:
    from .proof_engine import LLMClient, LLMResponse
    from .utils import quota_manager
    from . import model_providers
except ImportError:
    # Allow running as standalone script
    from proof_engine import LLMClient, LLMResponse
    from utils import quota_manager
    import model_providers


class ModelProviderClient(LLMClient):
    """
    LLMClient implementation using model_providers.
    
    This adapter wraps any ModelProvider from model_providers.py
    to implement the LLMClient interface.
    """
    
    def __init__(self, provider: model_providers.ModelProvider):
        """
        Initialize with a model provider.
        
        Args:
            provider: The model provider to wrap
        """
        self.provider = provider
    
    def generate(self,
                 system_prompt: str,
                 user_prompt: str,
                 other_prompts: Optional[List[str]] = None,
                 enable_thinking: bool = True,
                 streaming: bool = True) -> LLMResponse:
        """
        Generate a response using the underlying provider.
        """
        # Build request payload
        payload = self.provider.build_request_payload(
            system_prompt=system_prompt,
            question_prompt=user_prompt,
            other_prompts=other_prompts,
            enable_thinking=enable_thinking,
            streaming=streaming
        )
        
        # Send request
        response_data = self.provider.send_api_request(
            payload,
            streaming=streaming,
            show_thinking=enable_thinking
        )
        
        # Check for quota errors in response
        if isinstance(response_data, dict) and 'error' in response_data:
            error_msg = response_data.get('error', {}).get('message', '')
            if quota_manager.check_response_for_quota_error(error_msg):
                quota_manager.quota_exceeded = True
                raise Exception(f"Quota exceeded: {error_msg}")
        
        # Extract text and thinking
        text, thinking = self.provider.extract_text_from_response(response_data)
        
        return LLMResponse(
            text=text,
            thinking=thinking,
            raw_response=response_data
        )
    
    def check_quota(self) -> bool:
        """Check if API quota is available."""
        return not quota_manager.quota_exceeded
    
    @property
    def supports_thinking(self) -> bool:
        """Check if provider supports thinking."""
        # Check if it's a thinking-capable model
        if hasattr(self.provider, 'supports_thinking'):
            return self.provider.supports_thinking
        # For Gemini, check model name
        if isinstance(self.provider, model_providers.GeminiProvider):
            return '2.5' in self.provider.model_name
        return False
    
    @property
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        return getattr(self.provider, 'streaming_supported', False)
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.provider.model_name
    
    def set_model(self, model_name: str):
        """Change the model."""
        # Create new provider with same type but different model
        provider_name = self.provider.get_name().lower()
        new_provider = model_providers.create_provider(provider_name, model_name=model_name)
        self.provider = new_provider


class GeminiClient(ModelProviderClient):
    """Convenience client for Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-pro"):
        """
        Initialize Gemini client.
        
        Args:
            api_key: API key (uses GOOGLE_API_KEY env var if not provided)
            model_name: Model name to use
        """
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                print("Error: GOOGLE_API_KEY environment variable not set.")
                sys.exit(1)
        
        provider = model_providers.GeminiProvider(api_key, model_name)
        super().__init__(provider)


class OpenAIClient(ModelProviderClient):
    """Convenience client for OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o"):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: API key (uses OPENAI_API_KEY env var if not provided)
            model_name: Model name to use
        """
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Error: OPENAI_API_KEY environment variable not set.")
                sys.exit(1)
        
        provider = model_providers.OpenAIProvider(api_key, model_name)
        super().__init__(provider)


class KimiClient(ModelProviderClient):
    """Convenience client for Kimi API."""
    
    # Thinking-capable models
    THINKING_MODELS = [
        "kimi-k1.5",
        "kimi-k1.5-preview",
        "kimi-k1.5-long-context",
    ]
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "kimi-k1.5"):
        """
        Initialize Kimi client.
        
        Args:
            api_key: API key (uses KIMI_API_KEY env var if not provided)
            model_name: Model name to use (default: kimi-k1.5 with thinking)
        """
        if not api_key:
            api_key = os.getenv("KIMI_API_KEY")
            if not api_key:
                print("Error: KIMI_API_KEY environment variable not set.")
                sys.exit(1)
        
        provider = model_providers.KimiProvider(api_key, model_name)
        super().__init__(provider)
    
    @property
    def supports_thinking(self) -> bool:
        """Check if current model supports thinking."""
        return self.model_name in self.THINKING_MODELS


def create_client(provider_name: str, 
                  model_name: Optional[str] = None,
                  api_key: Optional[str] = None) -> LLMClient:
    """
    Factory function to create an LLM client.
    
    Args:
        provider_name: Provider name ('gemini', 'openai', 'kimi')
        model_name: Optional model name
        api_key: Optional API key
        
    Returns:
        Configured LLMClient instance
    """
    provider_name = provider_name.lower()
    
    if provider_name in ["gemini", "google"]:
        model = model_name or "gemini-2.5-pro"
        return GeminiClient(api_key, model)
    
    elif provider_name in ["openai", "gpt"]:
        model = model_name or "gpt-4o"
        return OpenAIClient(api_key, model)
    
    elif provider_name in ["kimi", "moonshot"]:
        model = model_name or "kimi-k1.5"
        return KimiClient(api_key, model)
    
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


# ============================================================================
# Legacy compatibility
# ============================================================================

# Global client instance for backwards compatibility
_default_client: Optional[LLMClient] = None


def set_default_client(client: LLMClient):
    """Set the default global client."""
    global _default_client
    _default_client = client


def get_default_client() -> Optional[LLMClient]:
    """Get the default global client."""
    return _default_client
