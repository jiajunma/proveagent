"""
IMO Agent Package

This package provides tools for solving IMO-level mathematical problems
using various LLM backends.
"""

# Export main components
from .proof_engine import ProofEngine, ProofState, LLMClient, LLMResponse
from .llm_clients import (
    create_client,
    GeminiClient,
    OpenAIClient,
    KimiClient,
    ModelProviderClient,
)
from . import prompts
from . import utils

__all__ = [
    # Core components
    'ProofEngine',
    'ProofState',
    'LLMClient',
    'LLMResponse',
    # Clients
    'create_client',
    'GeminiClient',
    'OpenAIClient',
    'KimiClient',
    'ModelProviderClient',
    # Modules
    'prompts',
    'utils',
]
