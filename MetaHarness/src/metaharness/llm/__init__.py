"""Clients LLM texte et parsing de réponses structurées en texte."""

from .chat import (
    LLMError,
    LLMHTTPError,
    LLMProtocolError,
    ConversationUnavailableError,
    ConversationContinuationClient,
    OpenAIChatTextClient,
    TextLLMResult,
)
from .wire import (
    AmbiguousFieldError,
    ParsedTextDocument,
    WireParseError,
    parse_labeled_document,
)
from ..models import LLMEndpointConfig

__all__ = [
    "AmbiguousFieldError",
    "LLMError",
    "LLMEndpointConfig",
    "LLMHTTPError",
    "LLMProtocolError",
    "ConversationUnavailableError",
    "ConversationContinuationClient",
    "OpenAIChatTextClient",
    "ParsedTextDocument",
    "TextLLMResult",
    "WireParseError",
    "parse_labeled_document",
]
