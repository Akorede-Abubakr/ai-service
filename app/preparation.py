"""Render inference engine; model preparation stays off the 8 GB service."""
from .chat import ChatEngine



class RenderChatEngine(ChatEngine):
    """Load the prebuilt, hash-pinned artifact and serve CPU inference."""
