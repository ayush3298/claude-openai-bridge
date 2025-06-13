"""
Custom exceptions for Claude OpenAI API Bridge
"""

class ClaudeError(Exception):
    """Base exception for Claude-related errors"""
    pass

class ClaudeNotFoundError(ClaudeError):
    """Claude command not found"""
    pass

class ClaudeTimeoutError(ClaudeError):
    """Claude command timed out"""
    pass

class ClaudeProcessError(ClaudeError):
    """Claude process failed"""
    pass

class ClaudeRateLimitError(ClaudeError):
    """Claude rate limit exceeded"""
    pass