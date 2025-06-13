"""
Configuration management for Claude OpenAI API Bridge
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the Claude OpenAI API Bridge"""
    
    def __init__(self):
        # Server configuration
        self.port = int(os.getenv("PORT", 8000))
        self.host = os.getenv("HOST", "0.0.0.0")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Claude CLI configuration
        self.claude_command = os.getenv("CLAUDE_COMMAND", "claude")
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", 60))
        self.max_retries = int(os.getenv("MAX_RETRIES", 3))
        self.retry_delay = float(os.getenv("RETRY_DELAY", 1.0))
        self.max_prompt_length = int(os.getenv("MAX_PROMPT_LENGTH", 100000))
        
        # History management configuration
        self.enable_history = os.getenv("ENABLE_HISTORY", "true").lower() == "true"
        self.history_storage = os.getenv("HISTORY_STORAGE", "memory")  # memory, file, redis
        self.history_dir = os.getenv("HISTORY_DIR", "./conversations")
        self.max_history_messages = int(os.getenv("MAX_HISTORY_MESSAGES", 100))
        self.history_ttl_hours = int(os.getenv("HISTORY_TTL_HOURS", 24))
        self.max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", 32000))
        
        # Security configuration - restrict Claude to web-only tools
        self.restricted_mode = os.getenv("RESTRICTED_MODE", "false").lower() == "true"
        self.allowed_tools = os.getenv("ALLOWED_TOOLS", "WebSearch,WebFetch,WebView").split(",")
        self.disallowed_tools = os.getenv("DISALLOWED_TOOLS", "Bash,Edit,Write,Read,LS,Grep,Glob,NotebookEdit,mcp__*").split(",")
        self.dangerously_skip_permissions = os.getenv("DANGEROUSLY_SKIP_PERMISSIONS", "false").lower() == "true"
        self.mcp_config = os.getenv("MCP_CONFIG", None)

# Global configuration instance
config = Config()