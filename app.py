#!/usr/bin/env python3
"""
Claude OpenAI Bridge - Main Application Entry Point
"""

import uvicorn
from claude_openai.api import create_app

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=5000)