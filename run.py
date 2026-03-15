"""
run.py — Start the DevOracle API server.
Usage: python run.py
"""
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from config.settings import settings

if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════╗
║           DevOracle API v0.2.0           ║
╠══════════════════════════════════════════╣
║  Docs:    http://localhost:{settings.api_port}/docs      ║
║  Health:  http://localhost:{settings.api_port}/health    ║
║  Status:  http://localhost:{settings.api_port}/status    ║
╚══════════════════════════════════════════╝
    """)

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=settings.api_port,
        reload=True,
        log_level="warning",
    )
