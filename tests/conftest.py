"""Shared pytest configuration and fixtures."""

import sys
import os

# Ensure module paths are available
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "webservice")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mcp-server")))
