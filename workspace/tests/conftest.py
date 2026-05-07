"""Pytest configuration for the Coding Agent test suite."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

pytest_plugins = ("pytest_asyncio",)
