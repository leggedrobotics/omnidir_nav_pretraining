"""
Python module serving as a project/extension template.
"""

import os

# Register Gym environments.
from .env_config import *
from .runner import *

EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

DATA_DIR = os.path.join(EXT_DIR, "data")