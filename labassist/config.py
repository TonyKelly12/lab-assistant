"""Config loading (YAML) with a clear error when a file is missing."""
import os
import sys

import yaml


def load(path):
    if not os.path.exists(path):
        sys.exit(f"Config file not found: {path}\nCopy the matching file from config/*.example.yaml and edit it.")
    with open(path) as f:
        return yaml.safe_load(f) or {}
