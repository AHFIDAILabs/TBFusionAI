"""
Clean Up Script for TBFusionAI.

This script recursively deletes all:
1. __pycache__ directories
2. .pyc files (compiled python files)
3. .pyo files (optimized python files)
4. .pyd files (python dynamic libraries, usually temp)
5. .pytest_cache directories (if they exist)

Usage:
    python clean_cache.py
"""

import os
import shutil
from pathlib import Path


def clean_project(root_dir: str = "."):
    """
    Traverses the directory tree starting from root_dir and removes
    python cache artifacts.
    """
    root_path = Path(root_dir).resolve()
    print(f"🧹 Starting cleanup in: {root_path}")

    deleted_dirs = 0
    deleted_files = 0

    # Walk through the directory tree
    for current_dir, dirs, files in os.walk(root_path):
        current_path = Path(current_dir)

        # 1. Remove __pycache__ and .pytest_cache directories
        # We iterate over a copy of 'dirs' so we can modify the original list safely
        for d in dirs[:]:
            if d == "__pycache__" or d == ".pytest_cache":
                dir_to_remove = current_path / d
                try:
                    shutil.rmtree(dir_to_remove)
                    print(f"   Deleted directory: {dir_to_remove}")
                    deleted_dirs += 1
                    # Remove from search list so we don't walk into it
                    dirs.remove(d)
                except Exception as e:
                    print(f"   ❌ Error deleting {dir_to_remove}: {e}")

        # 2. Remove .pyc, .pyo, .pyd files
        for f in files:
            if f.endswith((".pyc", ".pyo", ".pyd")):
                file_to_remove = current_path / f
                try:
                    file_to_remove.unlink()
                    # print(f"   Deleted file: {file_to_remove}") # Uncomment for verbose
                    deleted_files += 1
                except Exception as e:
                    print(f"   ❌ Error deleting {file_to_remove}: {e}")

    print("-" * 50)
    print("✨ Cleanup Complete!")
    print(f"   Directories Removed: {deleted_dirs}")
    print(f"   Files Removed:       {deleted_files}")
    print("-" * 50)


if __name__ == "__main__":
    clean_project()
