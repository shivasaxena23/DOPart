from __future__ import annotations

import runpy
import sys
from pathlib import Path


def run_project_script(script_name: str):
  repo_root = Path(__file__).resolve().parents[1]
  project_dir = repo_root / "project"
  for path in (repo_root, project_dir):
    path_str = str(path)
    if path_str not in sys.path:
      sys.path.insert(0, path_str)
  runpy.run_path(str(project_dir / script_name), run_name="__main__")
