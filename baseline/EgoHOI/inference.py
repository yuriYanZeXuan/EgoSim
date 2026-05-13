#!/usr/bin/env python
"""
EgoHOI Wan2.1 zero-shot baseline inference wrapper.

Re-exports the shared baseline inference entrypoint from
baseline/Wan2.1-Fun-14B-InP/inference.py so that EgoHOI can
produce comparable first-frame + prompt results.
"""
import importlib.util
import sys
from pathlib import Path

# Project root (EgoSim/)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Directory name contains dots, so we use importlib instead of plain import.
_CORE_PATH = _PROJECT_ROOT / "baseline" / "Wan2.1-Fun-14B-InP" / "inference.py"
_spec = importlib.util.spec_from_file_location("_wan_baseline_inference", _CORE_PATH)
_wan_baseline_inference = importlib.util.module_from_spec(_spec)
sys.modules["_wan_baseline_inference"] = _wan_baseline_inference
_spec.loader.exec_module(_wan_baseline_inference)

main = _wan_baseline_inference.main

if __name__ == "__main__":
    main()
