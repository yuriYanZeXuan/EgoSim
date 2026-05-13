"""Minimal DiffSynth facade used by the EgoHOI baseline.

The EgoHOI release vendors pipeline/data code here, while the shared Wan model
definitions live in the project-level `EgoSim/diffsynth/models` package.  Extend
this package path so relative imports such as `diffsynth.models...` resolve
there without importing unrelated, incomplete pipeline modules.
"""
from pathlib import Path
import importlib.util
import sys
import types

_VENDORED_DIFFSYNTH = Path(__file__).resolve().parent
_PROJECT_DIFFSYNTH = Path(__file__).resolve().parents[3] / "diffsynth"


def _register_namespace(package_name: str, package_path: Path) -> types.ModuleType:
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    return package


if _PROJECT_DIFFSYNTH.exists():
    project_diffsynth = str(_PROJECT_DIFFSYNTH)
    if project_diffsynth not in __path__:
        __path__.append(project_diffsynth)

    project_configs = _PROJECT_DIFFSYNTH / "configs"
    if project_configs.exists():
        _register_namespace("diffsynth.configs", project_configs)

# Avoid executing vendored package __init__.py files that import unrelated
# SD/Flux modules absent from this EgoHOI release.
_register_namespace("diffsynth.pipelines", _VENDORED_DIFFSYNTH / "pipelines")
_register_namespace("diffsynth.schedulers", _VENDORED_DIFFSYNTH / "schedulers")
_register_namespace("diffsynth.distributed", _VENDORED_DIFFSYNTH / "distributed")

prompters_pkg = _register_namespace("diffsynth.prompters", _VENDORED_DIFFSYNTH / "prompters")
_wan_prompter_path = _VENDORED_DIFFSYNTH / "prompters" / "wan_prompter.py"
_wan_prompter_spec = importlib.util.spec_from_file_location("diffsynth.prompters.wan_prompter", _wan_prompter_path)
_wan_prompter_module = importlib.util.module_from_spec(_wan_prompter_spec)
sys.modules["diffsynth.prompters.wan_prompter"] = _wan_prompter_module
_wan_prompter_spec.loader.exec_module(_wan_prompter_module)
prompters_pkg.WanPrompter = _wan_prompter_module.WanPrompter

from .data import VideoData, save_frames, save_video
from .models import ModelManager, load_state_dict
from .models.utils import load_state_dict_from_folder
from .pipelines.wan_video import (
    WanRepalceAnyoneVideoPipeline,
    WanUniAnimateLongVideoPipeline,
    WanUniAnimateVideoPipeline,
    WanVideoPipeline,
)

__all__ = [
    "ModelManager",
    "VideoData",
    "WanRepalceAnyoneVideoPipeline",
    "WanUniAnimateLongVideoPipeline",
    "WanUniAnimateVideoPipeline",
    "WanVideoPipeline",
    "load_state_dict",
    "load_state_dict_from_folder",
    "save_frames",
    "save_video",
]
