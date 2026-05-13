from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EgoSample:
    """Unified sample schema across egodex, egovid, agibot and continuous_generation datasets."""
    video_id: str
    output_id: str
    prompt: str
    dataset: str                          # "egodex" | "egovid" | "agibot" | "continuous_generation"
    ego_prior_video: Optional[str] = None # relative path to ego prior .mp4
    first_frame: Optional[str] = None     # relative path to first frame image
    hand_keypoint_video: Optional[str] = None
