from dataclasses import dataclass, field

import numpy as np
import pandas

from modules.processing import StableDiffusionProcessingImg2Img


@dataclass(repr=False)
class StableDiffusionProcessingMov2Mov(StableDiffusionProcessingImg2Img):
    fps: int = 0
    frames: list[np.ndarray] = field(default_factory=list, repr=False)
    keyframes: list[np.ndarray] = field(default_factory=list, repr=False)

    # beta
    beta: bool = False
    data_frames: pandas.DataFrame = field(default_factory=pandas.DataFrame, repr=False)
    ebsynth_weight: float = 4.0
    keyframe_mode: int = 0

    scripts_custom_args: dict = field(default_factory=dict, repr=False)
    scripts_preprocess1_args: dict = field(default_factory=dict, repr=False)
    scripts_preprocess2_args: dict = field(default_factory=dict, repr=False)
