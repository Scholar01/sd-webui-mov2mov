from dataclasses import dataclass, field

import PIL
import numpy as np
import pandas
import modules

from modules.processing import StableDiffusionProcessingImg2Img
from PIL.Image import Image

from .interface import Keyframe


@dataclass(repr=False)
class StableDiffusionProcessingMov2Mov(StableDiffusionProcessingImg2Img):
    fps: int = 0
    frames: list[np.ndarray] = field(default_factory=list, repr=False)

    # beta
    beta: bool = False
    data_frames: pandas.DataFrame = field(default_factory=pandas.DataFrame, repr=False)
    keyframes: list[Keyframe] = field(default_factory=list, repr=False)
    ebsynth_weight: float = 4.0
    keyframe_mode: int = 0

    scripts_custom_args: tuple = field(default_factory=tuple, repr=False)
    scripts_preprocess1_args: tuple = field(default_factory=tuple, repr=False)
    scripts_preprocess2_args: tuple = field(default_factory=tuple, repr=False)

    def setup_script_args(self, p1, p2, scripts, args):
        """
        初始化脚本参数

        Returns:

        """

        args = list(args)
        preprocess1_inputs_len = int(p1)
        preprocess2_inputs_len = int(p2)

        preprocess1_inputs = args[:preprocess1_inputs_len]
        preprocess2_inputs = args[preprocess1_inputs_len:preprocess1_inputs_len + preprocess2_inputs_len]
        args = args[preprocess1_inputs_len + preprocess2_inputs_len:]

        # fix seed
        for script in scripts.scripts:
            if script.section == 'seed':
                for i in range(script.args_from, script.args_to):
                    preprocess1_inputs.insert(i, args[i])
                    preprocess2_inputs.insert(i, args[i])

        args= tuple(args)
        preprocess1_inputs = tuple(preprocess1_inputs)
        preprocess2_inputs = tuple(preprocess2_inputs)

        self.scripts_custom_args = args
        self.scripts_preprocess1_args = preprocess1_inputs
        self.scripts_preprocess2_args = preprocess2_inputs

        self.scripts = scripts
        self.script_args = args

    def setup_frames(self):
        """
        初始化帧,并修改尺寸

        Returns:

        """

        for i, frame in enumerate(self.frames):
            image = PIL.Image.fromarray(frame)
            image = modules.images.resize_image(self.resize_mode, image, self.width, self.height)
            assert image.width == self.width and image.height == self.height
            self.frames[i] = np.asarray(image)

    def setup_keyframes(self):
        """
        初始化关键帧,并修改尺寸

        Returns:

        """

        if self.beta and self.data_frames:
            self.keyframes.clear()
            default_prompt = self.prompt
            for i, row in self.data_frames.iterrows():
                prompt = default_prompt + row['prompt']
                frame = self.frames[row['frame'] - 1]
                keyframe = Keyframe(row['frame'], frame, prompt)
                self.keyframes.append(keyframe)



