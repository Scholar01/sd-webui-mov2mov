from dataclasses import dataclass, field

import numpy as np


@dataclass
class Keyframe:
    """
    关键帧

    """
    num: int
    image: np.ndarray = field(repr=False)
    prompt: str = field(repr=False)

    # 用于多帧渲染存储自身位置
    col: int = field(default=0)
    row: int = field(default=0)


@dataclass
class Sequence:
    """
    序列
    """

    start: int
    keyframe: Keyframe
    end: int
    # 当前序列的所有帧
    frames: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    # 序列生成的所有帧
    generate_frames: dict[int, np.ndarray] = field(default_factory=dict, repr=False)


@dataclass
class EbSynthTask:
    style: np.ndarray = field(repr=False)
    source: np.ndarray = field(repr=False)
    target: np.ndarray = field(repr=False)
    frame_num: int
    key_frame_num: int
    weight: float = field(default=1.0, repr=False)