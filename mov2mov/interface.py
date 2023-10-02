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
