import numpy as np
from .interface import Keyframe, Sequence


def generate_sequences(frames: list[np.ndarray], keyframes: [Keyframe]) -> list[Sequence]:
    """
    根据关键帧生成序列

    Args:
        frames:
        keyframes:

    Returns:

    """
    sequences = []
    all_frames = len(frames)
    left_frame = 1
    for i, keyframe in enumerate(keyframes):
        right_frame = keyframes[i + 1].num if i + 1 < len(keyframes) else all_frames

        frames = {}
        for frame_num in range(left_frame, right_frame + 1):
            frames[frame_num] = frames[frame_num - 1]
        sequence = Sequence(left_frame, keyframe, right_frame, frames)
        sequences.append(sequence)
        left_frame = keyframe.num
    return sequences
