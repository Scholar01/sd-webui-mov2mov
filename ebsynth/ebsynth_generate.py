import numpy as np
from PIL.Image import Image
from dataclasses import dataclass, field


@dataclass
class Keyframe:
    num: int
    image: np.ndarray = field(repr=False)
    prompt: str = field(repr=False)


@dataclass
class Sequence:
    start: int
    keyframe: Keyframe
    end: int
    # todo dict
    frames: list[np.ndarray] = field(default_factory=list, repr=False)
    generate_frames: list[np.ndarray] = field(default_factory=list, repr=False)


class EbsynthGenerate:
    def __init__(self, keyframes: list[Keyframe], frames: list[np.ndarray], fps: int):
        self.keyframes = keyframes
        self.frames = frames
        self.fps = fps
        self.sequences = []

    def setup_sequences(self):
        all_frames = len(self.frames)
        left_frame = 1
        for i, keyframe in enumerate(self.keyframes):
            right_frame = self.keyframes[i + 1].num if i + 1 < len(self.keyframes) else all_frames
            sequence = Sequence(left_frame, keyframe, right_frame, self.frames[left_frame - 1:right_frame - 1])
            self.sequences.append(sequence)
            left_frame = keyframe.num
        return self.sequences

    # def merge_movie(self):
    #     # todo : test for frames
    #     for i, sequence in enumerate(self.sequences):
    #         next_sequence = self.sequences[i + 1] if i + 1 < len(self.sequences) else None
    #         sequence.generate_frames = sequence.frames
    #         if next_sequence:
    #             sequence.generate_frames.append(next_sequence.frames[0])
    #
    #     for i, sequence in enumerate(self.sequences):
    #         # 获取下一关键帧与当前关键帧之间需要合并的帧
    #         next_sequence = self.sequences[i + 1] if i + 1 < len(self.sequences) else None
    #         if next_sequence:
    #             print(sequence)
    #             print(next_sequence)
    #             # 取当前关键帧与下一关键帧的交集
    #             intersection = set(range(sequence.keyframe.num, next_sequence.keyframe.num + 1)) & set(
    #                 range(sequence.start, next_sequence.end + 1))
    #
    #             print(f'交集：{intersection}')
    #
    #             current = [keyframe - sequence.keyframe.num for keyframe in intersection]
    #             next = [next_sequence.keyframe.num - keyframe for keyframe in
    #                     reversed(list(intersection))]
    #             print(f'current: {current}')
    #             print(f'next: {next}')
