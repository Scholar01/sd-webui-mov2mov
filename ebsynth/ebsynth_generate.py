import cv2
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


class EbsynthGenerate:
    def __init__(self, keyframes: list[Keyframe], frames: list[np.ndarray], fps: int):
        self.keyframes = keyframes
        self.frames = frames
        self.fps = fps
        self.sequences = []
        self.setup_sequences()

    def setup_sequences(self):
        self.sequences.clear()
        all_frames = len(self.frames)
        left_frame = 1
        for i, keyframe in enumerate(self.keyframes):
            right_frame = self.keyframes[i + 1].num if i + 1 < len(self.keyframes) else all_frames
            frames = {}
            for frame_num in range(left_frame, right_frame + 1):
                frames[frame_num] = self.frames[frame_num - 1]

            sequence = Sequence(left_frame, keyframe, right_frame, frames)
            self.sequences.append(sequence)
            left_frame = keyframe.num
        return self.sequences

    def get_tasks(self, weight: float = 4.0) -> list[EbSynthTask]:
        tasks = []
        for i, sequence in enumerate(self.sequences):
            frames = sequence.frames.items()
            source = sequence.frames[sequence.start]
            style = sequence.keyframe.image
            for frame_num, frame in frames:
                target = frame
                task = EbSynthTask(style, source, target, frame_num, sequence.keyframe.num, weight)
                tasks.append(task)
        return tasks

    def append_generate_frames(self, key_frames_num, frame_num, generate_frames):
        """

        Args:
            key_frames_num:  用于定位sequence
            frame_num: key
            generate_frames: value

        Returns:

        """
        for i, sequence in enumerate(self.sequences):
            if sequence.keyframe.num == key_frames_num:
                sequence.generate_frames[frame_num] = generate_frames
                break
        else:
            raise ValueError(f'not found key frame num {key_frames_num}')

    def merge_generate_frames(self):

        temp = []
        for i, sequence in enumerate(self.sequences):
            # 获取下一序列
            next_sequence = self.sequences[i + 1] if i + 1 < len(self.sequences) else None
            if next_sequence:
                # 取两个序列的结果交集
                intersection = set(sequence.generate_frames.keys()) & set(next_sequence.generate_frames.keys())
                print(intersection)
                # 取当前序列的交集结果
                current_frames = [sequence.generate_frames[k] for k in intersection]
                # 取下一序列的交集结果
                next_frames = [next_sequence.generate_frames[k] for k in intersection]
                assert len(current_frames) == len(next_frames)

                temp1 = []
                # opencv 合并相同序列的两张图
                for j in range(len(current_frames)):
                    # 按照权重合并
                    weight = 1.0  # j / len(current_frames)
                    current_frame = current_frames[j]
                    next_frame = next_frames[j]
                    result = cv2.addWeighted(current_frame, weight, next_frame, 1 - weight, 0)
                    temp1.append(result)
                temp.append(temp1)
            else:
                # 取当前序列 keyframe之后的所有帧,包括keyframe
                current_frames = [sequence.generate_frames[k] for k in sequence.generate_frames.keys() if
                                  k >= sequence.keyframe.num]
                temp.append(current_frames)
        # 打印当前共多少帧
        i = 0
        for t in temp:
            i += len(t)
        print(i)
        # 合并当前数组的最后一帧和下一个数组的第一帧
        result = []
        result.extend(temp[0][:-1])
        for i in range(len(temp) - 1):
            current_frames = temp[i]
            next_frames = temp[i + 1]
            weight = 0.5
            merge = cv2.addWeighted(current_frames[-1], weight, next_frames[0], 1 - weight, 0)
            result.append(merge)
            result.extend(next_frames[1:-1])
        result.extend(temp[-1][-1:])
        return result
