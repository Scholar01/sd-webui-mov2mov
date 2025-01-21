import cv2
import numpy as np
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
        """
        初始化序列,在这个阶段,frame_num对应的帧就已经处理好了,在后面使用不需要再处理frame-1了
        """
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
            source = sequence.frames[sequence.keyframe.num]
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
        for sequence in self.sequences:
            if sequence.keyframe.num == key_frames_num:
                sequence.generate_frames[frame_num] = generate_frames
                break
        else:
            raise ValueError(f'not found key frame num {key_frames_num}')

    def merge_sequences(self):
        # 存储合并后的结果
        merged_frames = []
        border = 1
        for i in range(len(self.sequences)):
            current_seq = self.sequences[i]
            next_seq = self.sequences[i + 1] if i + 1 < len(self.sequences) else None

            # 如果存在下一个序列
            if next_seq:
                # 获取两个序列的帧交集
                common_frames_nums = set(current_seq.frames.keys()).intersection(
                    set(range(next_seq.start + border, next_seq.end)) if i > 0 else set(
                        range(next_seq.start, next_seq.end)))

                for j, frame_num in enumerate(common_frames_nums):
                    # 从两个序列中获取帧并合并
                    frame1 = current_seq.generate_frames[frame_num]
                    frame2 = next_seq.generate_frames[frame_num]

                    weight = float(j) / float(len(common_frames_nums))
                    merged_frame = cv2.addWeighted(frame1, 1 - weight, frame2, weight, 0)
                    merged_frames.append((frame_num, merged_frame))

            # 如果没有下一个序列
            else:
                # 添加与前一序列的差集帧到结果中
                if i > 0:
                    prev_seq = self.sequences[i - 1]
                    difference_frames_nums = set(current_seq.frames.keys()) - set(prev_seq.frames.keys())
                else:
                    difference_frames_nums = set(current_seq.frames.keys())

                for frame_num in difference_frames_nums:
                    merged_frames.append((frame_num, current_seq.generate_frames[frame_num]))

        # group_merged_frames = groupby(lambda x: x[0], merged_frames)
        # merged_frames.clear()
        # # 取出value长度大于1的元素
        # for key, value in group_merged_frames.items():
        #     if len(value) > 1:
        #         # 将value中的所有元素合并
        #         merged_frame = value[0][1]
        #         for i in range(1, len(value)):
        #             merged_frame = cv2.addWeighted(merged_frame, weight, value[i][1], 1 - weight, 0)
        #         merged_frames.append((key, merged_frame))
        #     else:
        #         merged_frames.append((key, value[0][1]))
        result = []
        for i, frame in sorted(merged_frames, key=lambda x: x[0]):
            result.append(frame)

        return result
