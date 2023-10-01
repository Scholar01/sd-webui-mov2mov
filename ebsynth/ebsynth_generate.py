import PIL

import cv2
import numpy as np
from dataclasses import dataclass, field
from PIL.Image import Image


@dataclass
class Keyframe:
    num: int
    image: np.ndarray = field(repr=False)
    prompt: str = field(repr=False)
    col: int = field(default=0)
    row: int = field(default=0)


@dataclass
class KeyframeGuide:
    """
    用于生成keyframe的指导图
    """
    width: int
    height: int
    original_width: int
    original_height: int
    rows: int
    cols: int
    image: np.ndarray = field(repr=False)


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

    def preprocess(self, resize_mode, width, height):
        import modules
        for keyframe in self.keyframes:
            image = PIL.Image.fromarray(keyframe.image)
            image = modules.images.resize_image(resize_mode, image, width, height)
            assert image.width == width and image.height == height
            keyframe.image = np.asarray(image)

        for i, frame in enumerate(self.frames):
            image = PIL.Image.fromarray(frame)
            image = modules.images.resize_image(resize_mode, image, width, height)
            assert image.width == width and image.height == height
            self.frames[i] = np.asarray(image)
        return self

    def setup_sequences(self):
        """
        初始化序列,在这个阶段,frame_num对应的帧就已经处理好了,在后面使用不需要再处理frame-1了
        """
        self.sequences.clear()
        all_frames = len(self.frames)
        left_frame = 1
        for i, keyframe in enumerate(self.keyframes):
            right_frame = (
                self.keyframes[i + 1].num if i + 1 < len(self.keyframes) else all_frames
            )
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
                task = EbSynthTask(
                    style, source, target, frame_num, sequence.keyframe.num, weight
                )
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
            raise ValueError(f"not found key frame num {key_frames_num}")

    def merge_sequences(self):
        merged_frames = []
        border = 1
        for i in range(len(self.sequences)):
            current_seq = self.sequences[i]
            next_seq = self.sequences[i + 1] if i + 1 < len(self.sequences) else None

            if next_seq:
                common_frames_nums = set(current_seq.frames.keys()).intersection(
                    set(range(next_seq.start + border, next_seq.end))
                    if i > 0
                    else set(range(next_seq.start, next_seq.end))
                )

                for j, frame_num in enumerate(common_frames_nums):
                    frame1 = current_seq.generate_frames[frame_num]
                    frame2 = next_seq.generate_frames[frame_num]

                    weight = float(j) / float(len(common_frames_nums))
                    merged_frame = cv2.addWeighted(
                        frame1, 1 - weight, frame2, weight, 0
                    )
                    merged_frames.append((frame_num, merged_frame))
            else:
                if i > 0:
                    prev_seq = self.sequences[i - 1]
                    difference_frames_nums = set(current_seq.frames.keys()) - set(
                        prev_seq.frames.keys()
                    )
                else:
                    difference_frames_nums = set(current_seq.frames.keys())

                for frame_num in difference_frames_nums:
                    merged_frames.append(
                        (frame_num, current_seq.generate_frames[frame_num])
                    )

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


class EbsynthSynthesizeGenerate(EbsynthGenerate):
    """
    keyframe的生成方式是拼接方式
    """

    def __init__(self, keyframes: list[Keyframe], frames: list[np.ndarray], fps: int):
        super().__init__(keyframes, frames, fps)

    def check_keyframes(self):
        """
        检查keyframe的宽高是否一致,并返回宽高
        """

        width = self.keyframes[0].image.shape[1]
        height = self.keyframes[0].image.shape[0]

        for i in range(len(self.keyframes) - 1):
            if (
                    self.keyframes[i].image.shape[0] != self.keyframes[i + 1].image.shape[0]
                    or self.keyframes[i].image.shape[1]
                    != self.keyframes[i + 1].image.shape[1]
            ):
                raise ValueError("keyframe's width and height must be same")

        return width, height

    def synthesize(self):
        """
        把keyframe拼接成一张大图
        """

        # 验证keyframe的宽高是否一致
        w, h = self.check_keyframes()
        keyframe_num = len(self.keyframes)
        # 计算初步的行列数
        base = int(np.ceil(np.sqrt(keyframe_num)))
        rows = base
        cols = base if base * (base - 1) < keyframe_num else base + 1

        print(f'keyframe_num: {keyframe_num} , rows: {rows} , cols: {cols}')

        canvas_h = rows * h
        canvas_w = cols * w
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # 在底图上放入每张图片
        for idx, keyframe in enumerate(self.keyframes):
            img = keyframe.image
            row_idx = idx // cols
            col_idx = idx % cols
            canvas[row_idx * h:(row_idx + 1) * h, col_idx * w:(col_idx + 1) * w] = img
            # 记录keyframe的行列
            keyframe.row = row_idx
            keyframe.col = col_idx

        # 把图片缩放到原始图片大小
        canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)

        guide = KeyframeGuide(w, h, canvas_w, canvas_h, rows, cols, canvas)
        return guide

    @staticmethod
    def split(guide: KeyframeGuide, frame: Keyframe):
        """
        把一张图片拆分成多张图片
        """
        w = guide.width // guide.cols
        h = guide.height // guide.rows
        x = frame.col * w
        y = frame.row * h
        img = guide.image[y:y + h, x:x + w]
        return img
