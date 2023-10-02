import numpy as np
from .interface import Keyframe, Sequence, EbSynthTask


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

        sequence_frames = {}
        for frame_num in range(left_frame, right_frame + 1):
            sequence_frames[frame_num] = frames[frame_num - 1]
        sequence = Sequence(left_frame, keyframe, right_frame, sequence_frames)
        sequences.append(sequence)
        left_frame = keyframe.num
    return sequences


def get_tasks(sequences: list[Sequence], weight: float = 4.0) -> list[EbSynthTask]:
    """
    把序列转换成ebsynth任务

    """
    tasks = []
    for i, sequence in enumerate(sequences):
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


def append_sequences_generate_frame(sequences, key_frames_num, frame_num, generate_frame):
    """
    把生成的图片写回到序列中
    """

    for sequence in sequences:
        if sequence.keyframe.num == key_frames_num:
            sequence.generate_frames[frame_num] = generate_frame
            break
    else:
        raise ValueError(f"not found key frame num {key_frames_num}")


def merge_sequences(sequences: list[Sequence]) -> list[np.ndarray]:
    """
    把序列合并成帧
    """
    merged_frames = []
    border = 1
    for i in range(len(sequences)):
        current_seq = sequences[i]
        next_seq = sequences[i + 1] if i + 1 < len(sequences) else None

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
                prev_seq = sequences[i - 1]
                difference_frames_nums = set(current_seq.frames.keys()) - set(
                    prev_seq.frames.keys()
                )
            else:
                difference_frames_nums = set(current_seq.frames.keys())

            for frame_num in difference_frames_nums:
                merged_frames.append(
                    (frame_num, current_seq.generate_frames[frame_num])
                )
    result = []
    for i, frame in sorted(merged_frames, key=lambda x: x[0]):
        result.append(frame)

    return result
