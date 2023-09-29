import importlib
import os
import unittest

import cv2

utils = importlib.import_module("extensions.sd-webui-mov2mov.tests.utils", "utils")
utils.setup_test_env()

from ebsynth.ebsynth_generate import EbsynthGenerate, Keyframe, Sequence
from ebsynth._ebsynth import task as EbsyncthRun
from scripts import m2m_util


class MyTestCase(unittest.TestCase):
    def get_image(self, folder, name):
        return cv2.imread(os.path.join(os.path.dirname(__file__), 'images', folder, name))

    def setUp(self) -> None:
        self.keyframes = [Keyframe(i, self.get_image('keys', f'{i:04d}.png'), '') for i in range(1, 72, 10)]
        frames = [self.get_image('video', f'{i:04d}.png') for i in range(0, 72)]

        self.eb_generate = EbsynthGenerate(self.keyframes, frames, 24)

    def test_keyframes(self):
        for i, sequence in enumerate(self.eb_generate.sequences):
            self.assertEqual(sequence.keyframe.num, self.keyframes[i].num)
            self.assertTrue((sequence.keyframe.image == self.keyframes[i].image).all())

    def test_task(self):
        """
        测试生成的任务是否正确

        Returns:

        """

        tasks = self.eb_generate.get_tasks(4.0)
        for task in tasks:
            result = EbsyncthRun(task.style, [(task.source, task.target, task.weight)])
            dir_name = os.path.join(os.path.dirname(__file__), 'images', 'test', f'out_{task.key_frame_num}')
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            cv2.imwrite(os.path.join(dir_name, f'{task.frame_num:04d}.png'), result)

    def test_merge(self):
        """
        测试merge是否正确

        """

        def get_sequence(keyframe_num):
            for sequence in self.eb_generate.sequences:
                if sequence.keyframe.num == keyframe_num:
                    return sequence
            else:
                raise ValueError(f'not found key frame num {keyframe_num}')

        # 模拟结果
        test_dir = os.path.join(os.path.dirname(__file__), 'images', 'test')

        # 获取out_{keyframe}文件夹
        for keyframe in self.keyframes:
            out_dir = os.path.join(test_dir, f'out_{keyframe.num:04d}')
            # 获取out_{keyframe}文件夹下的所有文件,并且按照 {i:04d}.png 的顺序添加到eb_generate.generate_frames
            sequence = get_sequence(keyframe.num)
            for i in range(sequence.start, sequence.end + 1):
                self.eb_generate.append_generate_frames(keyframe.num, i,
                                                        cv2.imread(os.path.join(out_dir, f'{i:04d}.png')))
        # 测试merge
        result = self.eb_generate.merge_sequences(0.4)

        if not os.path.exists(os.path.join(test_dir, 'merge_1')):
            os.mkdir(os.path.join(test_dir, 'merge_1'))

        frames = []

        for i, frame in enumerate(result):
            if frame is not None:
                cv2.imwrite(os.path.join(test_dir, 'merge_1', f'{i:04d}.png'), frame)
                frames.append(frame)
        m2m_util.images_to_video(frames, self.eb_generate.fps,
                                 os.path.join(test_dir, 'merge_1', f'm.mp4'))


if __name__ == '__main__':
    unittest.main()
