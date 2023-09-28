import importlib
import os
import unittest

import cv2

utils = importlib.import_module("extensions.sd-webui-mov2mov.tests.utils", "utils")
utils.setup_test_env()

from ebsynth.ebsynth_generate import EbsynthGenerate, Keyframe, Sequence
from ebsynth._ebsynth import task as EbsyncthRun


class MyTestCase(unittest.TestCase):
    def get_image(self, folder, name):
        return cv2.imread(os.path.join(os.path.dirname(__file__), 'images', folder, name))

    def setUp(self) -> None:
        self.keyframes = [Keyframe(i, self.get_image('keys', f'{i:04d}.png'), '') for i in range(1, 72, 10)]
        frames = [self.get_image('video', f'{i:04d}.png') for i in range(0, 73)]

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
            self.eb_generate.append_generate_frames(task.key_frame_num, task.frame_num, result)
        frames = self.eb_generate.merge_generate_frames()
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'images', 'test', 'merge')):
            os.mkdir(os.path.join(os.path.dirname(__file__), 'images', 'test', 'merge'))
        for i, frame in enumerate(frames):
            cv2.imwrite(os.path.join(os.path.dirname(__file__), 'images', 'test', 'merge', f'{i:04d}.png'), frame)


if __name__ == '__main__':
    unittest.main()
