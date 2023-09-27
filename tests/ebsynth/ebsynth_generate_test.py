import os

import numpy as np
import unittest
import cv2

import importlib

utils = importlib.import_module("extensions.sd-webui-mov2mov.tests.utils", "utils")
utils.setup_test_env()

from ebsynth.ebsynth_generate import EbsynthGenerate, Keyframe, Sequence


class EbsynthGenerateTestCase(unittest.TestCase):

    def get_image(self, folder, name):
        return cv2.imread(os.path.join(os.path.dirname(__file__), 'images', folder, name))

    def test_sequences(self):
        # 模拟100帧的视频
        keyframes = [
            Keyframe(1, None, None),
            Keyframe(10, None, None),
            Keyframe(20, None, None),
        ]
        frames = [np.zeros((100, 100, 3))] * 30

        eb_generate = EbsynthGenerate(keyframes, frames, 24)
        eb_generate.setup_sequences()
        self.assertEqual(len(eb_generate.sequences), 3)
        self.assertEqual(eb_generate.sequences[0].start, 1)
        self.assertEqual(eb_generate.sequences[0].keyframe.num, 1)
        self.assertEqual(eb_generate.sequences[0].end, 10)

        self.assertEqual(eb_generate.sequences[1].start, 1)
        self.assertEqual(eb_generate.sequences[1].keyframe.num, 10)
        self.assertEqual(eb_generate.sequences[1].end, 20)

        self.assertEqual(eb_generate.sequences[2].start, 10)
        self.assertEqual(eb_generate.sequences[2].keyframe.num, 20)
        self.assertEqual(eb_generate.sequences[2].end, 30)

        keyframes = [
            Keyframe(1, None, None),
            Keyframe(3, None, None),
            Keyframe(5, None, None),
        ]
        frames = [np.zeros((100, 100, 3))] * 10

        eb_generate = EbsynthGenerate(keyframes, frames, 24)
        eb_generate.setup_sequences()

        self.assertEqual(len(eb_generate.sequences), 3)
        self.assertEqual(eb_generate.sequences[0].start, 1)
        self.assertEqual(eb_generate.sequences[0].keyframe.num, 1)
        self.assertEqual(eb_generate.sequences[0].end, 3)

        self.assertEqual(eb_generate.sequences[1].start, 1)
        self.assertEqual(eb_generate.sequences[1].keyframe.num, 3)
        self.assertEqual(eb_generate.sequences[1].end, 5)

        self.assertEqual(eb_generate.sequences[2].start, 3)
        self.assertEqual(eb_generate.sequences[2].keyframe.num, 5)
        self.assertEqual(eb_generate.sequences[2].end, 10)

        keyframes = [
            Keyframe(1, None, None),
            Keyframe(3, None, None),
            Keyframe(5, None, None),
        ]
        frames = [np.zeros((100, 100, 3))] * 5

        eb_generate = EbsynthGenerate(keyframes, frames, 24)
        eb_generate.setup_sequences()

        self.assertEqual(len(eb_generate.sequences), 3)
        self.assertEqual(eb_generate.sequences[0].start, 1)
        self.assertEqual(eb_generate.sequences[0].keyframe.num, 1)
        self.assertEqual(eb_generate.sequences[0].end, 3)

        self.assertEqual(eb_generate.sequences[1].start, 1)
        self.assertEqual(eb_generate.sequences[1].keyframe.num, 3)
        self.assertEqual(eb_generate.sequences[1].end, 5)

        self.assertEqual(eb_generate.sequences[2].start, 3)
        self.assertEqual(eb_generate.sequences[2].keyframe.num, 5)
        self.assertEqual(eb_generate.sequences[2].end, 5)

    
if __name__ == '__main__':
    unittest.main()
