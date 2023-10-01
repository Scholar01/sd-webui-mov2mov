import os

import numpy as np
import unittest
import cv2

import importlib

utils = importlib.import_module("extensions.sd-webui-mov2mov.tests.utils", "utils")
utils.setup_test_env()

from ebsynth.ebsynth_generate import EbsynthSynthesizeGenerate, Keyframe, Sequence


class EbsynthGenerateTestCase(unittest.TestCase):
    def get_image(self, folder, name):
        return cv2.imread(os.path.join(os.path.dirname(__file__), 'images', folder, name))

    def setUp(self):
        self.keyframes = [Keyframe(i, self.get_image('keys', f'{i:04d}.png'), '') for i in range(1, 72, 10)]
        self.frames = [self.get_image('video', f'{i:04d}.png') for i in range(0, 72)]
        self.eb_generate = EbsynthSynthesizeGenerate(self.keyframes, self.frames, 24)
        self.eb_generate.setup_sequences()

    def test_synthesize(self):
        guide = self.eb_generate.synthesize()
        self.assertEqual(guide.width, self.frames[0].shape[1])
        self.assertEqual(guide.height, self.frames[0].shape[0])

        self.assertEqual(guide.rows, 3)
        self.assertEqual(guide.cols, 3)

    def test_split(self):
        guide = self.eb_generate.synthesize()
        img = EbsynthSynthesizeGenerate.split(guide, self.keyframes[0])

    def test_preprocess(self):
        self.eb_generate.preprocess(0, 512, 512)
        print(self.eb_generate.sequences[0].keyframes[0].image.shape)


if __name__ == '__main__':
    unittest.main()
