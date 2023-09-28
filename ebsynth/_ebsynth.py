# fork for Ezsynth(https://github.com/Trentonom0r3/Ezsynth)

import os
import sys
from ctypes import *
from pathlib import Path

import cv2
import numpy as np

libebsynth = None
cached_buffer = {}

EBSYNTH_BACKEND_CPU = 0x0001
EBSYNTH_BACKEND_CUDA = 0x0002
EBSYNTH_BACKEND_AUTO = 0x0000
EBSYNTH_MAX_STYLE_CHANNELS = 8
EBSYNTH_MAX_GUIDE_CHANNELS = 24
EBSYNTH_VOTEMODE_PLAIN = 0x0001  # weight = 1
EBSYNTH_VOTEMODE_WEIGHTED = 0x0002  # weight = 1/(1+error)


def _normalize_img_shape(img):
    img_len = len(img.shape)
    if img_len == 2:
        sh, sw = img.shape
        sc = 0
    elif img_len == 3:
        sh, sw, sc = img.shape

    if sc == 0:
        sc = 1
        img = img[..., np.newaxis]
    return img


def run(img_style, guides,
        patch_size=5,
        num_pyramid_levels=-1,
        num_search_vote_iters=6,
        num_patch_match_iters=4,
        stop_threshold=5,
        uniformity_weight=3500.0,
        extraPass3x3=False,
        ):
    if patch_size < 3:
        raise ValueError("patch_size is too small")
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be an odd number")
    if len(guides) == 0:
        raise ValueError("at least one guide must be specified")

    global libebsynth
    if libebsynth is None:
        if sys.platform[0:3] == 'win':
            libebsynth_path = str(Path(__file__).parent / 'ebsynth.dll')
            libebsynth = CDLL(libebsynth_path)
        else:
            # todo: implement for linux
            pass

        if libebsynth is not None:
            libebsynth.ebsynthRun.argtypes = ( \
                c_int,
                c_int,
                c_int,
                c_int,
                c_int,
                c_void_p,
                c_void_p,
                c_int,
                c_int,
                c_void_p,
                c_void_p,
                POINTER(c_float),
                POINTER(c_float),
                c_float,
                c_int,
                c_int,
                c_int,
                POINTER(c_int),
                POINTER(c_int),
                POINTER(c_int),
                c_int,
                c_void_p,
                c_void_p
            )

    if libebsynth is None:
        return img_style

    img_style = _normalize_img_shape(img_style)
    sh, sw, sc = img_style.shape
    t_h, t_w, t_c = 0, 0, 0

    if sc > EBSYNTH_MAX_STYLE_CHANNELS:
        raise ValueError(f"error: too many style channels {sc}, maximum number is {EBSYNTH_MAX_STYLE_CHANNELS}")

    guides_source = []
    guides_target = []
    guides_weights = []

    for i in range(len(guides)):
        source_guide, target_guide, guide_weight = guides[i]
        source_guide = _normalize_img_shape(source_guide)
        target_guide = _normalize_img_shape(target_guide)
        s_h, s_w, s_c = source_guide.shape
        nt_h, nt_w, nt_c = target_guide.shape

        if s_h != sh or s_w != sw:
            raise ValueError("guide source and style resolution must match style resolution.")

        if t_c == 0:
            t_h, t_w, t_c = nt_h, nt_w, nt_c
        elif nt_h != t_h or nt_w != t_w:
            raise ValueError("guides target resolutions must be equal")

        if s_c != nt_c:
            raise ValueError("guide source and target channels must match exactly.")

        guides_source.append(source_guide)
        guides_target.append(target_guide)

        guides_weights += [guide_weight / s_c] * s_c

    guides_source = np.concatenate(guides_source, axis=-1)
    guides_target = np.concatenate(guides_target, axis=-1)
    guides_weights = (c_float * len(guides_weights))(*guides_weights)

    styleWeight = 1.0
    style_weights = [styleWeight / sc for i in range(sc)]
    style_weights = (c_float * sc)(*style_weights)

    maxPyramidLevels = 0
    for level in range(32, -1, -1):
        if min(min(sh, t_h) * pow(2.0, -level), \
               min(sw, t_w) * pow(2.0, -level)) >= (2 * patch_size + 1):
            maxPyramidLevels = level + 1
            break

    if num_pyramid_levels == -1:
        num_pyramid_levels = maxPyramidLevels
    num_pyramid_levels = min(num_pyramid_levels, maxPyramidLevels)

    num_search_vote_iters_per_level = (c_int * num_pyramid_levels)(*[num_search_vote_iters] * num_pyramid_levels)
    num_patch_match_iters_per_level = (c_int * num_pyramid_levels)(*[num_patch_match_iters] * num_pyramid_levels)
    stop_threshold_per_level = (c_int * num_pyramid_levels)(*[stop_threshold] * num_pyramid_levels)

    buffer = cached_buffer.get((t_h, t_w, sc), None)
    if buffer is None:
        buffer = create_string_buffer(t_h * t_w * sc)
        cached_buffer[(t_h, t_w, sc)] = buffer

    libebsynth.ebsynthRun(EBSYNTH_BACKEND_AUTO,  # backend
                          sc,  # numStyleChannels
                          guides_source.shape[-1],  # numGuideChannels
                          sw,  # sourceWidth
                          sh,  # sourceHeight
                          img_style.tobytes(),
                          # sourceStyleData (width * height * numStyleChannels) bytes, scan-line order
                          guides_source.tobytes(),
                          # sourceGuideData (width * height * numGuideChannels) bytes, scan-line order
                          t_w,  # targetWidth
                          t_h,  # targetHeight
                          guides_target.tobytes(),
                          # targetGuideData (width * height * numGuideChannels) bytes, scan-line order
                          None,
                          # targetModulationData (width * height * numGuideChannels) bytes, scan-line order; pass NULL to switch off the modulation
                          style_weights,  # styleWeights (numStyleChannels) floats
                          guides_weights,  # guideWeights (numGuideChannels) floats
                          uniformity_weight,
                          # uniformityWeight reasonable values are between 500-15000, 3500 is a good default
                          patch_size,  # patchSize odd sizes only, use 5 for 5x5 patch, 7 for 7x7, etc.
                          EBSYNTH_VOTEMODE_WEIGHTED,  # voteMode use VOTEMODE_WEIGHTED for sharper result
                          num_pyramid_levels,  # numPyramidLevels

                          num_search_vote_iters_per_level,
                          # numSearchVoteItersPerLevel how many search/vote iters to perform at each level (array of ints, coarse first, fine last)
                          num_patch_match_iters_per_level,
                          # numPatchMatchItersPerLevel how many Patch-Match iters to perform at each level (array of ints, coarse first, fine last)
                          stop_threshold_per_level,
                          # stopThresholdPerLevel stop improving pixel when its change since last iteration falls under this threshold
                          1 if extraPass3x3 else 0,
                          # extraPass3x3 perform additional polishing pass with 3x3 patches at the finest level, use 0 to disable
                          None,  # outputNnfData (width * height * 2) ints, scan-line order; pass NULL to ignore
                          buffer  # outputImageData  (width * height * numStyleChannels) bytes, scan-line order
                          )

    return np.frombuffer(buffer, dtype=np.uint8).reshape((t_h, t_w, sc)).copy()


# transfer color from source to target
def color_transfer(img_source, img_target):
    guides = [(cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY),
               cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY),
               1)]
    h, w, c = img_source.shape
    result = []
    for i in range(c):
        result += [
            run(img_source[..., i:i + 1], guides=guides,
                patch_size=11,
                num_pyramid_levels=40,
                num_search_vote_iters=6,
                num_patch_match_iters=4,
                stop_threshold=5,
                uniformity_weight=500.0,
                extraPass3x3=True,
                )

        ]
    return np.concatenate(result, axis=-1)


def task(img_style, guides):
    return run(img_style,
               guides,
               patch_size=5,
               num_pyramid_levels=6,
               num_search_vote_iters=12,
               num_patch_match_iters=6,
               uniformity_weight=3500.0,
               extraPass3x3=False
               )
