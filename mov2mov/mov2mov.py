import os
import time

import cv2
import pandas
import PIL
from PIL import Image

from modules import scripts, shared, processing
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.processing import process_images, Processed
from modules.shared import opts, state
from modules.ui import plaintext_to_html

from .processing import StableDiffusionProcessingMov2Mov
from .config import mov2mov_outpath_samples, mov2mov_output_dir
from .util import get_mov_all_images, images_to_video
from .ebsynth_util import *
from .interface import *

scripts_mov2mov = scripts.ScriptRunner()
scripts_preprocess1 = scripts.ScriptRunner()
scripts_preprocess2 = scripts.ScriptRunner()


def save_video(images, fps, extension='.mp4'):
    if not os.path.exists(shared.opts.data.get("mov2mov_output_dir", mov2mov_output_dir)):
        os.makedirs(shared.opts.data.get("mov2mov_output_dir", mov2mov_output_dir), exist_ok=True)

    r_f = extension

    print(f'Start generating {r_f} file')

    video = images_to_video(images, fps,
                            os.path.join(shared.opts.data.get("mov2mov_output_dir", mov2mov_output_dir),
                                         str(int(time.time())) + r_f, ))
    print(f'The generation is complete, the directory::{video}')

    return video


def process_mov2mov(p: StableDiffusionProcessingMov2Mov):
    """
    默认处理程序

    """
    print(f'\nStart generate of mov frames')

    max_frames = len(p.frames)
    state.job_count = max_frames
    generate_images = []
    for i, frame in enumerate(p.frames):
        state.job = f"{i + 1} out of {max_frames}"
        if state.skipped:
            state.skipped = False

        if state.interrupted:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 'RGB')

        p.init_images = [img]
        proc = scripts_mov2mov.run(p, *p.scripts_custom_args)
        if proc is None:
            print(f'current progress: {i + 1}/{max_frames}')
            processed = process_images(p)
            # 只取第一张
            gen_image = processed.images[0]
            generate_images.append(gen_image)

    video = save_video(generate_images, p.fps)
    return video

def process_step_by_step(p: StableDiffusionProcessingMov2Mov):
    """
    根据关键帧逐逐帧生成

    """

    print(f'Start generate keyframes')




def mov2mov(id_task: str,
            prompt,
            negative_prompt,
            prompt_styles,
            mov_file,
            steps,
            sampler_name,
            cfg_scale,
            image_cfg_scale,
            denoising_strength,
            height,
            width,
            resize_mode,
            override_settings_texts,

            noise_multiplier,
            movie_fps,
            # editor
            enable_movie_editor,
            df: pandas.DataFrame,
            eb_weight,
            keyframe_mode,
            preprocess1_inputs_len,
            preprocess2_inputs_len,
            *args):
    if not mov_file:
        raise Exception('Error！ Please add a video file!')

    override_settings = create_override_settings_dict(override_settings_texts)
    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    mask_blur = 4
    inpainting_fill = 1
    inpaint_full_res = False
    inpaint_full_res_padding = 32
    inpainting_mask_invert = 0

    frames = get_mov_all_images(mov_file, movie_fps)
    if not frames:
        print('Failed to parse the video, please check')
        return

    p = StableDiffusionProcessingMov2Mov(
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.data.get("mov2mov_outpath_samples", mov2mov_outpath_samples),
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=prompt_styles,
        sampler_name=sampler_name,
        batch_size=1,
        n_iter=1,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        init_images=[None],
        mask=None,

        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        image_cfg_scale=image_cfg_scale,
        inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
        override_settings=override_settings,
        initial_noise_multiplier=noise_multiplier,

        fps=movie_fps,
        frames=frames,

        beta=enable_movie_editor,
        data_frames=df,
        keyframes=[],
        ebsynth_weight=eb_weight,
        keyframe_mode=keyframe_mode,
    )

    # 初始化数据
    p.setup_script_args(preprocess1_inputs_len, preprocess2_inputs_len, scripts_mov2mov, args)
    p.setup_frames()
    p.setup_keyframes()

    # 修复seed并保持固定
    processing.fix_seed(p)

    p.do_not_save_grid = True

    if not enable_movie_editor:
        video = process_mov2mov(p)

    processed = Processed(p, [], p.seed, "")
    p.close()
    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return video, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(
        processed.comments, classname="comments")
