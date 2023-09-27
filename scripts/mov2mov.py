import os.path
import platform
import time

import cv2
import numpy as np
import pandas
from PIL import Image
from modules import shared, processing
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.processing import StableDiffusionProcessingImg2Img, process_images, Processed
from modules.shared import opts, state
from modules.ui import plaintext_to_html
import modules.scripts as scripts

from scripts.m2m_util import get_mov_all_images, images_to_video
from scripts.m2m_config import mov2mov_outpath_samples, mov2mov_output_dir
from scripts.module_ui_extensions import scripts_mov2mov


def process_mov2mov(p, mov_file, movie_frames, max_frames, resize_mode, w, h, args):
    processing.fix_seed(p)
    images = get_mov_all_images(mov_file, movie_frames)
    if not images:
        print('Failed to parse the video, please check')
        return

    print(f'The video conversion is completed, images:{len(images)}')
    if max_frames == -1 or max_frames > len(images):
        max_frames = len(images)

    max_frames = int(max_frames)

    p.do_not_save_grid = True
    state.job_count = max_frames  # * p.n_iter
    generate_images = []
    for i, image in enumerate(images):
        if i >= max_frames:
            break

        state.job = f"{i + 1} out of {max_frames}"
        if state.skipped:
            state.skipped = False

        if state.interrupted:
            break

        # 存一张底图
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 'RGB')

        p.init_images = [img] * p.batch_size
        proc = scripts_mov2mov.run(p, *args)
        if proc is None:
            print(f'current progress: {i + 1}/{max_frames}')
            processed = process_images(p)
            # 只取第一张
            gen_image = processed.images[0]
            generate_images.append(gen_image)

    if not os.path.exists(shared.opts.data.get("mov2mov_output_dir", mov2mov_output_dir)):
        os.makedirs(shared.opts.data.get("mov2mov_output_dir", mov2mov_output_dir), exist_ok=True)

    r_f = '.mp4'

    print(f'Start generating {r_f} file')

    video = images_to_video(generate_images, movie_frames,
                            os.path.join(shared.opts.data.get("mov2mov_output_dir", mov2mov_output_dir),
                                         str(int(time.time())) + r_f, ))
    print(f'The generation is complete, the directory::{video}')

    return video


def process_keyframes(p, mov_file, fps, df, controlnet_module, controlnet_model, args):
    pass
    # processing.fix_seed(p)
    # images = get_mov_all_images(mov_file, fps)
    # if not images:
    #     print('Failed to parse the video, please check')
    #     return
    #
    # default_prompt = p.prompt
    #
    # # 先生成一张风格图
    # row = df.iloc[0]
    # p.prompt = default_prompt + row['prompt']
    # frame = images[row['frame'] - 1]
    # img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 'RGB')
    # p.init_images = [img]
    # processed = process_images(p)
    # # 只取第一张
    # style = processed.images[0]
    # style = np.asarray(style)
    #
    # # 保存一个初始的controlnet
    # cn = controlnet_extensions.get_process_controlnet(p)
    #
    # generate_images = []
    # p.do_not_save_grid = True
    # state.job_count = len(df)  # * p.n_iter
    # for i, row in df.iterrows():
    #     p.prompt = default_prompt + row['prompt']
    #     frame = images[row['frame'] - 1]
    #     state.job = f"{i + 1} out of {len(df)}"
    #     if state.skipped:
    #         state.skipped = False
    #
    #     if state.interrupted:
    #         break
    #
    #     img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 'RGB')
    #     p.init_images = [img] * p.batch_size
    #
    #     # 介入controlnet
    #     print('insert controlnet ip-adapter')
    #     # 获取controlnet process参数
    #     units = [
    #         {
    #             "module": controlnet_module,
    #             "model": controlnet_model,
    #             "weight": 1.0,
    #             "pixel_perfect": True,
    #             "guidance_start": 0.0,
    #             "guidance_end": 1.0,
    #             "processor_res": 512,
    #             'image': style,
    #         }
    #     ]
    #
    #     controlnet_extensions.extend_units(p, cn, units)
    #
    #     proc = scripts.scripts_img2img.run(p, *args)
    #     if proc is None:
    #         print(f'current progress: {i + 1}/{len(df)}')
    #         processed = process_images(p)
    #         # 只取第一张
    #         gen_image = processed.images[0]
    #         generate_images.append(gen_image)
    #         style = np.asarray(gen_image)
    # # 还原cn
    # controlnet_extensions.extend_units(p, cn, [])
    #
    # if not os.path.exists(shared.opts.data.get("mov2mov_output_dir", mov2mov_output_dir)):
    #     os.makedirs(shared.opts.data.get("mov2mov_output_dir", mov2mov_output_dir), exist_ok=True)
    #
    # return images_to_video(generate_images, fps,
    #                        os.path.join(shared.opts.data.get("mov2mov_output_dir", mov2mov_output_dir),
    #                                     str(int(time.time())) + '.mp4', ))


def check_data_frame(df: pandas.DataFrame):
    # 删除df的frame值为0的行
    df = df[df['frame'] > 0]

    # 判断df是否为空
    if len(df) <= 0:
        return False

    return True


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

            # refiner
            enable_refiner, refiner_checkpoint, refiner_switch_at,
            # mov2mov params

            noise_multiplier,
            movie_frames,
            max_frames,
            # editor
            enable_movie_editor,
            df: pandas.DataFrame,

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

    p = StableDiffusionProcessingImg2Img(
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
        initial_noise_multiplier=noise_multiplier

    )

    p.scripts = scripts_mov2mov
    p.script_args = args

    if not enable_refiner or refiner_checkpoint in (None, "", "None"):
        p.refiner_checkpoint = None
        p.refiner_switch_at = None
    else:
        p.refiner_checkpoint = refiner_checkpoint
        p.refiner_switch_at = refiner_switch_at

    if shared.cmd_opts.enable_console_prompts:
        print(f"\nmov2mov: {prompt}", file=shared.progress_print_out)

    p.extra_generation_params["Mask blur"] = mask_blur

    if not enable_movie_editor:
        print(f'\nStart parsing the number of mov frames')
        generate_video = process_mov2mov(p, mov_file, movie_frames, max_frames, resize_mode, width, height, args)
        processed = Processed(p, [], p.seed, "")
    else:
        # editor
        if platform.system() != 'Windows':
            raise Exception('The editor is currently only supported on Windows')

        # check df no frame
        if not check_data_frame(df):
            raise Exception('Please add a frame')

        # sort df for index
        df = df.sort_values(by='frame').reset_index(drop=True)

        # generate keyframes
        print(f'Start generate keyframes')
        # generate_video = process_keyframes(p, mov_file, movie_frames, df, controlnet_preprocessor, controlnet_model,
        #                                    args)
        # processed = Processed(p, [], p.seed, "")
    p.close()

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return generate_video, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(
        processed.comments, classname="comments")
