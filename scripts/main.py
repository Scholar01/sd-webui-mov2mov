import functools
import os.path
import time

import gradio

import modules.img2img
import modules.scripts as scripts

from modules import shared, script_callbacks
import gradio as gr

from scripts.m2m_util import video_to_images, images_to_video



def hook(hookfunc, oldfunc):
    def foo(*args, **kwargs):
        return hookfunc(*args, **kwargs)

    return foo


class Script(scripts.Script):
    def __init__(self) -> None:
        self.debug = True
        self.video_file = ''
        self.movie_frames = 30
        self.enabled = False

        modules.img2img.img2img = hook(self.mov2mov, modules.img2img.img2img)
        super().__init__()

    def title(self):
        return "Mov2mov"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def noise_multiplier_change(self, noise_multiplier):
        if self.debug:
            print(f'update noise_multiplier:{noise_multiplier}')
        shared.opts.data['initial_noise_multiplier'] = noise_multiplier

    def color_correction_change(self, color_correction):
        if self.debug:
            print(f'update color_correction:{color_correction}')
        shared.opts.data['img2img_color_correction'] = color_correction

    def video_file_component_change(self, video_file_component):
        if self.debug:
            print(f'update video_file:{video_file_component}')
        self.video_file = video_file_component

    def movie_frames_change(self, movie_frames):
        if self.debug:
            print(f'update movie_frames:{movie_frames}')
        self.movie_frames = movie_frames

    def enabled_change(self, enabled):
        if self.debug:
            print(f'update enabled:{enabled}')
        self.enabled = enabled

    def ui(self, is_img2img):
        if is_img2img:
            with gr.Group():
                with gr.Accordion("Mov2mov", open=False):
                    video_file_component = gr.Video()

                    enabled = gr.Checkbox(value=self.enabled, label="Enabled")
                    with gr.Row():
                        noise_multiplier = gr.Slider(minimum=0,
                                                     maximum=1.5,
                                                     step=0.1,
                                                     label='System: Noise multiplier for img2img',
                                                     elem_id='mm_img2img_noise_multiplier',
                                                     value=0)

                        color_correction = gr.Checkbox(
                            value=False,
                            elem_id='mm_img2img_color_correction',
                            label='System: Apply color correction to img2img results to match original colors.')

                    with gr.Row():
                        movie_frames = gr.Slider(minimum=10,
                                                 maximum=60,
                                                 step=1,
                                                 label='Movie Frames',
                                                 elem_id='mm_img2img_movie_frames',
                                                 value=self.movie_frames)

                        # button_apply = gr.Button(value='Apply', variant='secondary', elem_id='mm_img2img_apply')

            noise_multiplier.change(fn=self.noise_multiplier_change, inputs=[noise_multiplier])
            color_correction.change(fn=self.color_correction_change, inputs=[color_correction])
            video_file_component.change(fn=self.video_file_component_change, inputs=[video_file_component])
            movie_frames.change(fn=self.movie_frames_change, inputs=[movie_frames])
            enabled.change(fn=self.enabled_change, inputs=[enabled])

            # button_apply.click(fn=self.do_apply,
            #                    inputs=[enabled, video_file_component, movie_frames, noise_multiplier,
            #                            color_correction])

            return [enabled, video_file_component, noise_multiplier, color_correction, movie_frames]

    # def do_apply(self, enabled, video_file, movie_frames, noise_multiplier, color_correction):
    #     self.enabled = enabled
    #     self.video_file = video_file
    #     self.movie_frames = movie_frames
    #     shared.opts.data['initial_noise_multiplier'] = noise_multiplier
    #     shared.opts.data['img2img_color_correction'] = color_correction

    def mov2mov(self, id_task: str, mode: int, prompt: str, negative_prompt: str, prompt_styles, init_img,
                sketch,
                init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig, init_img_inpaint,
                init_mask_inpaint,
                steps: int, sampler_index: int, mask_blur: int, mask_alpha: float, inpainting_fill: int,
                restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float,
                image_cfg_scale: float,
                denoising_strength: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int,
                seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int, resize_mode: int,
                inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int,
                img2img_batch_input_dir: str, img2img_batch_output_dir: str, img2img_batch_inpaint_mask_dir: str,
                override_settings_texts, *args):
        if self.enabled:
            if not self.video_file:
                raise ValueError('not video file')

            # 路径处理

            print(f'Start converting video, frame number:{self.movie_frames}')

            mov2mov_images_path = os.path.join(scripts.basedir(), 'outputs', 'mov2mov-images')
            if self.debug:
                print(f'mov2mov_images_path：{mov2mov_images_path}')

            if not os.path.exists(mov2mov_images_path):
                os.mkdir(mov2mov_images_path)

            current_mov2mov_images_path = os.path.join(mov2mov_images_path, str(int(time.time())))
            os.mkdir(current_mov2mov_images_path)

            if self.debug:
                print(f'current_mov2mov_images_path：{current_mov2mov_images_path}')

            v2i_path = os.path.join(current_mov2mov_images_path, 'In')
            i2v_path = os.path.join(current_mov2mov_images_path, 'Out')
            os.mkdir(v2i_path)
            os.mkdir(i2v_path)

            # 首先把视频转换成图片
            frames, counts = video_to_images(self.movie_frames, self.video_file, v2i_path)

            print(f'The video conversion is completed, frames:{frames}, images:{counts}')

            mode = 5
            img2img_batch_input_dir = v2i_path
            img2img_batch_output_dir = i2v_path

        result = img2img(id_task, mode, prompt, negative_prompt, prompt_styles, init_img, sketch,
                         init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig, init_img_inpaint,
                         init_mask_inpaint,
                         steps, sampler_index, mask_blur, mask_alpha, inpainting_fill,
                         restore_faces, tiling, n_iter, batch_size, cfg_scale,
                         image_cfg_scale,
                         denoising_strength, seed, subseed, subseed_strength, seed_resize_from_h,
                         seed_resize_from_w, seed_enable_extras, height, width, resize_mode,
                         inpaint_full_res, inpaint_full_res_padding, inpainting_mask_invert,
                         img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
                         override_settings_texts, *args)
        if self.enabled:
            print('Start generating .mp4 file')
            # todo: frames not update
            movie = images_to_video(self.movie_frames, width, height, i2v_path, os.path.join(i2v_path, 'movie.mp4'))
            print(f'The generation is complete, the directory::{movie}')
        return result


img2img = modules.img2img.img2img


def on_script_unloaded():
    modules.img2img.img2img = img2img


script_callbacks.on_script_unloaded(on_script_unloaded)
