import os
import shutil
import sys

import gradio as gr
import platform
import modules.scripts as scripts
from modules import script_callbacks, shared, ui_postprocessing, call_queue, ui_extra_networks, sd_samplers
from modules.call_queue import wrap_gradio_gpu_call
from modules.sd_samplers import samplers_for_img2img
from modules.shared import opts
from rich import print
import subprocess as sp
from modules.ui import Toprow, ordered_ui_categories, create_sampler_and_steps_selection, switch_values_symbol, \
    create_override_settings_dropdown, detect_image_size_symbol, resize_from_to_html, plaintext_to_html
from modules.ui_common import folder_symbol, update_generation_info
from modules.ui_components import ResizeHandleRow, FormRow, ToolButton, FormGroup, FormHTML

from scripts import mov2mov
from scripts import m2m_util
from scripts.m2m_config import mov2mov_outpath_samples, mov2mov_output_dir
from scripts.m2m_modnet import modnet_models

id_part = "mov2mov"


def save_video(video):
    path = 'logs/movies'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    index = len([path for path in os.listdir(path) if path.endswith('.mp4')]) + 1
    video_path = os.path.join(path, str(index).zfill(5) + '.mp4')
    shutil.copyfile(video, video_path)
    filename = os.path.relpath(video_path, path)
    return gr.File.update(value=video_path, visible=True), plaintext_to_html(f"Saved: {filename}")


def create_output_panel(tabname, outdir):
    def open_folder(f):
        if not os.path.exists(f):
            print(f'Folder "{f}" does not exist. After you create an image, the folder will be created.')
            return
        elif not os.path.isdir(f):
            print(f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
""", file=sys.stderr)
            return

        if not shared.cmd_opts.hide_ui_dir_config:
            path = os.path.normpath(f)
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                sp.Popen(["open", path])
            elif "microsoft-standard-WSL2" in platform.uname().release:
                sp.Popen(["wsl-open", path])
            else:
                sp.Popen(["xdg-open", path])

    with gr.Column(variant='panel', elem_id=f"{tabname}_results"):
        with gr.Group(elem_id=f"{tabname}_gallery_container"):
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id=f"{tabname}_gallery", columns=4,
                                        preview=True, height=shared.opts.gallery_height or None)
            result_video = gr.PlayableVideo(label='Output Video', show_label=False,
                                    elem_id=f'{tabname}_video')

        generation_info = None
        with gr.Column():
            with gr.Row(elem_id=f"image_buttons_{tabname}", elem_classes="image-buttons"):
                open_folder_button = ToolButton(folder_symbol, elem_id=f'{tabname}_open_folder',
                                                visible=not shared.cmd_opts.hide_ui_dir_config,
                                                tooltip="Open images output directory.")

                if tabname != "extras":
                    save = ToolButton('💾', elem_id=f'save_{tabname}',
                                      tooltip=f"Save the image to a dedicated directory ({shared.opts.outdir_save}).")

            open_folder_button.click(
                fn=lambda: open_folder(shared.opts.outdir_samples or outdir),
                inputs=[],
                outputs=[],
            )

            download_files = gr.File(None, file_count="multiple", interactive=False, show_label=False,
                                     visible=False, elem_id=f'download_files_{tabname}')

            with gr.Group():
                html_info = gr.HTML(elem_id=f'html_info_{tabname}', elem_classes="infotext")
                html_log = gr.HTML(elem_id=f'html_log_{tabname}', elem_classes="html-log")

                generation_info = gr.Textbox(visible=False, elem_id=f'generation_info_{tabname}')
                if tabname == 'txt2img' or tabname == 'img2img' or tabname == 'mov2mov':
                    generation_info_button = gr.Button(visible=False, elem_id=f"{tabname}_generation_info_button")
                    generation_info_button.click(
                        fn=update_generation_info,
                        _js="function(x, y, z){ return [x, y, selected_gallery_index()] }",
                        inputs=[generation_info, html_info, html_info],
                        outputs=[html_info, html_info],
                        show_progress=False,
                    )

                save.click(
                    fn=call_queue.wrap_gradio_call(save_video),
                    inputs=[
                        result_video
                    ],
                    outputs=[
                        download_files,
                        html_log,
                    ],
                    show_progress=False,
                )

            return result_gallery, result_video, generation_info, html_info, html_log


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as mov2mov_interface:
        toprow = Toprow(is_img2img=False, id_part=id_part)
        dummy_component = gr.Label(visible=False)
        with gr.Tab("Generation", id=f"{id_part}_generation") as mov2mov_generation_tab, ResizeHandleRow(
                equal_height=False):
            with gr.Column(variant='compact', elem_id="mov2mov_settings"):
                with gr.Tabs(elem_id=f"mode_{id_part}"):
                    with gr.TabItem('Input Video', id='mov2mov', elem_id=f"{id_part}_mov2mov_tab") as tab_mov2mov:
                        init_mov = gr.Video(label="Video for mov2mov", elem_id="{id_part}_mov", show_label=False,
                                            source="upload")

                with FormRow():
                    resize_mode = gr.Radio(label="Resize mode", elem_id="resize_mode",
                                           choices=["Just resize", "Crop and resize", "Resize and fill",
                                                    "Just resize (latent upscale)"], type="index", value="Just resize")
                scripts.scripts_img2img.prepare_ui()

                for category in ordered_ui_categories():
                    if category == "sampler":
                        steps, sampler_name = create_sampler_and_steps_selection(sd_samplers.visible_sampler_names(),
                                                                                 id_part)
                    elif category == "dimensions":
                        with FormRow():
                            with gr.Column(elem_id=f"{id_part}_column_size", scale=4):
                                with gr.Tabs():
                                    with gr.Tab(label="Resize to", elem_id=f"{id_part}_tab_resize_to") as tab_scale_to:
                                        with FormRow():
                                            with gr.Column(elem_id=f"{id_part}_column_size", scale=4):
                                                width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width",
                                                                  value=512, elem_id=f"{id_part}_width")
                                                height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height",
                                                                   value=512, elem_id=f"{id_part}_height")
                                            with gr.Column(elem_id=f"{id_part}_dimensions_row", scale=1,
                                                           elem_classes="dimensions-tools"):
                                                res_switch_btn = ToolButton(value=switch_values_symbol,
                                                                            elem_id=f"{id_part}_res_switch_btn")
                                                detect_image_size_btn = ToolButton(value=detect_image_size_symbol,
                                                                                   elem_id=f"{id_part}_detect_image_size_btn")
                    elif category == "denoising":
                        denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength',
                                                       value=0.75, elem_id=f"{id_part}_denoising_strength")

                        noise_multiplier = gr.Slider(minimum=0,
                                                     maximum=1.5,
                                                     step=0.01,
                                                     label='Noise multiplier',
                                                     elem_id=f'{id_part}_noise_multiplier',
                                                     value=1)
                        with gr.Row(elem_id=f"{id_part}_frames_setting"):
                            movie_frames = gr.Slider(minimum=10,
                                                     maximum=60,
                                                     step=1,
                                                     label='Movie Frames',
                                                     elem_id=f'{id_part}_movie_frames',
                                                     value=30)
                            max_frames = gr.Number(label='Max Frames', value=-1, elem_id=f'{id_part}_max_frames')


                    elif category == "cfg":
                        with gr.Row():
                            cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0,
                                                  elem_id=f"{id_part}_cfg_scale")
                            image_cfg_scale = gr.Slider(minimum=0, maximum=3.0, step=0.05, label='Image CFG Scale',
                                                        value=1.5, elem_id=f"{id_part}_image_cfg_scale", visible=False)

                    elif category == "checkboxes":

                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            pass

                    elif category == "accordions":

                        with gr.Row(elem_id=f"{id_part}_accordions", elem_classes="accordions"):
                            scripts.scripts_img2img.setup_ui_for_section(category)

                    elif category == "override_settings":
                        with FormRow(elem_id=f"{id_part}_override_settings_row") as row:
                            override_settings = create_override_settings_dropdown('mov2mov', row)

                    elif category == "scripts":
                        with FormGroup(elem_id="img2img_script_container"):
                            custom_inputs = scripts.scripts_img2img.setup_ui()

                    if category not in {"accordions"}:
                        scripts.scripts_img2img.setup_ui_for_section(category)

            mov2mov_gallery, result_video, generation_info, html_info, html_log = create_output_panel(id_part,
                                                                                                      opts.mov2mov_output_dir)

            res_switch_btn.click(fn=None, _js="function(){switchWidthHeight('mov2mov')}", inputs=None, outputs=None,
                                 show_progress=False)

            # calc video size
            detect_image_size_btn.click(fn=calc_video_w_h, inputs=[init_mov, width, height], outputs=[width, height])

            mov2mov_args = dict(
                fn=wrap_gradio_gpu_call(mov2mov.mov2mov, extra_outputs=[None, '', '']),
                _js="submit_mov2mov",
                inputs=[
                           dummy_component,
                           # dummy_component,
                           toprow.prompt,
                           toprow.negative_prompt,
                           toprow.ui_styles.dropdown,
                           init_mov,
                           steps,
                           sampler_name,
                           cfg_scale,
                           image_cfg_scale,
                           denoising_strength,
                           height,
                           width,
                           resize_mode,
                           override_settings,

                           noise_multiplier,
                           movie_frames,
                           max_frames,

                       ] + custom_inputs,
                outputs=[
                    result_video,
                    generation_info,
                    html_info,
                    html_log,
                ],
                show_progress=False,
            )

            toprow.submit.click(**mov2mov_args)

    return [(mov2mov_interface, "mov2mov", f"{id_part}_tabs")]


def calc_video_w_h(video, width, height):
    if not video:
        return width, height

    return m2m_util.calc_video_w_h(video)


def on_ui_settings():
    section = ('mov2mov', "Mov2Mov")
    shared.opts.add_option("mov2mov_outpath_samples", shared.OptionInfo(
        mov2mov_outpath_samples, "Mov2Mov output path for image", section=section))
    shared.opts.add_option("mov2mov_output_dir", shared.OptionInfo(
        mov2mov_output_dir, "Mov2Mov output path for video", section=section))


script_callbacks.on_ui_settings(on_ui_settings)  # 注册进设置页
script_callbacks.on_ui_tabs(on_ui_tabs)
