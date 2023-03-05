import os
import shutil
import sys

import gradio as gr
import platform
import modules.scripts as scripts
from modules import script_callbacks, shared, ui_postprocessing, call_queue
from modules.call_queue import wrap_gradio_gpu_call
from modules.sd_samplers import samplers_for_img2img
from modules.shared import opts
from modules.ui import paste_symbol, clear_prompt_symbol, extra_networks_symbol, apply_style_symbol, save_style_symbol, \
    create_refresh_button, create_sampler_and_steps_selection, ordered_ui_categories, switch_values_symbol, \
    create_seed_inputs, create_override_settings_dropdown
from modules.ui_common import folder_symbol, plaintext_to_html
from modules.ui_components import ToolButton, FormRow, FormGroup
import modules.generation_parameters_copypaste as parameters_copypaste
import subprocess as sp
from scripts import mov2mov
from scripts.m2m_config import mov2mov_outpath_samples, mov2mov_output_dir
from scripts.m2m_modnet import modnet_models

id_part = "mov2mov"


def create_toprow():
    with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
        with gr.Column(elem_id=f"{id_part}_prompt_container", scale=6):
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        prompt = gr.Textbox(label="Prompt", elem_id=f"{id_part}_prompt", show_label=False, lines=3,
                                            placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)")

            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        negative_prompt = gr.Textbox(label="Negative prompt", elem_id=f"{id_part}_neg_prompt",
                                                     show_label=False, lines=2,
                                                     placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)")

        with gr.Column(scale=1, elem_id=f"{id_part}_actions_column"):
            with gr.Row(elem_id=f"{id_part}_generate_box"):
                interrupt = gr.Button('Interrupt', elem_id=f"{id_part}_interrupt")
                skip = gr.Button('Skip', elem_id=f"{id_part}_skip")
                submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')

                skip.click(
                    fn=lambda: shared.state.skip(),
                    inputs=[],
                    outputs=[],
                )

                interrupt.click(
                    fn=lambda: shared.state.interrupt(),
                    inputs=[],
                    outputs=[],
                )

    return prompt, negative_prompt, submit


def save_video(video):
    path = shared.opts.outdir_save
    index = len([path for path in os.listdir(path) if path.endswith('.mp4')]) + 1
    video_path = os.path.join(path, str(index).zfill(5) + '.mp4')
    shutil.copyfile(video, video_path)
    filename = os.path.relpath(video_path, path)
    return gr.File.update(value=video_path, visible=True), plaintext_to_html(f"Saved: {filename}")


def create_output_panel(tabname, outdir):
    from modules import shared
    import modules.generation_parameters_copypaste as parameters_copypaste

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
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id=f"{tabname}_gallery").style(grid=4)
            result_video = gr.Video(label='Output Video', show_label=False, elem_id=f'{tabname}_video')

        generation_info = None
        with gr.Column():
            with gr.Row(elem_id=f"image_buttons_{tabname}"):
                open_folder_button = gr.Button(folder_symbol,
                                               elem_id="hidden_element" if shared.cmd_opts.hide_ui_dir_config else f'open_folder_{tabname}')

                save = gr.Button('Save', elem_id=f'save_{tabname}')
                # save_zip = gr.Button('Zip', elem_id=f'save_zip_{tabname}')

            open_folder_button.click(
                fn=lambda: open_folder(shared.opts.outdir_samples or outdir),
                inputs=[],
                outputs=[],
            )

            with gr.Row():
                download_files = gr.File(None, file_count="multiple", interactive=False, show_label=False,
                                         visible=False, elem_id=f'download_files_{tabname}')

            with gr.Group():
                html_info = gr.HTML(elem_id=f'html_info_{tabname}')
                html_log = gr.HTML(elem_id=f'html_log_{tabname}')

                generation_info = gr.Textbox(visible=False, elem_id=f'generation_info_{tabname}')

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
    with gr.Blocks(analytics_enabled=False) as mov2mov_tabs:
        dummy_component = gr.Label(visible=False)

        mov2mov_prompt, mov2mov_negative_prompt, submit = create_toprow()

        # create tabs
        with FormRow().style(equal_height=False):
            with gr.Column(variant='compact', elem_id=f"{id_part}_settings"):
                with gr.Tabs(elem_id=f"mode_{id_part}"):
                    with gr.TabItem('mov2mov', id='mov2mov', elem_id=f"{id_part}_mov2mov_tab") as tab_mov2mov:
                        init_mov = gr.Video(label="Video for mov2mov", elem_id="{id_part}_mov", show_label=False,
                                            source="upload")  # .style(height=480)

                with FormRow():
                    resize_mode = gr.Radio(label="Resize mode", elem_id="resize_mode",
                                           choices=["Just resize", "Crop and resize", "Resize and fill",
                                                    "Just resize (latent upscale)"], type="index", value="Just resize")

                for category in ordered_ui_categories():
                    if category == "sampler":
                        steps, sampler_index = create_sampler_and_steps_selection(samplers_for_img2img, "mov2mov")

                    elif category == "dimensions":
                        with FormRow():
                            with gr.Column(elem_id=f"{id_part}_column_size", scale=4):
                                width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512,
                                                  elem_id=f"{id_part}_width")
                                height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512,
                                                   elem_id=f"{id_part}_height")

                            res_switch_btn = ToolButton(value=switch_values_symbol, elem_id=f"{id_part}_res_switch_btn")
                            if opts.dimensions_and_batch_together:
                                with gr.Column(elem_id=f"{id_part}_column_batch"):
                                    generate_mov_mode = gr.Radio(label="Generate Movie Mode", elem_id="movie_mode",
                                                                 choices=["MP4V", "H.264", "XVID", ], type="index",
                                                                 value="XVID")

                                    noise_multiplier = gr.Slider(minimum=0,
                                                                 maximum=1.5,
                                                                 step=0.1,
                                                                 label='Noise multiplier',
                                                                 elem_id=f'{id_part}_noise_multiplier',
                                                                 value=0)

                                    # color_correction = gr.Checkbox(
                                    #     value=False,
                                    #     elem_id=f'{id_part}_color_correction',
                                    #     label='Color correction')

                    elif category == "cfg":
                        with FormGroup():
                            with FormRow():
                                cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0,
                                                      elem_id=f"{id_part}_cfg_scale")
                                image_cfg_scale = gr.Slider(minimum=0, maximum=3.0, step=0.05, label='Image CFG Scale',
                                                            value=1.5, elem_id=f"{id_part}_image_cfg_scale",
                                                            visible=shared.sd_model and shared.sd_model.cond_stage_key == "edit")
                            denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01,
                                                           label='Denoising strength', value=0.75,
                                                           elem_id=f"{id_part}_denoising_strength")
                            movie_frames = gr.Slider(minimum=10,
                                                     maximum=60,
                                                     step=1,
                                                     label='Movie Frames',
                                                     elem_id=f'{id_part}_movie_frames',
                                                     value=30)

                    elif category == "seed":
                        max_frames = gr.Number(label='Max Frames', value=-1, elem_id=f'{id_part}_max_frames')
                        seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox = create_seed_inputs(
                            'mov2mov')

                        seed.style(container=False)

                    elif category == "checkboxes":
                        with FormRow(elem_id=f"{id_part}_checkboxes", variant="compact"):
                            restore_faces = gr.Checkbox(label='Restore faces', value=False,
                                                        visible=len(shared.face_restorers) > 1,
                                                        elem_id=f"{id_part}_restore_faces")
                            tiling = gr.Checkbox(label='Tiling', value=False, elem_id=f"{id_part}_tiling")

                            extract_characters = gr.Checkbox(label='Extract characters (Need modnet models)',
                                                             value=False, elem_id=f"{id_part}_extract_characters")

                            extract_characters.change(fn=None, inputs=[extract_characters], _js='showModnetModels')

                        with FormRow(elem_id=f'{id_part}_modnet_models'):
                            modnet_model = gr.Dropdown(label='Model', choices=list(modnet_models), value='none',
                                                       elem_id=f"{id_part}_modnet_model")



                    elif category == "override_settings":
                        with FormRow(elem_id=f"{id_part}_override_settings_row") as row:
                            override_settings = create_override_settings_dropdown('mov2mov', row)

                    elif category == "scripts":
                        with FormGroup(elem_id=f"{id_part}_script_container"):
                            custom_inputs = scripts.scripts_img2img.setup_ui()

            mov2mov_gallery, result_video, generation_info, html_info, html_log = create_output_panel("mov2mov",
                                                                                                      mov2mov_output_dir)

            mov2mov_args = dict(
                fn=wrap_gradio_gpu_call(mov2mov.mov2mov, extra_outputs=[None, '', '']),
                _js="submit_mov2mov",
                inputs=[
                           dummy_component,
                           # dummy_component, # mode
                           mov2mov_prompt,
                           mov2mov_negative_prompt,
                           init_mov,
                           steps,
                           sampler_index,
                           restore_faces,
                           tiling,
                           extract_characters,
                           modnet_model,

                           generate_mov_mode,
                           noise_multiplier,
                           # color_correction,
                           cfg_scale,
                           image_cfg_scale,
                           denoising_strength,
                           movie_frames,
                           max_frames,
                           seed,
                           subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox,
                           height,
                           width,
                           resize_mode,
                           override_settings,
                       ] + custom_inputs,
                outputs=[
                    mov2mov_gallery,
                    result_video,
                    generation_info,
                    html_info,
                    html_log,
                ],
                show_progress=False,
            )
            submit.click(**mov2mov_args)

    return [(mov2mov_tabs, "mov2mov", f"{id_part}_tabs")]


script_callbacks.on_ui_tabs(on_ui_tabs)
