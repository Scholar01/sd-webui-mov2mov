import sys

import cv2
import gradio as gr
import pandas

import modules.scripts as scripts
import os
import platform
import shutil
import subprocess as sp
import modules
from PIL import Image
from modules import script_callbacks, shared, call_queue, sd_samplers, \
    ui_prompt_styles, sd_models, deepbooru
from modules.images import image_data
from modules.call_queue import wrap_gradio_gpu_call
from modules.shared import opts
from modules.ui import ordered_ui_categories, create_sampler_and_steps_selection, switch_values_symbol, \
    create_override_settings_dropdown, detect_image_size_symbol, plaintext_to_html, paste_symbol, \
    clear_prompt_symbol, restore_progress_symbol
from modules.ui_common import folder_symbol, update_generation_info, create_refresh_button
from modules.ui_components import ResizeHandleRow, FormRow, ToolButton, FormGroup, InputAccordion

from scripts import m2m_util
from scripts import mov2mov
from scripts import m2m_hook as patches
from scripts.m2m_config import mov2mov_outpath_samples, mov2mov_output_dir

from tqdm import tqdm

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


class Toprow:
    """Creates a top row UI with prompts, generate button, styles, extra little buttons for things, and enables some functionality related to their operation"""

    def __init__(self, is_img2img, id_part=None):
        if not id_part:
            id_part = "img2img" if is_img2img else "txt2img"
        self.id_part = id_part

        with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
            with gr.Column(elem_id=f"{id_part}_prompt_container", scale=6):
                with gr.Row():
                    with gr.Column(scale=80):
                        with gr.Row():
                            self.prompt = gr.Textbox(label="Prompt", elem_id=f"{id_part}_prompt", show_label=False,
                                                     lines=3,
                                                     placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)",
                                                     elem_classes=["prompt"])
                            self.prompt_img = gr.File(label="", elem_id=f"{id_part}_prompt_image", file_count="single",
                                                      type="binary", visible=False)

                with gr.Row():
                    with gr.Column(scale=80):
                        with gr.Row():
                            self.negative_prompt = gr.Textbox(label="Negative prompt", elem_id=f"{id_part}_neg_prompt",
                                                              show_label=False, lines=3,
                                                              placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)",
                                                              elem_classes=["prompt"])

            self.button_interrogate = None
            self.button_deepbooru = None
            if is_img2img:
                with gr.Column(scale=1, elem_classes="interrogate-col"):
                    self.button_interrogate = gr.Button('Interrogate\nCLIP', elem_id="interrogate")
                    self.button_deepbooru = gr.Button('Interrogate\nDeepBooru', elem_id="deepbooru")

            with gr.Column(scale=1, elem_id=f"{id_part}_actions_column"):
                with gr.Row(elem_id=f"{id_part}_generate_box", elem_classes="generate-box"):
                    self.interrupt = gr.Button('Interrupt', elem_id=f"{id_part}_interrupt",
                                               elem_classes="generate-box-interrupt")
                    self.skip = gr.Button('Skip', elem_id=f"{id_part}_skip", elem_classes="generate-box-skip")
                    self.submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')

                    self.skip.click(
                        fn=lambda: shared.state.skip(),
                        inputs=[],
                        outputs=[],
                    )

                    self.interrupt.click(
                        fn=lambda: shared.state.interrupt(),
                        inputs=[],
                        outputs=[],
                    )

                with gr.Row(elem_id=f"{id_part}_tools"):
                    self.paste = ToolButton(value=paste_symbol, elem_id="paste")

                    self.clear_prompt_button = ToolButton(value=clear_prompt_symbol, elem_id=f"{id_part}_clear_prompt")
                    self.restore_progress_button = ToolButton(value=restore_progress_symbol,
                                                              elem_id=f"{id_part}_restore_progress", visible=False)

                    self.token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_token_counter",
                                                 elem_classes=["token-counter"])
                    self.token_button = gr.Button(visible=False, elem_id=f"{id_part}_token_button")
                    self.negative_token_counter = gr.HTML(value="<span>0/75</span>",
                                                          elem_id=f"{id_part}_negative_token_counter",
                                                          elem_classes=["token-counter"])
                    self.negative_token_button = gr.Button(visible=False, elem_id=f"{id_part}_negative_token_button")

                    self.clear_prompt_button.click(
                        fn=lambda *x: x,
                        _js="confirm_clear_prompt",
                        inputs=[self.prompt, self.negative_prompt],
                        outputs=[self.prompt, self.negative_prompt],
                    )

                self.ui_styles = ui_prompt_styles.UiPromptStyles(id_part, self.prompt, self.negative_prompt)

        self.prompt_img.change(
            fn=modules.images.image_data,
            inputs=[self.prompt_img],
            outputs=[self.prompt, self.prompt_img],
            show_progress=False,
        )


class MovieEditor:
    def __init__(self, gr_movie: gr.Video, gr_fps: gr.Slider):
        self.gr_keyframe = None
        self.gr_frame_number = None
        self.gr_frame_image = None
        self.gr_movie = gr_movie
        self.gr_fps = gr_fps
        self.gr_enable_movie_editor = None
        self.is_windows = platform.system() == "Windows"
        self.frames = []
        self.frame_count = 0
        self.selected_data_frame = -1

    def render(self):
        with InputAccordion(True, label="Movie Editor",
                            elem_id=f"{id_part}_editor_enable") as enable_movie_editor:
            self.gr_enable_movie_editor = enable_movie_editor
            gr.HTML(
                "<div style='color:red;font-weight: bold;border: 2px solid yellow;padding: 10px;font-size: 20px; '>"
                "This feature is in beta version!!! </br>"
                "It only supports Windows!!! </br>"
                "Make sure you have installed the controlnet and T2I-Adapter models."
                "</div>")

            self.gr_frame_image = gr.Image(label="Frame", elem_id=f"{id_part}_video_frame", source="upload",
                                           visible=False, height=480)

            # key frame tabs
            with gr.Tabs(elem_id=f"{id_part}_keyframe_tabs"):
                with gr.TabItem("Custom", id=f"{id_part}_keyframe_tab_custom"):
                    with gr.Row():
                        self.gr_frame_number = gr.Slider(label="Frame number", elem_id=f"{id_part}_video_frame_number",
                                                         step=1,
                                                         maximum=0, minimum=0)
                    with gr.Row(elem_id=f'{id_part}_keyframe_custom_container'):
                        add_keyframe = ToolButton('✚', elem_id=f'{id_part}_video_editor_add_keyframe',
                                                  visible=True,
                                                  tooltip="Add keyframe")
                        remove_keyframe = ToolButton('✖', elem_id=f'{id_part}_video_editor_remove_keyframe',
                                                     visible=True,
                                                     tooltip="Remove selected keyframe")

                        clear_keyframe = ToolButton('🗑', elem_id=f'{id_part}_video_editor_clear_keyframe',
                                                    visible=True,
                                                    tooltip="Clear keyframe")

                with gr.TabItem('Auto', elem_id=f'{id_part}_keyframe_tab_auto'):
                    with gr.Row():
                        key_frame_interval = gr.Slider(label="Key frame interval",
                                                       elem_id=f"{id_part}_video_keyframe_interval", step=1,
                                                       maximum=100, minimum=0, value=2)
                        key_frame_interval_generate = ToolButton('♺', elem_id=f'{id_part}_video_editor_auto_keyframe',
                                                                 visible=True,
                                                                 tooltip="generate")

            with gr.Row():
                data_frame = gr.Dataframe(
                    headers=["id", "frame", "prompt"],
                    datatype=["number", "number", "str"],
                    row_count=1,
                    col_count=(3, 'fixed'),
                    max_rows=None,
                    height=480,
                    elem_id=f"{id_part}_video_editor_custom_data_frame",
                )

            with gr.Row():
                interrogate = gr.Button(value="Clip Interrogate Keyframe", size='sm',
                                        elem_id=f"{id_part}_video_editor_interrogate")
                deepbooru = gr.Button(value="Deepbooru Keyframe", size='sm',
                                      elem_id=f"{id_part}_video_editor_deepbooru")

        self.gr_movie.change(fn=self.movie_change, inputs=[self.gr_movie],
                             outputs=[self.gr_frame_image, self.gr_frame_number, self.gr_fps],
                             show_progress=True)

        self.gr_frame_number.change(fn=self.movie_frame_change,
                                    inputs=[self.gr_movie, self.gr_frame_number],
                                    outputs=[self.gr_frame_image], show_progress=True)

        self.gr_fps.change(fn=self.fps_change, inputs=[self.gr_movie, self.gr_fps],
                           outputs=[self.gr_frame_image, self.gr_frame_number], show_progress=True)

        data_frame.select(self.data_frame_select, data_frame, self.gr_frame_number)

        add_keyframe.click(fn=self.add_keyframe_click, inputs=[data_frame, self.gr_frame_number], outputs=[data_frame],
                           show_progress=False)
        remove_keyframe.click(fn=self.remove_keyframe_click, inputs=[data_frame], outputs=[data_frame],
                              show_progress=False)

        clear_keyframe.click(fn=lambda df: df.drop(df.index, inplace=True), inputs=[data_frame], outputs=[data_frame],
                             show_progress=False)

        key_frame_interval_generate.click(fn=self.key_frame_interval_generate_click,
                                          inputs=[data_frame, key_frame_interval],
                                          outputs=[data_frame], show_progress=True)

        interrogate.click(fn=self.interrogate_keyframe, inputs=[data_frame],
                          outputs=[data_frame], show_progress=True)

        deepbooru.click(fn=self.deepbooru_keyframe, inputs=[data_frame],
                        outputs=[data_frame], show_progress=True)

    def interrogate_keyframe(self, data_frame: pandas.DataFrame):
        """
        Interrogate key frame
        """
        bar = tqdm(total=len(data_frame))
        for index, row in data_frame.iterrows():
            if row['frame'] <= 0:
                continue
            bar.set_description(f'Interrogate key frame {row["frame"]}')
            frame = row['frame'] - 1
            image = self.frames[frame]
            image = Image.fromarray(image)
            prompt = shared.interrogator.interrogate(image.convert("RGB"))
            data_frame.at[index, 'prompt'] = prompt
            bar.update(1)

        return data_frame

    def deepbooru_keyframe(self, data_frame: pandas.DataFrame):
        """
        Deepbooru key frame

        """
        bar = tqdm(total=len(data_frame))
        for index, row in data_frame.iterrows():
            if row['frame'] <= 0:
                continue
            bar.set_description(f'Interrogate key frame {row["frame"]}')
            frame = row['frame'] - 1
            image = self.frames[frame]
            image = Image.fromarray(image)
            prompt = deepbooru.model.tag(image)
            data_frame.at[index, 'prompt'] = prompt
            bar.update(1)

        return data_frame

    def data_frame_select(self, event: gr.SelectData, data_frame: pandas.DataFrame):
        row, col = event.index
        self.selected_data_frame = row
        row = data_frame.iloc[row]
        frame = row['frame']
        if 0 < frame <= self.frame_count:
            return int(frame)
        else:
            return 0

    def add_keyframe_click(self, data_frame: pandas.DataFrame, gr_frame_number: int):
        """
        Add a key frame to the data frame
        """
        if gr_frame_number < 1:
            return data_frame

        data_frame = data_frame[data_frame['frame'] > 0]

        if gr_frame_number in data_frame['frame'].values:
            return data_frame

        row = {
            "id": len(data_frame),
            "frame": gr_frame_number,
            "prompt": ""
        }
        data_frame.loc[len(data_frame)] = row

        data_frame = data_frame.sort_values(by='frame').reset_index(drop=True)

        data_frame['id'] = range(len(data_frame))

        return data_frame

    def remove_keyframe_click(self, data_frame: pandas.DataFrame):
        """
        Remove the selected key frame
        """
        if self.selected_data_frame < 0:
            return data_frame

        data_frame = data_frame.drop(self.selected_data_frame)

        data_frame = data_frame.sort_values(by='frame').reset_index(drop=True)

        data_frame['id'] = range(len(data_frame))

        return data_frame

    def key_frame_interval_generate_click(self, data_frame: pandas.DataFrame, key_frame_interval: int):
        if key_frame_interval < 1:
            return data_frame

        # 按照key_frame_interval的间隔添加关键帧
        for i in range(0, self.frame_count, key_frame_interval):
            data_frame = self.add_keyframe_click(data_frame, i + 1)

        return data_frame

    def movie_change(self, movie_path):
        if not movie_path:
            return gr.Image.update(visible=False), gr.Slider.update(maximum=0, minimum=0), gr.Slider.update()
        fps = m2m_util.get_mov_fps(movie_path)
        self.frames = m2m_util.get_mov_all_images(movie_path, fps, True)
        self.frame_count = len(self.frames)
        return (gr.Image.update(visible=True),
                gr.Slider.update(maximum=self.frame_count, minimum=0, value=0),
                gr.Slider.update(maximum=fps, minimum=0, value=fps))

    def movie_frame_change(self, movie_path, frame_number):
        if not movie_path:
            return gr.Image.update(visible=False)

        if frame_number <= 0:
            return gr.Image.update(visible=True, label=f"Frame: {frame_number}", value=None)

        return gr.Image.update(visible=True, label=f"Frame: {frame_number}", value=self.frames[frame_number - 1])

    def fps_change(self, movie_path, fps):
        if not movie_path:
            return gr.Image.update(visible=False), gr.Slider.update(maximum=0, minimum=0)

        self.frames = m2m_util.get_mov_all_images(movie_path, fps, True)
        self.frame_count = len(self.frames)
        return (gr.Image.update(visible=True),
                gr.Slider.update(maximum=self.frame_count, minimum=0, value=0))


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


def create_refiner():
    with InputAccordion(False, label="Refiner", elem_id=f"{id_part}_enable") as enable_refiner:
        with gr.Row():
            refiner_checkpoint = gr.Dropdown(label='Checkpoint', elem_id=f"{id_part}_checkpoint",
                                             choices=sd_models.checkpoint_tiles(), value='',
                                             tooltip="switch to another model in the middle of generation")
            create_refresh_button(refiner_checkpoint, sd_models.list_models,
                                  lambda: {"choices": sd_models.checkpoint_tiles()},
                                  f"{id_part}_checkpoint_refresh")

            refiner_switch_at = gr.Slider(value=0.8, label="Switch at", minimum=0.01, maximum=1.0, step=0.01,
                                          elem_id=f"{id_part}_switch_at",
                                          tooltip="fraction of sampling steps when the switch to refiner model should happen; 1=never, 0.5=switch in the middle of generation")
    return enable_refiner, refiner_checkpoint, refiner_switch_at


def movie_change(movie):
    if not movie:
        return gr.Image.update(visible=False), gr.Slider.update(visible=False)
    global video_frames
    frames = m2m_util.get_mov_frames(movie)
    video_frames = m2m_util.get_mov_all_images(movie, frames, True)

    return gr.Image.update(visible=True), gr.Slider.update(maximum=frames, visible=True)


video_frames = None


def video_frame_change(movie, frame_number):
    if not movie:
        return gr.Image.update(visible=False)
    global video_frames

    return gr.Image.update(visible=True, label=f"Frame: {frame_number}", value=video_frames[frame_number])


def on_ui_tabs():
    # with gr.Blocks(analytics_enabled=False) as mov2mov_interface:
    with gr.TabItem('mov2mov', id=f"tab_{id_part}", elem_id=f"tab_{id_part}") as mov2mov_interface:
        toprow = Toprow(is_img2img=False, id_part=id_part)
        dummy_component = gr.Label(visible=False)
        with gr.Tab("Generation", id=f"{id_part}_generation") as mov2mov_generation_tab, ResizeHandleRow(
                equal_height=False):
            with gr.Column(variant='compact', elem_id="mov2mov_settings"):
                with gr.Tabs(elem_id=f"mode_{id_part}"):

                    init_mov = gr.Video(label="Video for mov2mov", elem_id=f"{id_part}_mov", show_label=False,
                                        source="upload")

                with FormRow():
                    resize_mode = gr.Radio(label="Resize mode", elem_id=f"{id_part}_resize_mode",
                                           choices=["Just resize", "Crop and resize", "Resize and fill",
                                                    "Just resize (latent upscale)"], type="index",
                                           value="Just resize")
                scripts.scripts_img2img.prepare_ui()

                for category in ordered_ui_categories():
                    if category == "sampler":
                        steps, sampler_name = create_sampler_and_steps_selection(
                            sd_samplers.visible_sampler_names(),
                            id_part)
                    elif category == "dimensions":
                        with FormRow():
                            with gr.Column(elem_id=f"{id_part}_column_size", scale=4):
                                with gr.Tabs():
                                    with gr.Tab(label="Resize to",
                                                elem_id=f"{id_part}_tab_resize_to") as tab_scale_to:
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
                        denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01,
                                                       label='Denoising strength',
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
                                                     label='Movie FPS',
                                                     elem_id=f'{id_part}_movie_frames',
                                                     value=30)
                            max_frames = gr.Number(label='Max FPS', value=-1, elem_id=f'{id_part}_max_frames')


                    elif category == "cfg":
                        with gr.Row():
                            cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0,
                                                  elem_id=f"{id_part}_cfg_scale")
                            image_cfg_scale = gr.Slider(minimum=0, maximum=3.0, step=0.05, label='Image CFG Scale',
                                                        value=1.5, elem_id=f"{id_part}_image_cfg_scale",
                                                        visible=False)

                    elif category == "checkboxes":

                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            pass

                    elif category == "accordions":

                        with gr.Row(elem_id=f"{id_part}_accordions", elem_classes="accordions"):
                            enable_refiner, refiner_checkpoint, refiner_switch_at = create_refiner()



                    elif category == "override_settings":
                        with FormRow(elem_id=f"{id_part}_override_settings_row") as row:
                            override_settings = create_override_settings_dropdown('mov2mov', row)

                    elif category == "scripts":
                        # frame_image, frame_number = create_video_editor(init_mov)
                        editor = MovieEditor(init_mov, movie_frames)
                        editor.render()
                        with FormGroup(elem_id="img2img_script_container"):
                            custom_inputs = scripts.scripts_img2img.setup_ui()

                    if category not in {"accordions"}:
                        scripts.scripts_img2img.setup_ui_for_section(category)

            for script in scripts.scripts_img2img.alwayson_scripts:
                print(script.name, script.args_from, script.args_to, script.filename)
            mov2mov_gallery, result_video, generation_info, html_info, html_log = create_output_panel(id_part,
                                                                                                      opts.mov2mov_output_dir)

            res_switch_btn.click(fn=None, _js="function(){switchWidthHeight('mov2mov')}", inputs=None, outputs=None,
                                 show_progress=False)

            # calc video size
            detect_image_size_btn.click(fn=calc_video_w_h, inputs=[init_mov, width, height],
                                        outputs=[width, height])

            mov2mov_args = dict(
                fn=wrap_gradio_gpu_call(mov2mov.mov2mov, extra_outputs=[None, '', '']),
                _js="submit_mov2mov",
                inputs=[
                           dummy_component,
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

                           # refiner
                           enable_refiner, refiner_checkpoint, refiner_switch_at,

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


img2img_toprow: gr.Row = None


def block_context_init(self, *args, **kwargs):
    origin_block_context_init(self, *args, **kwargs)

    if self.elem_id == 'tab_img2img':
        self.parent.__enter__()
        on_ui_tabs()
        self.parent.__exit__()


def on_app_reload():
    global origin_block_context_init
    if origin_block_context_init:
        patches.undo(__name__, obj=gr.blocks.BlockContext, field="__init__")
        origin_block_context_init = None


origin_block_context_init = patches.patch(__name__, obj=gr.blocks.BlockContext, field="__init__",
                                          replacement=block_context_init)
script_callbacks.on_before_reload(on_app_reload)
script_callbacks.on_ui_settings(on_ui_settings)
# script_callbacks.on_ui_tabs(on_ui_tabs)
