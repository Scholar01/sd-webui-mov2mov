from contextlib import ExitStack

import gradio as gr

from modules import (
    script_callbacks,
    scripts,
    shared,
    ui_toprow,
)
from modules.call_queue import wrap_gradio_gpu_call
from modules.shared import opts
from modules.ui import (
    create_output_panel,
    create_override_settings_dropdown,
    ordered_ui_categories,
    resize_from_to_html,
    switch_values_symbol,
    detect_image_size_symbol,
)
from modules.ui_components import (
    FormGroup,
    FormHTML,
    FormRow,
    ResizeHandleRow,
    ToolButton,
)
from scripts import m2m_hook as patches
from scripts import mov2mov
from scripts.m2m_config import mov2mov_outpath_samples, mov2mov_output_dir
from scripts.mov2mov import scripts_mov2mov
from scripts.movie_editor import MovieEditor
from scripts.m2m_ui_common import create_output_panel

id_part = "mov2mov"


def on_ui_settings():
    section = ("mov2mov", "Mov2Mov")
    shared.opts.add_option(
        "mov2mov_outpath_samples",
        shared.OptionInfo(
            mov2mov_outpath_samples, "Mov2Mov output path for image", section=section
        ),
    )
    shared.opts.add_option(
        "mov2mov_output_dir",
        shared.OptionInfo(
            mov2mov_output_dir, "Mov2Mov output path for video", section=section
        ),
    )


img2img_toprow: gr.Row = None


def on_ui_tabs():
    """

    构造ui
    """
    scripts.scripts_current = scripts_mov2mov
    scripts_mov2mov.initialize_scripts(is_img2img=True)
    with gr.TabItem(
        "mov2mov", id=f"tab_{id_part}", elem_id=f"tab_{id_part}"
    ) as mov2mov_interface:
        toprow = ui_toprow.Toprow(
            is_img2img=True, is_compact=shared.opts.compact_prompt_box, id_part=id_part
        )
        dummy_component = gr.Label(visible=False)

        extra_tabs = gr.Tabs(
            elem_id="txt2img_extra_tabs", elem_classes=["extra-networks"]
        )
        extra_tabs.__enter__()

        with gr.Tab(
            "Generation", id=f"{id_part}_generation"
        ) as mov2mov_generation_tab, ResizeHandleRow(equal_height=False):

            with ExitStack() as stack:
                stack.enter_context(
                    gr.Column(variant="compact", elem_id=f"{id_part}_settings")
                )

                for category in ordered_ui_categories():

                    if category == "prompt":
                        toprow.create_inline_toprow_prompts()

                    if category == "image":
                        init_mov = gr.Video(
                            label="Video for mov2mov",
                            elem_id=f"{id_part}_mov",
                            show_label=False,
                            source="upload",
                        )

                        with FormRow():
                            resize_mode = gr.Radio(
                                label="Resize mode",
                                elem_id="resize_mode",
                                choices=[
                                    "Just resize",
                                    "Crop and resize",
                                    "Resize and fill",
                                    "Just resize (latent upscale)",
                                ],
                                type="index",
                                value="Just resize",
                            )

                        scripts_mov2mov.prepare_ui()

                    elif category == "dimensions":
                        with FormRow():
                            with gr.Column(elem_id=f"{id_part}_column_size", scale=4):
                                selected_scale_tab = gr.Number(value=0, visible=False)
                                with gr.Tabs(elem_id=f"{id_part}_tabs_resize"):
                                    with gr.Tab(
                                        label="Resize to",
                                        id="to",
                                        elem_id=f"{id_part}_tab_resize_to",
                                    ) as tab_scale_to:
                                        with FormRow():
                                            with gr.Column(
                                                elem_id=f"{id_part}_column_size",
                                                scale=4,
                                            ):
                                                width = gr.Slider(
                                                    minimum=64,
                                                    maximum=2048,
                                                    step=8,
                                                    label="Width",
                                                    value=512,
                                                    elem_id=f"{id_part}_width",
                                                )
                                                height = gr.Slider(
                                                    minimum=64,
                                                    maximum=2048,
                                                    step=8,
                                                    label="Height",
                                                    value=512,
                                                    elem_id=f"{id_part}_height",
                                                )

                                            with gr.Column(
                                                elem_id=f"{id_part}_dimensions_row",
                                                scale=1,
                                                elem_classes="dimensions-tools",
                                            ):
                                                res_switch_btn = ToolButton(
                                                    value=switch_values_symbol,
                                                    elem_id=f"{id_part}_res_switch_btn",
                                                    tooltip="Switch width/height",
                                                )
                                                detect_image_size_btn = ToolButton(
                                                    value=detect_image_size_symbol,
                                                    elem_id=f"{id_part}_detect_image_size_btn",
                                                    tooltip="Auto detect size from img2img",
                                                )

                                    with gr.Tab(
                                        label="Resize by",
                                        id="by",
                                        elem_id=f"{id_part}_tab_resize_by",
                                    ) as tab_scale_by:
                                        scale_by = gr.Slider(
                                            minimum=0.05,
                                            maximum=4.0,
                                            step=0.05,
                                            label="Scale",
                                            value=1.0,
                                            elem_id=f"{id_part}_scale",
                                        )

                                        with FormRow():
                                            scale_by_html = FormHTML(
                                                resize_from_to_html(0, 0, 0.0),
                                                elem_id=f"{id_part}_scale_resolution_preview",
                                            )
                                            gr.Slider(
                                                label="Unused",
                                                elem_id=f"{id_part}_unused_scale_by_slider",
                                            )
                                            button_update_resize_to = gr.Button(
                                                visible=False,
                                                elem_id=f"{id_part}_update_resize_to",
                                            )

                                    on_change_args = dict(
                                        fn=resize_from_to_html,
                                        _js="currentMov2movSourceResolution",
                                        inputs=[
                                            dummy_component,
                                            dummy_component,
                                            scale_by,
                                        ],
                                        outputs=scale_by_html,
                                        show_progress=False,
                                    )

                                    scale_by.release(**on_change_args)
                                    button_update_resize_to.click(**on_change_args)

                                tab_scale_to.select(
                                    fn=lambda: 0,
                                    inputs=[],
                                    outputs=[selected_scale_tab],
                                )
                                tab_scale_by.select(
                                    fn=lambda: 1,
                                    inputs=[],
                                    outputs=[selected_scale_tab],
                                )

                    elif category == "denoising":
                        denoising_strength = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            label="Denoising strength",
                            value=0.75,
                            elem_id=f"{id_part}_denoising_strength",
                        )
                        noise_multiplier = gr.Slider(
                            minimum=0,
                            maximum=1.5,
                            step=0.01,
                            label="Noise multiplier",
                            elem_id=f"{id_part}_noise_multiplier",
                            value=1,
                        )
                        with gr.Row(elem_id=f"{id_part}_frames_setting"):
                            movie_frames = gr.Slider(
                                minimum=10,
                                maximum=60,
                                step=1,
                                label="Movie FPS",
                                elem_id=f"{id_part}_movie_frames",
                                value=30,
                            )
                            max_frames = gr.Number(
                                label="Max FPS",
                                value=-1,
                                elem_id=f"{id_part}_max_frames",
                            )

                    elif category == "cfg":
                        with gr.Row():
                            cfg_scale = gr.Slider(
                                minimum=1.0,
                                maximum=30.0,
                                step=0.5,
                                label="CFG Scale",
                                value=7.0,
                                elem_id=f"{id_part}_cfg_scale",
                            )
                            image_cfg_scale = gr.Slider(
                                minimum=0,
                                maximum=3.0,
                                step=0.05,
                                label="Image CFG Scale",
                                value=1.5,
                                elem_id=f"{id_part}_image_cfg_scale",
                                visible=False,
                            )

                    elif category == "checkboxes":
                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            pass

                    elif category == "accordions":
                        with gr.Row(
                            elem_id=f"{id_part}_accordions", elem_classes="accordions"
                        ):
                            scripts_mov2mov.setup_ui_for_section(category)

                    elif category == "override_settings":
                        with FormRow(elem_id=f"{id_part}_override_settings_row") as row:
                            override_settings = create_override_settings_dropdown(
                                id_part, row
                            )

                    elif category == "scripts":
                        editor = MovieEditor(id_part, init_mov, movie_frames)
                        editor.render()
                        with FormGroup(elem_id=f"{id_part}_script_container"):
                            custom_inputs = scripts_mov2mov.setup_ui()

                    if category not in {"accordions"}:
                        scripts_mov2mov.setup_ui_for_section(category)

            output_panel = create_output_panel(id_part, opts.mov2mov_output_dir)
            mov2mov_args = dict(
                fn=wrap_gradio_gpu_call(mov2mov.mov2mov, extra_outputs=[None, "", ""]),
                _js="submit_mov2mov",
                inputs=[
                    dummy_component,
                    dummy_component,
                    toprow.prompt,
                    toprow.negative_prompt,
                    toprow.ui_styles.dropdown,
                    init_mov,
                    cfg_scale,
                    image_cfg_scale,
                    denoising_strength,
                    selected_scale_tab,
                    height,
                    width,
                    scale_by,
                    resize_mode,
                    override_settings,
                    # refiner
                    # enable_refiner, refiner_checkpoint, refiner_switch_at,
                    # mov2mov params
                    noise_multiplier,
                    movie_frames,
                    max_frames,
                    # editor
                    editor.gr_enable_movie_editor,
                    editor.gr_df,
                    editor.gr_eb_weight,
                ]
                + custom_inputs,
                outputs=[
                   
                    output_panel.video,
                    output_panel.infotext,
                    output_panel.html_log,
                ],
                show_progress=False,
            )

            toprow.prompt.submit(**mov2mov_args)
            toprow.submit.click(**mov2mov_args)

            res_switch_btn.click(
                fn=None,
                _js="function(){switchWidthHeight('mov2mov')}",
                inputs=None,
                outputs=None,
                show_progress=False,
            )
            detect_image_size_btn.click(
                fn=lambda w, h, _: (w or gr.update(), h or gr.update()),
                _js="currentMov2movSourceResolution",
                inputs=[dummy_component, dummy_component, dummy_component],
                outputs=[width, height],
                show_progress=False,
            )

        extra_tabs.__exit__()
        scripts.scripts_current = None

        return [(mov2mov_interface, "mov2mov", f"{id_part}_tabs")]


def block_context_init(self, *args, **kwargs):
    origin_block_context_init(self, *args, **kwargs)

    if self.elem_id == "tab_img2img":
        self.parent.__enter__()
        on_ui_tabs()
        self.parent.__exit__()


def on_app_reload():
    global origin_block_context_init
    if origin_block_context_init:
        patches.undo(__name__, obj=gr.blocks.BlockContext, field="__init__")
        origin_block_context_init = None


origin_block_context_init = patches.patch(
    __name__,
    obj=gr.blocks.BlockContext,
    field="__init__",
    replacement=block_context_init,
)
script_callbacks.on_before_reload(on_app_reload)
script_callbacks.on_ui_settings(on_ui_settings)
# script_callbacks.on_ui_tabs(on_ui_tabs)
