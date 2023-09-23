# Fix ui library to support custom tabs
from modules import patches, shared, ui_prompt_styles
import modules
import gradio as gr

from modules.ui import paste_symbol, clear_prompt_symbol, restore_progress_symbol
from modules.ui_components import ToolButton



def Toprow_init(self, is_img2img, id_part=None):
    if not id_part:
        original_Toprow_init(self, is_img2img)
    else:
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

original_Toprow_init = patches.patch(__name__, obj=modules.ui.Toprow, field="__init__", replacement=Toprow_init)
