import pandas

from modules import scripts
from modules.generation_parameters_copypaste import create_override_settings_dict

from .processing import StableDiffusionProcessingMov2Mov

scripts_mov2mov = scripts.ScriptRunner()
scripts_preprocess1 = scripts.ScriptRunner()
scripts_preprocess2 = scripts.ScriptRunner()


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
            movie_frames,
            max_frames,
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


    print(1111111)
