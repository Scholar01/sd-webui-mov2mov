import os

import cv2
import gradio as gr
from PIL import Image

import modules.scripts as scripts
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from scripts.modnet.modnet import MODNet

"""
from modnet-entry("https://github.com/RimoChan/modnet-entry")
"""

script_path = scripts.basedir()
models_path = os.path.join(script_path, "models")
modnet_models = ['none'] + [model for model in os.listdir(models_path) if model.endswith('.ckpt')]


def get_model(ckpt_name):
    ckpt_path = os.path.join(models_path, ckpt_name)
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(ckpt_path)
    else:
        weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    modnet.eval()
    return modnet


def create_modnet():
    ctrls = ()
    with gr.Group():
        with gr.Accordion("ModNet", open=True):
            background_image = gr.Image(label='Background', type='numpy', elem_id='modnet_background_image').style()
            background_movie = gr.Video(label='Background', elem_id='modnet_background_movie').style()
            enable = gr.Checkbox(label='Enable', value=False, )
            ctrls += (background_image, background_movie, enable)
            with gr.Row():
                mode = gr.Radio(label='Mode', choices=[
                    'Image', 'Movie'
                ], type='index', value='Image')
                guidance = gr.Radio(label='Guidance', choices=[
                    'Start', 'End'
                ], type='index', value='Start')
                ctrls += (mode, guidance)

            movie_frames = gr.Slider(minimum=10,
                                     maximum=60,
                                     step=1,
                                     label='Movie Frames',
                                     elem_id='modnet_movie_frames',
                                     value=30)
            ctrls += (movie_frames,)
            with gr.Row():
                models = gr.Dropdown(label='Model', choices=list(modnet_models), value='none')
                ctrls += (models,)

            with gr.Row():
                resize_mode = gr.Radio(label="Resize mode",
                                       choices=["Just resize", "Crop and resize", "Resize and fill",
                                                ], type="index", value="Just resize")
                ctrls += (resize_mode,)

    mode.change(fn=None, inputs=[mode], outputs=[], _js=f'switchModnetMode')
    return ctrls


def infer(modnet, im, ref_size=1024):
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # unify image channels to 3
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # convert image to PyTorch tensor
    im = Image.fromarray(im)
    im = im_transform(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    # resize image for input
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    # inference
    _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    return matte


def infer2(modnet, img):
    image = np.asarray(img)
    h, w, _ = image.shape
    alpha = infer(modnet, image, max(h, w))
    alpha_bool = (~alpha.astype(np.bool)).astype('int')

    alpha_uint8 = (alpha * 255).astype('uint8')
    new_image = np.concatenate((image, alpha_uint8[:, :, None]), axis=2)
    return Image.fromarray(new_image, 'RGBA'), Image.fromarray(alpha_uint8, mode='L')
