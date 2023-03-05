import modules.img2img
from modules import script_callbacks
from modules.processing import process_images

img2img_orig = modules.img2img.img2img


class M2M_Hook:
    @staticmethod
    def install(func):
        modules.img2img.img2img = func

    @staticmethod
    def uninstall():
        modules.img2img.img2img = img2img_orig

