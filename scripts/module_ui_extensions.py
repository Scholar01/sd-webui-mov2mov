import modules
from modules import patches, script_callbacks

from modules.processing_scripts.refiner import ScriptRefiner


# fix refiner
def on_app_reload():
    global origin_refiner_ui
    if origin_refiner_ui:
        patches.undo(__name__, obj=modules.processing_scripts.refiner.ScriptRefiner, field="ui")
        origin_refiner_ui = None


def refiner_ui(self, is_img2img):
    """Fix the problem that refiner does not take effect in mov2mov tab"""
    print("refiner_ui", is_img2img)
    # global refiner_img2img_index
    # if is_img2img:
    #     refiner_img2img_index += 1
    #
    # if refiner_img2img_index >= 1:
    #     def get_elem_id(item_id):
    #         return 'mov2mov_' + item_id
    #
    #     self.elem_id = get_elem_id

    return origin_refiner_ui(self, is_img2img)


origin_refiner_ui = patches.patch(__name__, obj=modules.processing_scripts.refiner.ScriptRefiner,
                                  field="ui",
                                  replacement=refiner_ui)
refiner_img2img_index = 0

script_callbacks.on_before_reload(on_app_reload)
