import gradio
from modules import script_callbacks, ui_components
from scripts import m2m_hook as patches


elem_ids = []


def fix_elem_id(component, **kwargs):
    if "elem_id" not in kwargs:
        return None
    elem_id = kwargs["elem_id"]
    if not elem_id:
        return None
    if elem_id not in elem_ids:
        elem_ids.append(elem_id)
    else:
        elem_id = elem_id + "_" + str(elem_ids.count(elem_id))
        elem_ids.append(elem_id)

    return elem_id


def IOComponent_init(self, *args, **kwargs):
    elem_id = fix_elem_id(self, **kwargs)
    if elem_id:
        kwargs.pop("elem_id")
        res = original_IOComponent_init(self, elem_id=elem_id, *args, **kwargs)
    else:
        res = original_IOComponent_init(self, *args, **kwargs)
    return res


def InputAccordion_init(self, *args, **kwargs):
    elem_id = fix_elem_id(self, **kwargs)
    if elem_id:
        kwargs.pop("elem_id")
        res = original_InputAccordion_init(self, elem_id=elem_id, *args, **kwargs)
    else:
        res = original_InputAccordion_init(self, *args, **kwargs)
    return res


original_IOComponent_init = patches.patch(
    __name__,
    obj=gradio.components.IOComponent,
    field="__init__",
    replacement=IOComponent_init,
)

original_InputAccordion_init = patches.patch(
    __name__,
    obj=ui_components.InputAccordion,
    field="__init__",
    replacement=InputAccordion_init,
)


def on_before_reload():
    elem_ids.clear()

    global original_IOComponent_init
    if original_IOComponent_init:
        patches.undo(__name__, obj=gradio.components.IOComponent, field="__init__")
        original_IOComponent_init = None

    global original_InputAccordion_init
    if original_InputAccordion_init:
        patches.undo(__name__, obj=ui_components.InputAccordion, field="__init__")
        original_InputAccordion_init = None


script_callbacks.on_before_reload(on_before_reload)
