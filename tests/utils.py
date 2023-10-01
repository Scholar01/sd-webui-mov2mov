import sys

import os


def setup_test_env():
    ext_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if ext_root not in sys.path:
        sys.path.append(ext_root)

    project_root = os.path.dirname(os.path.dirname(ext_root))
    if project_root not in sys.path:
        sys.path.append(project_root)

    modules_root = os.path.join(project_root, 'modules')
    if modules_root not in sys.path:
        sys.path.append(modules_root)