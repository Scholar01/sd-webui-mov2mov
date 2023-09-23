import os

import launch


requirements = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
with open(requirements) as f:
    for module in f:
        module = module.strip()
        if not launch.is_installed(module):
            launch.run_pip(f"install {module}", f"requirement for mov2mov: {module}")
