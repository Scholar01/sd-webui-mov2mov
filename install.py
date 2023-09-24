import os
import platform
import launch

if not launch.is_installed("cv2"):
    print('Installing requirements for Mov2mov')
    launch.run_pip("install opencv-python", "requirements for opencv")

if platform.system() == 'Windows':
    if not launch.is_installed('imageio'):
        print('Installing requirements for Mov2mov')
        launch.run_pip("install imageio", "requirements for imageio")
    if not launch.is_installed('imageio-ffmpeg'):
        print('Installing requirements for Mov2mov')
        launch.run_pip("install imageio-ffmpeg", "requirements for imageio-ffmpeg")
else:
    if not launch.is_installed('ffmpeg'):
        print('Installing requirements for Mov2mov')
        launch.run_pip("install ffmpeg", "requirements for ffmpeg")
