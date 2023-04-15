import launch
import platform


plat = platform.system().lower()


if not launch.is_installed("cv2"):
    print('Installing requirements for Mov2mov')
    launch.run_pip("install opencv-python", "requirements for opencv")
    if plat == 'linux':
        launch.run_pip("install imageio", "requirements for imageio")


if not launch.is_installed('ffmpeg'):
    print('Installing requirements for Mov2mov')
    launch.run_pip("install ffmpeg", "requirements for ffmpeg")
    if plat == 'linux':
        launch.run_pip("install install imageio-ffmpeg", "requirements for imageio-ffmpeg")
