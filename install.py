import launch

if not launch.is_installed("cv2"):
    print('Installing requirements for Mov2mov')
    launch.run_pip("install opencv-python", "requirements for opencv")
    launch.run_pip("install imageio", "requirements for imageio")
    launch.run_pip("install imageio[ffmpeg]", "requirements for imageio[ffmpeg]")

if not launch.is_installed('ffmpeg'):
    print('Installing requirements for Mov2mov')
    launch.run_pip("install ffmpeg", "requirements for ffmpeg")
    launch.run_pip("install install imageio-ffmpeg", "requirements for imageio-ffmpeg")
