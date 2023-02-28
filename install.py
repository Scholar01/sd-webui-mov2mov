import launch

print('Installing requirements for Mov2mov')
if not launch.is_installed("cv2"):
    launch.run_pip("install opencv-python", "requirements for opencv")
