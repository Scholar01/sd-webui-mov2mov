### 2023/9/30
1. automatic video fps parsing
2. Video editing features.
   1. Customizable selection of keyframes or automatic generation of keyframes.
   2. Backtrack keyframe tag.
   3. automatically synthesize video based on keyframes via Ezsynth(https://github.com/Trentonom0r3/Ezsynth).
   4. Currently, only the Windows system is supported. If your system does not support it, you can close this tab.

### 2023/9/24
1. Move the tab behind img2img.
2. Fix the issue of video synthesis failure on the Mac system.
3. Fix the problem of refiner not taking effect

### 2023/9/23
1. Fixed the issue where the tab is not displayed in the sd1.6 version.
2. Inference of video width and height.
3. Support for Refiner.
4. Temporarily removed modnet functionality.
5. Temporarily removed the function to add prompts frame by frame (ps: I believe there's a better approach, will add in the next version).
6. Changed video synthesis from ffmpeg to imageio.
