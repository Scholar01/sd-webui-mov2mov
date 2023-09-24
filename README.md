[English](README.md) | [中文简体](README_CN.md)

## Mov2mov

![img.png](images/2.jpg)

This is the Mov2mov plugin for Automatic1111/stable-diffusion-webui.

Features:
- Directly process frames from videos
- Package into a video after processing
- ~~Video matting, synthesis, and other preprocessing and postprocessing~~
  - ~~Extract human figures~~
  - ~~Synthesize with a transparent background~~
  - ~~Synthesize with the original background~~
  - ~~Synthesize with a green screen~~
  - ~~Synthesize with a specified image background~~
  - ~~Synthesize with a specified video background~~
- ~~Frame-by-frame processing prompt and negative_prompt~~
  - ~~Use *frame number:prompt|| to mark *start ||end~~
  - ~~*1:1girl||*100:2girl|| Use 1girl for the first frame to the 99th frame, then use 2girl from the 100th frame to the end.~~
  - ~~The same applies to negative_prompt.~~

## Installation

1. Open the Extensions tab.
2. Click on Install from URL.
3. Enter the URL for the extension's git repository.
4. Click Install.
5. Restart WebUI.

## Usage Regulations

1. Please resolve the authorization issues of the video source on your own. Any problems caused by using unauthorized videos for conversion must be borne by the user. It has nothing to do with mov2mov!
2. Any video made with mov2mov and published on video platforms must clearly specify the source of the video used for conversion in the description. For example, if you use someone else's video and convert it through AI, you must provide a clear link to the original video; if you use your own video, you must also state this in the description.
3. All copyright issues caused by the input source must be borne by the user. Note that many videos explicitly state that they cannot be reproduced or copied!
4. Please strictly comply with national laws and regulations to ensure that the content is legal and compliant. Any legal responsibility caused by using this plugin must be borne by the user. It has nothing to do with mov2mov!

## Notes

- The packaged video is located in the `outputs/mov2mov-images/` directory.
- You may need to install opencv.
- ~~The directory cannot contain Chinese characters!!!~~

## Update Log

[Update Log](CHANGELOG.md)



## Instructions

- Video tutorials:
  - [https://www.bilibili.com/video/BV1Mo4y1a7DF](https://www.bilibili.com/video/BV1Mo4y1a7DF)
  - [https://www.bilibili.com/video/BV1rY4y1C7Q5](https://www.bilibili.com/video/BV1rY4y1C7Q5)
- QQ channel: [https://pd.qq.com/s/akxpjjsgd](https://pd.qq.com/s/akxpjjsgd)
- Discord: [https://discord.gg/hUzF3kQKFW](https://discord.gg/hUzF3kQKFW)

## Thanks

- modnet-entry: [https://github.com/RimoChan/modnet-entry](https://github.com/RimoChan/modnet-entry)
- MODNet: [https://github.com/ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet)
