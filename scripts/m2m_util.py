import os.path

import cv2
import numpy


def video_to_images(frames, video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    judge = cap.isOpened()
    if not judge:
        raise ValueError("Can't open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if frames > fps:
        frames = int(fps)

    skip = fps // frames
    count = 1
    fs = 1

    while (judge):
        flag, frame = cap.read()
        if not flag:
            break
        else:
            if fs % skip == 0:
                imgname = 'jpgs_' + str(count).rjust(3, '0') + ".jpg"
                newPath = os.path.join(out_path, imgname)
                cv2.imwrite(newPath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                count += 1
        fs += 1
    cap.release()
    return frames, count


def get_mov_all_images(file, frames):
    if file is None:
        return None
    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if frames > fps:
        print('Waring: The set number of frames is greater than the number of video frames')
        frames = int(fps)

    skip = fps // frames
    count = 1
    fs = 1
    image_list = []
    while (True):
        flag, frame = cap.read()
        if not flag:
            break
        else:
            if fs % skip == 0:
                image_list.append(frame)
                count += 1
        fs += 1
    cap.release()
    return image_list


def images_to_video(images, frames, mode, w, h, out_path):
    # 判断out_path是否存在,不存在则创建
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*mode)
    if len(images) > 0:
        img = images[0]
        img_width, img_height = img.size
        w = img_width
        h = img_height
    video = cv2.VideoWriter(out_path, fourcc, frames, (w, h))
    for image in images:
        img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
        video.write(img)
    video.release()
    return out_path
