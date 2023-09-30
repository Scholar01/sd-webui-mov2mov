import os.path
import platform
import cv2
import numpy
import imageio


def calc_video_w_h(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Can't open video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return width, height


def get_mov_frame_count(file):
    if file is None:
        return None
    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        return None

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames


def get_mov_fps(file):
    if file is None:
        return None
    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def get_mov_all_images(file, frames, rgb=False):
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
                if rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_list.append(frame)
                count += 1
        fs += 1
    cap.release()
    return image_list


def images_to_video(images, frames, out_path):
    if platform.system() == 'Windows':
        return images_to_video_win(images, frames, out_path)
    else:
        return images_to_video_mac(images, frames, out_path)


def images_to_video_win(images, frames, out_path):
    # 判断out_path是否存在,不存在则创建
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with imageio.v2.get_writer(out_path, format='ffmpeg', mode='I', fps=frames, codec='libx264') as writer:
        for img in images:
            writer.append_data(numpy.asarray(img))
    return out_path


def images_to_video_mac(images, frames, out_path):
    if len(images) <= 0:
        return None
    # 判断out_path是否存在,不存在则创建
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
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
