import os.path

import cv2


def video_to_images(frames, video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    judge = cap.isOpened()
    if not judge:
        raise ValueError("Can't open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if frames > fps:
        frames = fps

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
    return frames


def images_to_video(frames, w, h, in_path, out_path):
    images = os.listdir(in_path)
    images = [file for file in images if file.endswith('.jpg')]
    if len(images) == 0:
        raise FileNotFoundError('not images')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(out_path, fourcc, frames, (w, h))

    images.sort(key=lambda x: int(x.replace('.jpg', '').replace('jpgs_', '')))

    for image in images:
        p = os.path.join(in_path, image)
        img = cv2.imread(p)
        video.write(img)
        del img
    video.release()
    return out_path