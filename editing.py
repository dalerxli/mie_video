'''Module for miscellaneous image operations'''

import cv2
import numpy as np
import os


def background(f, file_type=".avi", folder=False,
               shape=(640, 480)):
    '''Randomly grab 100 frames from one video or a
    folder of videos to get a normalized image
    '''
    n_frames = 100
    frames = np.zeros((n_frames, shape[0], shape[1]),
                      dtype=np.float_)
    if not folder:
        file = f
        count = count_frames(file)
        for idx in range(n_frames):
            rand_idx = np.random.choice(range(count))
            cap = cv2.VideoCapture(file)
            cap.set(1, rand_idx)
            ret, rand_frame = cap.read()
            i = 0
            while ret is False:
                if i == 10:
                    print("Failed to read {} at frame {}".
                          format(file, rand_idx))
                    break
                ret, rand_frame = cap.read()
                i += 1
            frames[idx] = rand_frame
    else:
        folder = f
        directory = sorted(os.listdir(folder))
        for idx in range(n_frames):
            file = np.random.choice(directory)
            while file.endswith(file_type) is False:
                file = np.random.choice(directory)
            path = os.path.join(folder, file)
            count = count_frames(path)
            rand_idx = np.random.choice(range(count))
            cap = cv2.VideoCapture(path)
            cap.set(1, rand_idx)
            ret, rand_frame = cap.read()
            idx = 0
            while ret is False:
                ret, rand_frame = cap.read()
                idx += 1
                if idx == 10:
                    print("Failed to read {} at frame {}".
                          format(file, rand_idx))
                    break
            frames[idx] = rand_frame
    background = np.median(frames, axis=0)
    dark_count = 31
    return background, dark_count


def count_frames(path):
    '''Gets the number of frames in a video
    '''
    video = cv2.VideoCapture(path)
    total = 0
    if cv2.__version__.startswith('3.'):
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        total = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    return total


def inflate(image):
    '''
    Returns a BW image from a RGB image.
    '''
    shape = image.shape
    new_image = np.zeros([shape[0], shape[1], 3])
    new_image[:, :, 0] = image
    new_image[:, :, 1] = image
    new_image[:, :, 2] = image
    return new_image


def crop(image, xc, yc, w, h):
    '''
    Returns a square, cropped image.
    
    Args:
        image: image to be cropped
        xc: x coordinate of crop's top right corner
        yc: y coordinate of crop's top right corner
        w: width of crop
        h: height of crop
    '''
    cropped_image = image[yc - h//2: yc + h//2,
                          xc - w//2: xc + w//2].astype(float)
    cropped_image /= np.mean(cropped_image)
    xdim, ydim = cropped_image.shape
    if xdim == ydim:
        return cropped_image
    if xdim > ydim:
        cropped_image = cropped_image[1:-1, :]
    else:
        cropped_image = cropped_image[:, 1:-1]
    return cropped_image
