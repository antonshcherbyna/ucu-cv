import cv2
import os

import matplotlib.pyplot as plt
from ipywidgets import interact, widgets


def get_frames(path): 
    for root, dir, files in os.walk(path):
        img_paths = [os.path.join(root, file) for file in files]

    try:
        img_paths = sorted(img_paths, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    except:
        pass

    frames = []
    gray_frames = []

    for img_path in img_paths:
        
        # read image
        frame = cv2.imread(img_path)
        
        # get original image in RGB for visualization
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # get grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        frames.append(frame)
        gray_frames.append(gray)

    return frames, gray_frames


def show_single_frame(frame, title):
    fig, ax = plt.subplots(figsize=(10, 10))
    _ = ax.axis('off')
    _ = ax.imshow(frame)
    _ = ax.set_title(title)
    plt.show()


def show_multiply_frames(frames):
    def show_single_frame(frame_num):
        frame = frames[frame_num]
        fig, ax = plt.subplots(figsize=(10, 10))
        _ = ax.axis('off')
        _ = ax.imshow(frame)
        plt.show()
    interact(show_single_frame, frame_num=widgets.IntSlider(min=0, max=len(frames) - 1, step=1, value=0))


def draw_rois(frames, rois):
    if len(rois[0]) == 2:
        frames_with_rois = []
        for roi, frame in zip(rois, frames):
            frame_with_roi = cv2.rectangle(frame.copy(), roi[0], roi[1], (0, 0, 255), 2)
            frames_with_rois.append(frame_with_roi)
    elif len(rois[0]) == 4:
        frames_with_rois = []
        for roi, frame in zip(rois, frames):
            poly = cv2.convexHull(roi).reshape(1, -1, 2)
            frame_with_roi = cv2.polylines(frame.copy(), poly, True, (0, 0, 255), 2)
            frames_with_rois.append(frame_with_roi)
    else:
        raise(Exception('Invalid shapes for rois!'))

    return frames_with_rois
