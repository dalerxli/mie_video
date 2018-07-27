'''Module for localizing particle trajectories with tensorflow tracking.'''

import numpy as np
import pandas as pd
import trackpy as tp
import cv2
from tracker import tracker
from mie_video.editing import inflate
import pylab as pl
from matplotlib import animation
from matplotlib.patches import Rectangle


def localize(video, background=None, dark_count=31):
    '''
    Returns DataFrame of particle parameters in each frame
    of a video linked with their trajectory index
    
    Args:
        video: video filename
    Keywords:
        background: background image for normalization
        dark_count: dark count of camera
    '''
    trk = tracker.tracker()
    # Create VideoCapture to read video
    cap = cv2.VideoCapture(video)
    # Initialize components to build overall dataset.
    frame_no = 0
    data = []
    while(cap.isOpened()):
        ret, image = cap.read()
        if ret is False:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Normalize image
        if background is not None:
            image = (image - dark_count) / (background - dark_count)
        # Find features in each frame
        features = trk.predict(inflate(image))
        for feature in features:
            # Build dataset over all frames.
            feature = np.append(feature, frame_no)
            data.append(feature)
        # Advance frame_no
        frame_no += 1
    cap.release()
    # Put data set in DataFrame and link
    result_df = pd.DataFrame(columns=['x', 'y', 'w', 'h', 'frame'], data=data)
    linked_df = tp.link(result_df, search_range=5, pos_columns=['y', 'x'])
    return linked_df


def separate(trajectories):
    '''
    Returns list of separated DataFrames for each particle
    
    Args:
        trajectories: Pandas DataFrame linked by trackpy.link(df)
    '''
    result = []
    for idx in range(trajectories.particle.max() + 1):
        result.append(trajectories[trajectories.particle == idx])
    return result


class Animate(object):
    """Creates an animated video of particle tracking
    """

    def __init__(self, video, dest='animation/test_mpl_anim.avi', **kwargs):
        self.video = video
        self.dest = dest
        self.fig, self.ax = pl.subplots(figsize=(8, 6))
        self.ax.set_xlabel('X [pixel]')
        self.ax.set_ylabel('Y [pixel]')
        self.ax.set_ylim([0, 479])
        self.cap = cv2.VideoCapture(self.video)
        self.im = None
        self.trk = tracker.tracker()

    def run(self):
        ret, frame = self.cap.read()
        if ret:
            self.im = self.ax.imshow(frame, interpolation='none',
                                     cmap=pl.get_cmap('gray'))
            self.anim = animation.FuncAnimation(self.fig,
                                                self.anim, init_func=self.init,
                                                blit=True, interval=50)
            self.anim.save(self.dest)
        else:
            print("Failed")

    def init(self):
        ret = False
        while not ret:
            ret, frame = self.cap.read()
        self.im.set_data(frame)
        return self.im,

    def anim(self, i):
        ret, frame = self.cap.read()
        if ret:
            features = self.trk.predict(frame)
            for feature in features:
                x, y, w, h = feature
            rect = Rectangle(xy=(x - w/2, y - h/2), width=w, height=h,
                             fill=False, linewidth=3, edgecolor='r')
            self.ax.add_patch(rect)
        self.im.set_array(frame)
        return self.im,


if __name__ == '__main__':
    video = 'animation/example.avi'
    anim = Animate(video)
    anim.run()
