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
import lab_io.h5video as h5
import features.circletransform as ct
from time import time


def localize(video, method='tf', background=None, dark_count=31):
    '''
    Returns DataFrame of particle parameters in each frame
    of a video linked with their trajectory index
    
    Args:
        video: video filename
    Keywords:
        background: background image for normalization
        dark_count: dark count of camera
    '''
    if method == 'tf':
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
        if method == 'tf':
            features = trk.predict(inflate(image))
        elif method == 'oat':
            features, circ = oat(image, frame_no)
        else:
            raise(ValueError("method must be either \'oat\' or \'tf\'"))
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


def oat(frame, frame_no):
    '''
    Use the orientational alignment transform 
    on every pixel of an image and return features.'''
    t = time()
    circ = ct.circletransform(frame, theory='orientTrans')
    circ = h5.TagArray(circ, frame_no)
    features = tp.locate(circ, 31, minmass=5.0, engine='numba')
    features['w'] = 301
    features['h'] = 301
    features = np.array(features[['x', 'y', 'w', 'h']])
    print("Time to find {} features at frame {}: {}".format(features.shape[0],
                                                            , frame_no
                                                            time() - t))
    return features, circ


def separate(trajectories):
    '''
    Returns list of separated DataFrames for each particle
    
    Args:
        trajectories: Pandas DataFrame linked by trackpy.link(df)
    '''
    result = []
    for idx in range(int(trajectories.particle.max()) + 1):
        result.append(trajectories[trajectories.particle == idx])
    return result


class Animate(object):
    """Creates an animated video of particle tracking
    """

    def __init__(self, video, method='oat', transform=True,
                 dest='animation/test_mpl_anim_oat.avi', **kwargs):
        self.frame_no = 0
        self.transform = transform
        self.video = video
        self.dest = dest
        self.fig, self.ax = pl.subplots(figsize=(8, 6))
        self.ax.set_xlabel('X [pixel]')
        self.ax.set_ylabel('Y [pixel]')
        self.cap = cv2.VideoCapture(self.video)
        self.im = None
        self.method = method
        self.rects = None
        if self.method == 'tf':
            self.trk = tracker.tracker()

    def run(self):
        ret, frame = self.cap.read()
        if self.transform:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            features, frame = oat(frame, self.frame_no)
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
            if self.method == 'tf':
                features = self.trk.predict(frame)
            elif self.method == 'oat':
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                features, frame_ct = oat(frame_gray, self.frame_no)
            else:
                raise(ValueError("method must be either \'oat\' or \'tf\'"))
            if self.rects is not None:
                for rect in self.rects:
                    rect.remove()
            self.rects = []
            for feature in features:
                x, y, w, h = feature
                rect = Rectangle(xy=(x - w/2, y - h/2),
                                 width=w, height=h,
                                 fill=False, linewidth=3, edgecolor='r')
                self.rects.append(rect)
                self.ax.add_patch(rect)
        if self.transform:
            self.im.set_array(frame_ct)
        else:
            self.im.set_array(frame)
        self.frame_no += 1
        return self.im,


if __name__ == '__main__':
    import sys
    args = sys.argv
    anim = Animate(args[1], dest=args[2])
    anim.run()
