'''Class to fit trajectories in a video to Lorenz-Mie theory.'''

from mie_video.tracking import localize, separate
from mie_video.editing import crop
import numpy as np
import pandas as pd
from lorenzmie.theory import spheredhm
from lorenzmie.fitting.mie_fit import Mie_Fitter
import cv2
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import time


class Video_Fitter(object):

    def __init__(self, guesses, fn, fixed=['n_m', 'mpp', 'lamb'],
                 background=None, dark_count=31):
        self.background = background
        self.dark_count = dark_count
        self._params = OrderedDict(zip(['x', 'y', 'z', 'a_p',
                                        'n_p', 'n_m', 'mpp', 'lamb'], guesses))
        self.fn = os.path.expanduser(fn)
        self.fitter = Mie_Fitter(self.params, fixed=fixed)
        self.linked_df = localize(self.fn, background=self.background)
        self.trajectories = separate(self.linked_df)
        self.fit_df = None

    @property
    def params(self):
        '''
        Returns OrderedDict of parameters x, y, z, a_p, n_p, n_m, mpp, lamb
        '''
        return self._params

    @params.setter
    def params(self, guesses):
        '''
        Sets parameters for fitter

        Args:
            guesses: list of parameters ordered [x, y, z, a_p, n_p, n_m, mpp, lamb]
        '''
        new_params = OrderedDict(zip(['x', 'y', 'z', 'a_p', 'n_p',
                                      'n_m', 'mpp', 'lamb'], guesses))
        for key in self.params.keys():
            if key == 'x' or key == 'y':
                self.fitter.set_param(key, 0.0)
            else:
                self.fitter.set_param(key, new_params[key])
            self._params = self.fitter.p.valuesdict()

    def fit(self, trajectory):
        '''
        Sets DataFrame of fitted parameters in each frame
        for a given trajectory.
        
        Args:
            trajectory: index of particle trajectory in self.trajectories.
        '''
        p_df = self.trajectories[trajectory]
        cap = cv2.VideoCapture(self.fn)
        frame_no = 0
        data = {}
        for key in self.params:
            data[key] = []
            data['frame'] = []
            data['redchi'] = []
        while(cap.isOpened()):
            ret, image = cap.read()
            if ret is False:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Normalize image
            if self.background is not None:
                image = (image - self.dark_count) / (self.background -
                                                     self.dark_count)
            # Find the cropped image around particle.
            x, y, w, h, frame, particle = p_df.iloc[frame_no, :]
            if frame != frame_no:
                print('ERROR: Frame numbers don\'t match.')
            cropped_image = crop(image, x, y, w, h)
            # Fit frame
            start = time.time()
            fit = self.fitter.fit(cropped_image)
            fit_time = time.time() - start
            print(self.fn[-7:-4] + " time to fit frame " + str(frame_no) +
                  ": " + str(fit_time))
            print("Fit RedChiSq: " + str(fit.redchi))
            # Add fit to dataset
            for key in data.keys():
                if key == 'frame':
                    data[key].append(frame_no)
                elif key == 'redchi':
                    data[key].append(fit.redchi)
                else:
                    data[key].append(fit.params[key].value)
            frame_no += 1
            # Set guesses for next fit
            guesses = []
            for param in fit.params.values():
                guesses.append(param.value)
            self.params = guesses
        cap.release()
        self.fit_df = pd.DataFrame(data=data)

    def test(self, guesses, trajectory=0, frame_no=0):
        '''
        Plot guessed image vs. image of a trajectory at a given frame
        
        Args:
            guesses: list of parameters ordered
                     [x, y, z, a_p, n_p, n_m, mpp, lamb]
        Keywords:
            trajectory: index of trajectory in self.trajectory
            frame_no: index of frame to test
        Returns:
            Raw frame from camera
        '''
        p_df = self.trajectories[trajectory]
        if frame_no > max(p_df.index) or frame_no < min(p_df.index):
            raise(IndexError("Trajectory not found in frame {} for particle {}"
                             .format(frame_no, trajectory)))
        cap = cv2.VideoCapture(self.fn)
        cap.set(1, frame_no)
        ret, frame = cap.read()
        if not ret:
            print("Frame not read.")
            return
        frame = frame[:, :, 0]
        x, y, w, h, frame_no, particle = p_df.iloc[0, :]
        cropped_image = crop(frame, x, y, w, h)
        x, y, z, a_p, n_p, n_m, mpp, lamb = guesses
        image = spheredhm.spheredhm([0, 0, z],
                                    a_p, n_p, n_m,
                                    dim=cropped_image.shape,
                                    lamb=lamb, mpp=mpp)
        plt.imshow(np.hstack([cropped_image, image]), cmap='gray')
        plt.show()
        cap.release()
        return frame
