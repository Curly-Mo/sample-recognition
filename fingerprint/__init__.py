import logging
import numpy as np

from .sift import Sift


logger = logging.getLogger(__name__)


class Fingerprint(object):
    def __init__(self, audio_path, sr, settings, implementation='sift'):
        self.audio_path = audio_path
        self.sr = sr
        self.settings = settings
        if implementation == 'sift':
            sift = Sift(audio_path, sr, settings)
            self.keypoints = sift.keypoints
            self.descriptors = sift.descriptors
            self.spectrogram = sift.spectrogram
        else:
            raise NotImplementedError

    def remove_similar_keypoints(self):
        if len(self.descriptors) > 0:
            logger.info('Removing duplicate/similar keypoints...')
            a = np.array(self.descriptors)
            rounding_factor = 10
            b = np.ascontiguousarray((a//rounding_factor)*rounding_factor).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
            _, idx = np.unique(b, return_index=True)
            desc = a[sorted(idx)]
            kp = [k for i, k in enumerate(self.keypoints) if i in idx]
            logger.info('Removed {} duplicate keypoints'.format(a.shape[0] - idx.shape[0]))
        self.keypoints = kp
        self.descriptors = desc
