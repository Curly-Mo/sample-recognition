import logging

import numpy as np


logger = logging.getLogger(__name__)


def from_file(audio_path, sr, id, settings, feature='sift', **kwargs):
    if feature == 'sift':
        from . import sift
        return sift.from_file(audio_path, sr, id, settings, **kwargs)
    else:
        raise NotImplementedError


class Fingerprint(object):
    _keypoints = NotImplemented
    _descriptors = NotImplemented
    _spectrogram = NotImplemented

    @property
    def keypoints(self):
        if self._keypoints is NotImplemented:
            raise NotImplementedError
        return self._keypoints

    @keypoints.setter
    def keypoints(self, value):
        self._keypoints = value

    @property
    def descriptors(self):
        if self._descriptors is NotImplemented:
            raise NotImplementedError
        return self._descriptors

    @descriptors.setter
    def descriptors(self, value):
        self._descriptors = value

    @property
    def spectrogram(self):
        if self._spectrogram is NotImplemented:
            raise NotImplementedError
        return self._spectrogram

    @spectrogram.setter
    def spectrogram(self, value):
        self._spectrogram = value

    def remove_similar_keypoints(self):
        if len(self.descriptors) > 0:
            logger.info('Removing duplicate/similar keypoints...')
            a = np.array(self.descriptors)
            rounding_factor = 10
            b = np.ascontiguousarray(
                (a//rounding_factor)*rounding_factor
            ).view(
                np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
            )
            _, idx = np.unique(b, return_index=True)
            desc = a[sorted(idx)]
            kp = [k for i, k in enumerate(self.keypoints) if i in idx]
            logger.info('Removed {} duplicate keypoints'.format(
                a.shape[0] - idx.shape[0])
            )
        self.keypoints = kp
        self.descriptors = desc
