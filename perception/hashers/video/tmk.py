# pylint: disable=invalid-name,too-many-instance-attributes,too-many-locals
import numpy as np
import scipy.special

from ..hasher import VideoHasher, ImageHasher
from ..image.phash import PHashF


class TMKL2(VideoHasher):
    """The TMK L2 video hashing algorithm."""

    dtype = 'float32'
    distance_metric = 'custom'

    def __init__(self,
                 frame_hasher: ImageHasher = None,
                 frames_per_second: int = 15,
                 normalization: str = 'matrix'):
        T = np.array([2731, 4391, 9767, 14653]).astype('float32')
        m = 32
        if frame_hasher is None:
            frame_hasher = PHashF(
                hash_size=16, exclude_first_term=True, freq_shift=1)
        self.frames_per_second = frames_per_second
        assert frame_hasher.dtype != 'bool', 'This hasher requires real valued hashes.'

        # Beta parameter of the modified Bessel function of the first kind
        self.beta = 32

        # Number of Fourier coefficients per period
        self.m = m

        # The periods with shape (T, )
        self.T = T  # (T)

        # The Fourier coefficients with shape (T, m, 1)
        self.ms = 2 * np.pi * np.arange(0, self.m).astype('float32')  # (m)
        self.ms_normed = (
            self.ms[np.newaxis, ] / self.T.reshape(-1, 1)).reshape(
                len(self.T), self.m, 1)  # (T, m, 1)

        # The weights with shape (T, 2m, 1)
        a = np.array([(scipy.special.iv(0, self.beta) - np.exp(-self.beta)) /
                      (2 * np.sinh(self.beta))] + [
                          scipy.special.iv(i, self.beta) / np.sinh(self.beta)
                          for i in range(1, self.m)
                      ])
        a = a.reshape(1, -1).repeat(repeats=len(self.T), axis=0)
        a = np.sqrt(a)
        self.a = a[..., np.newaxis]

        # The frame-wise hasher
        self.frame_hasher = frame_hasher

        # pylint: disable=unsubscriptable-object
        self.hash_length = self.T.shape[
            0] * 2 * self.m * self.frame_hasher.hash_length

        self.normalization = normalization

    def process_frame(self, frame, frame_index, frame_timestamp, state=None):
        if state is None:
            state = {'features': [], 'timestamps': []}
        state['features'].append(
            self.frame_hasher.compute(frame, hash_format='vector'))
        state['timestamps'].append(frame_timestamp)
        return state

    def hash_from_final_state(self, state):
        timestamps = np.array(state['timestamps'])
        # pylint: disable=unsubscriptable-object
        features = np.array(state['features']).reshape(
            (1, 1, timestamps.shape[0], self.frame_hasher.hash_length))
        x = self.ms_normed * timestamps
        yw1 = np.sin(x) * self.a
        yw2 = np.cos(x) * self.a
        yw = np.concatenate([yw1, yw2], axis=1)[...,
                                                np.newaxis]  # (T, 2m, t, 1)
        y = (yw * features).sum(axis=2)  # (T, 2m, d)
        return y.flatten()

    def _compute_distance(self, vector1, vector2):
        shape = (len(self.T), 2 * self.m, self.frame_hasher.hash_length)
        return 1 - self._score_pair(
            fv_a=vector1.reshape(shape),
            fv_b=vector2.reshape(shape),
            offsets=None,
            normalization=self.normalization)

    def _score_pair(self, fv_a, fv_b, offsets=None, normalization='matrix'):
        eps = 1e-8

        if offsets is None:
            offsets = np.array([0])

        assert normalization in ['feat', 'freq', 'feat_freq',
                                 'matrix'], 'Invalid normalization'

        if "feat" in normalization:
            a_xp = np.concatenate([self.a, self.a], axis=1)  # (T, 2m, 1)
            fv_a_0 = fv_a / a_xp
            fv_b_0 = fv_b / a_xp
            norm_a = np.sqrt(np.sum(fv_a_0**2, axis=2, keepdims=True) +
                             eps) + eps
            norm_b = np.sqrt(np.sum(fv_b_0**2, axis=2, keepdims=True) +
                             eps) + eps
            fv_a = fv_a / norm_a
            fv_b = fv_b / norm_b

        if "freq" in normalization:
            norm_a, norm_b = [
                np.sqrt((fv**2).sum(axis=1, keepdims=True) / self.m + eps) +
                eps for fv in [fv_a, fv_b]
            ]
            fv_a = fv_a / norm_a
            fv_b = fv_b / norm_b

        if normalization == "matrix":
            norm_a, norm_b = [
                np.sqrt(np.sum(fv**2, axis=(1, 2)) + eps)[..., np.newaxis] +
                eps for fv in [fv_a, fv_b]
            ]  # (T, 1)

        fv_a_sin, fv_b_sin = [fv[:, :self.m]
                              for fv in [fv_a, fv_b]]  # (T, m, d)
        fv_a_cos, fv_b_cos = [fv[:, self.m:]
                              for fv in [fv_a, fv_b]]  # (T, m, d)
        ms = self.ms.reshape(-1, 1)  # (m, 1)
        dot_sin_sin, dot_sin_cos, dot_cos_cos, dot_cos_sin = [
            np.sum(p, axis=2, keepdims=True) for p in [
                fv_a_sin * fv_b_sin, fv_a_sin * fv_b_cos, fv_a_cos *
                fv_b_cos, fv_a_cos * fv_b_sin
            ]
        ]  # (T, m, 1)
        delta = ms.reshape(1, -1, 1) * offsets.reshape(1, -1) / self.T.reshape(
            (-1, 1, 1))
        cos_delta = np.cos(delta)  # (T, m, delta)
        sin_delta = np.sin(delta)  # (T, m, delta)
        dots = (dot_sin_sin * cos_delta + dot_sin_cos * sin_delta +
                dot_cos_cos * cos_delta - dot_cos_sin * sin_delta).sum(axis=1)
        if normalization == "matrix":
            dots = dots / (norm_a * norm_b)
        if normalization == "freq":
            dots = dots / self.m  # (T, m, delta)
        elif normalization in ["feat", "feat_freq"]:
            dots = dots / 512
        return dots.mean(axis=0)


class TMKL1(VideoHasher):
    """The TMK L1 video hashing algorithm."""

    # pylint: disable=too-many-arguments
    def __init__(self,
                 frame_hasher: ImageHasher = None,
                 frames_per_second: int = 15,
                 dtype='float32',
                 distance_metric='cosine',
                 norm=2,
                 quality_threshold=None):
        if frame_hasher is None:
            frame_hasher = PHashF(
                hash_size=16, exclude_first_term=True, freq_shift=1)
        self.hash_length = frame_hasher.hash_length
        self.frames_per_second = frames_per_second
        assert frame_hasher.dtype != 'bool', 'This hasher requires real valued hashes.'
        self.frame_hasher = frame_hasher
        self.norm = norm
        self.dtype = dtype or self.frame_hasher.dtype
        self.distance_metric = distance_metric or self.frame_hasher.distance_metric
        self.quality_threshold = quality_threshold

    # pylint: disable=unused-argument
    def process_frame(self, frame, frame_index, frame_timestamp, state=None):
        if state is None:
            state = {
                'sum': np.zeros(self.frame_hasher.hash_length),
                'frame_count': 0
            }
        if not self.quality_threshold:
            hash_vector = self.frame_hasher.compute(
                frame, hash_format='vector')
        else:
            hash_vector, quality = self.frame_hasher.compute_with_quality(
                frame, hash_format='vector')
            if quality < self.quality_threshold:
                return state
        if hash_vector is not None:
            state['sum'] += np.float32(hash_vector)
            state['frame_count'] += 1
        return state

    def hash_from_final_state(self, state):
        if state['frame_count'] == 0:
            return None
        average_vector = state['sum'] / state['frame_count']
        if self.norm is not None:
            return (average_vector / np.linalg.norm(
                average_vector, ord=self.norm)).astype(self.frame_hasher.dtype)
        return average_vector.astype(self.frame_hasher.dtype)
