# pylint: disable=invalid-name
import logging

import numpy as np
import cv2

from .. import tools
from ...utils import flatten
from ..hasher import VideoHasher
from ..image.phash import PHashU8
from .tmk import TMKL1

logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes
class SimpleSceneDetection(VideoHasher):
    """The SimpleSceneDetection hasher is a wrapper around other video hashers
    to create separate hashes for different scenes / shots in a video. It works
    by shrinking each frame, blurring it, and doing a simple delta with the previous
    frame. If they are different, this marks the start of a new scene. In addition,
    this wrapper will also remove letterboxing from videos by checking for solid
    black areas on the edges of the frame.

    Args:
        base_hasher: The base video hasher to use for each scene.
        interscene_threshold: The distance threshold between sequential scenes that
            new hashes must meet to be included (this is essentially for deduplication)
        min_frame_size: The minimum frame size to use for computing hashes. This is
            relevant for letterbox detection as black frames will tend to be completely
            "cropped" and make the frame very small.
        max_scene_length: The maximum length of a single scene.
    """
    returns_multiple = True

    def __init__(self,
                 base_hasher: VideoHasher = None,
                 interscene_threshold=None,
                 min_frame_size=50,
                 max_scene_length=None):
        if base_hasher is None:
            base_hasher = TMKL1(
                frames_per_second=2,
                frame_hasher=PHashU8(
                    exclude_first_term=False, freq_shift=1, hash_size=12),
                distance_metric='euclidean',
                dtype='uint8',
                norm=None,
                quality_threshold=90)
            if interscene_threshold is None:
                interscene_threshold = 50
        if interscene_threshold is not None and base_hasher.returns_multiple:
            raise ValueError(
                'Interscene thresholds not supported for hashers returning multiple hashes.'
            )
        self.base_hasher = base_hasher
        self.frames_per_second = base_hasher.frames_per_second
        self.distance_metric = base_hasher.distance_metric
        self.dtype = base_hasher.dtype
        self.hash_length = base_hasher.hash_length
        self.max_scene_length = max_scene_length
        self.interscene_threshold = interscene_threshold
        self.min_frame_size = min_frame_size

    def compute_batches(self,
                        filepath,
                        errors='raise',
                        hash_format='base64',
                        batch_size=10):
        """Compute a hash for a video at a given filepath and
        yield hashes in a given batch size.

        Args:
            filepath: Path to video file
            errors: One of "raise", "ignore", or "warn". Passed
                to perception.hashers.tools.read_video.
            hash_format: The hash format to use when returning hashes.
            batch_size: The minimum number of hashes to include in each batch.
        """

        def convert(scenes):
            if hash_format == 'vector':
                return scenes
            if self.base_hasher.returns_multiple:
                return [([
                    self.vector_to_string(h, hash_format=hash_format)
                    for h in hs
                ], frames) for hs, frames in scenes]
            return [(self.vector_to_string(h, hash_format=hash_format), frames)
                    for h, frames in scenes]

        state = None
        for frame, frame_index, frame_timestamp in tools.read_video(
                filepath=filepath,
                frames_per_second=self.frames_per_second,
                errors=errors):
            state = self.process_frame(
                frame=frame,
                frame_index=frame_index,
                frame_timestamp=frame_timestamp,
                state=state,
                batch_mode=True)
            if len(state['scenes']) >= batch_size:
                yield convert(state['scenes'])
                state['scenes'] = []
        assert state is not None
        if state['substate']:
            self.handle_scene(state)
        if state['scenes']:
            yield convert(state['scenes'])

    # pylint: disable=bad-continuation
    def handle_scene(self, state, frame_timestamp=None):
        subhash = self.base_hasher.hash_from_final_state(state['substate'])
        if subhash is not None and (self.base_hasher.returns_multiple or (
            (self.interscene_threshold is None or not state['scenes']
             or self.compute_distance(state['scenes'][-1][0],
                                      subhash) > self.interscene_threshold))):
            state['scenes'].append((subhash, state['frames']))
        state['substate'] = None
        state['bounds'] = None
        state['frames'] = []
        state['pre'] = None
        if frame_timestamp is not None:
            state['start'] = frame_timestamp

    def crop(self, frame, bounds):
        # Check to see we have set bounds for this scene yet.
        if not bounds:
            # We don't have bounds, so we'll set them.
            bounds = tools.unletterbox(frame)
            # If the bounds come back invalid (i.e., the frame is too small)
            # or no bounds are found (i.e., the frame is all back), we
            # return None.
            if bounds is None or min(bounds[0][1] - bounds[0][0], bounds[1][1]
                                     - bounds[1][0]) < self.min_frame_size:
                return None, None, None
        (x1, x2), (y1, y2) = bounds
        cropped = np.ascontiguousarray(frame[y1:y2, x1:x2])
        current = cv2.resize(
            cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY), (128, 128))
        current = cv2.blur(current, ksize=(4, 4))
        return cropped, current, bounds

    # pylint: disable=arguments-differ,too-many-arguments
    def process_frame(self,
                      frame,
                      frame_index,
                      frame_timestamp,
                      state=None,
                      batch_mode=False):
        if not state:
            state = {
                'pre': None,
                'substate': None,
                'start': 0,
                'bounds': None,
                'frames': [],
                'scenes': []
            }
        cropped, current, state['bounds'] = self.crop(frame, state['bounds'])
        if cropped is None:
            # A good crop was not found so we set the start of the scene to this
            # point and continue on to the next frame. This will repeat until we
            # find appropriate bounds.
            state['start'] = frame_timestamp
            return state

        # Check if we have a previous frame to compare the
        # current frame to.
        if state['pre'] is not None:
            # Compute similarity between the previous frame and the
            # current frame.
            similarity = 1 - np.abs(state['pre'].astype('float32') - current.
                                    astype('float32')).sum() / (255 * 128**2)
            if similarity < 0.95 or (self.max_scene_length is not None
                                     and frame_timestamp - state['start'] >
                                     self.max_scene_length):
                # The similarity is too low. We've started a new scene.
                self.handle_scene(state, frame_timestamp)
                cropped, current, state['bounds'] = self.crop(
                    frame, state['bounds'])
                if cropped is None:
                    # See comment above about invalid crops.
                    state['start'] = frame_timestamp
                    return state

        state['pre'] = current
        try:
            state['substate'] = self.base_hasher.process_frame(
                cropped, frame_index, frame_timestamp, state=state['substate'])
            if batch_mode:
                state['frames'].append((frame, frame_index, frame_timestamp))
        except Exception as e:  # pylint: disable=broad-except
            logger.warning('An error occurred while processing a frame: %s',
                           str(e))
        return state

    def hash_from_final_state(self, state):
        if state['substate']:
            self.handle_scene(state)
        if not self.base_hasher.returns_multiple:
            return [h for h, _ in state['scenes']]
        return flatten([hs for hs, _ in state['scenes']])
