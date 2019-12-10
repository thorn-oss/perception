from ..hasher import VideoHasher, ImageHasher
from .. import tools


class FramewiseHasher(VideoHasher):
    """A hasher that simply returns frame-wise hashes at some
    regular interval with some minimum inter-frame distance threshold."""

    returns_multiple = True

    def __init__(self,
                 frame_hasher: ImageHasher,
                 interframe_threshold: float,
                 frames_per_second: int = 15,
                 quality_threshold: float = None):
        self.hash_length = frame_hasher.hash_length
        self.frames_per_second = frames_per_second
        self.frame_hasher = frame_hasher
        self.distance_metric = frame_hasher.distance_metric
        self.dtype = frame_hasher.dtype
        self.interframe_threshold = interframe_threshold
        self.quality_threshold = quality_threshold

    # pylint: disable=unused-argument
    def process_frame(self, frame, frame_index, frame_timestamp, state=None):
        if self.quality_threshold is not None:
            current, quality = self.frame_hasher.compute_with_quality(
                frame, hash_format='vector')
            if quality < self.quality_threshold:
                return state or {'previous': None, 'hashes': []}
        else:
            current = self.frame_hasher.compute(frame, hash_format='vector')
        if state is None or state['previous'] is None:
            # We keep a separate reference to the previous hash instead of using
            # the last entry in the hashes list because `compute_batches` may
            # clear the hashes list but we still want to be able to compare
            # the final entry.
            state = {
                'previous': current,
                'hashes': [current],
            }
        else:
            if self.frame_hasher.compute_distance(
                    current, state['previous']) > self.interframe_threshold:
                state['hashes'].append(current)
        return state

    def compute_batches(self,
                        filepath: str,
                        batch_size: int,
                        errors='raise',
                        hash_format='base64'):
        """Compute hashes for a video in batches.

        Args:
            filepath: Path to video file
            batch_size: The batch size to use for returning hashes
            errors: One of "raise", "ignore", or "warn". Passed
                to perception.hashers.tools.read_video.
            hash_format: The format in which to return hashes
        """

        def format_batch(hashes):
            return [
                self.vector_to_string(vector, hash_format=hash_format)
                if hash_format != 'vector' else vector for vector in hashes
            ]

        state = None
        for frame, frame_index, frame_timestamp in tools.read_video(
                filepath=filepath,
                frames_per_second=self.frames_per_second,
                errors=errors):
            state = self.process_frame(
                frame=frame,
                frame_index=frame_index,
                frame_timestamp=frame_timestamp,
                state=state)
            if state is not None and len(state['hashes']) > batch_size:
                yield format_batch(state['hashes'])
                state['hashes'] = []
        if state is not None and state['hashes']:
            yield format_batch(state['hashes'])

    def hash_from_final_state(self, state):
        if state is None:
            return []
        return state['hashes']
