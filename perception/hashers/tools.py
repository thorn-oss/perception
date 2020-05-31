# pylint: disable=too-many-locals
import os
import io
import math
import json
import queue
import base64
import typing
import hashlib
import warnings
import threading
import functools
import itertools
import subprocess
from urllib import request
from http import client

import numpy as np
import validators
import cv2

try:
    import PIL
    import PIL.Image
except ImportError:  # pragma: no cover
    PIL = None

ImageInputType = typing.Union[str, np.ndarray, 'PIL.Image.Image', io.BytesIO]

SIZES = {'float32': 32, 'uint8': 8, 'bool': 1}


# pylint: disable=invalid-name
def compute_quality(image):
    """Compute a quality metric, using the calculation proposed by
    `Facebook <https://github.com/facebook/ThreatExchange/blob/master/hashing/hashing.pdf/>`_
    for their PDQ hash algorithm."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)
    if image.shape[0] != 64 or image.shape[1] != 64:
        image = cv2.resize(src=image, dsize=(64, 64)).astype('float32')
    dx = 100 * np.abs(image[:, 1:] - image[:, :-1]) / 255
    dy = 100 * np.abs(image[1:] - image[:-1]) / 255
    dx = dx.astype('int').sum()
    dy = dy.astype('int').sum()
    return np.clip(a=int((dx + dy) / 90), a_min=0, a_max=100)


def compute_md5(filepath) -> str:
    """Compute the md5 hash for a file at `filepath`.

    Args:
        filepath: The path to the file
    """
    with open(filepath, 'rb') as f:  # pylint: disable=invalid-name
        hash_str = hashlib.md5(f.read()).hexdigest()
    return hash_str


def get_string_length(hash_length: int, dtype: str, hash_format='hex') -> int:
    """Compute the expected length of a hash string.

    Args:
        hash_length: The length of the hash vector
        dtype: The dtype of the vector
        hash_format: One of 'base64' or 'hex'

    Returns:
        The expected string length
    """
    hash_bytes = math.ceil(hash_length * SIZES[dtype] / 8)

    if hash_format == 'base64':
        return int((4 * hash_bytes / 3) + 3) & ~3
    if hash_format == 'hex':
        return 2 * hash_bytes
    raise NotImplementedError('Unknown hash format: ' + hash_format)


def vector_to_string(vector: np.ndarray, dtype: str, hash_format: str):
    """Convert vector to hash.

    Args:
        vector: Input vector
    """
    # At times, a vector returned by a hasher is None (e.g., for hashes
    # that depend on the image not being featureless). In those cases,
    # we need to just return None, which is the least surprising outcome
    # because after all, the string representation of None is None.
    if vector is None:
        return vector
    if hash_format == 'vector':
        return vector.astype(dtype)
    if dtype == 'uint8':
        vector_bytes = vector.astype('uint8')
    elif dtype == 'float32':
        vector_bytes = vector.astype('float32')
    elif dtype == 'bool':
        vector_bytes = np.packbits(vector.astype('bool'))
    else:
        raise NotImplementedError(f'Cannot convert hash of type {dtype}.')
    if hash_format == 'base64':
        return base64.b64encode(vector_bytes).decode('utf-8')
    if hash_format == 'hex':
        return vector_bytes.tobytes().hex()
    raise NotImplementedError(
        f'Cannot convert to string format: {hash_format}.')


def string_to_vector(hash_string: str,
                     dtype: str,
                     hash_length: int,
                     hash_format: str,
                     verify_length: bool = True):
    """Convert hash back to vector.

    Args:
        hash_string: The input base64 hash string
        dtype: The data type of the hash
        hash_length: The length of the hash vector
        verify_length: Whether to verify the string length
    """
    assert not verify_length or len(hash_string) == get_string_length(
        hash_length=hash_length, hash_format=hash_format,
        dtype=dtype), 'Incorrect string length for this hash format.'
    if hash_format == 'base64':
        vector_bytes = np.frombuffer(
            base64.b64decode(hash_string),
            dtype='uint8' if dtype in ['bool', 'uint8'] else dtype)
    elif hash_format == 'hex':
        vector_bytes = np.frombuffer(
            bytearray.fromhex(hash_string),
            dtype='uint8' if dtype in ['bool', 'uint8'] else dtype)
    else:
        raise NotImplementedError(
            f'Cannot convert to string format: {hash_format}')
    if dtype == 'uint8':
        return vector_bytes[:hash_length]
    if dtype == 'float32':
        return vector_bytes[:hash_length]
    if dtype == 'bool':
        return np.unpackbits(vector_bytes)[:hash_length].astype('bool')
    raise NotImplementedError(f'Cannot convert hash of type {dtype}.')


def to_image_array(image: ImageInputType, require_color=True):
    if isinstance(image, np.ndarray):
        assert image.flags['C_CONTIGUOUS'], (
            'Provided arrays must be contiguous to avoid '
            'erroneous results when arrays are passed to '
            'underlying libraries. This can be achieved using'
            'np.ascontiguousarray(image)')
        assert not require_color or (len(image.shape) == 3
                                     and image.shape[-1] == 3), (
                                         'Provided images must be RGB images.')
        return image
    return read(image)


def get_common_framerates(id_rates: dict):
    """Compute an optimal set of framerates for a list
    of framerates. Optimal here means that reading the video
    at each of the framerates will allow one to collect all
    of the frames required with the smallest possible number of
    frames decoded.

    For example, consider if we need to read a video at
    3 fps, 5 fps, 1 fps and 0.5 fps. We could read the video
    4 times (once per framerate). But a more optimal approach
    is to read the video only twice, once at 3 frames per second
    and another time at 5 frames per second. For the 1 fps hasher,
    we simply pass every 3rd frame of the 3 fps pass. For the
    0.5 fps hasher, we pass every 6th frame of the 3 fps pass. So
    if you pass this function {A: 3, B: 5, C: 1, D: 0.5}, you will
    get back {3: [A, C, D], 5: C}.

    Args:
        id_rates: A dictionary with IDs as keys and frame rates as values.

    Returns:
        rate_ids: A dictionary with framerates as keys and a list of
            ids as values.
    """

    def partition(collection):
        """This function taken from
        https://stackoverflow.com/questions/19368375/set-partitions-in-python/30134039#30134039
        """
        if len(collection) == 1:
            yield [collection]
            return

        first = collection[0]
        for smaller in partition(collection[1:]):
            # insert `first` in each of the subpartition's subsets
            for n, subset in enumerate(smaller):
                yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
            # put `first` in its own subset
            yield [[first]] + smaller

    framerates = list(id_rates.values())
    factor = 2 * 3 * 5 * 7 * 11 * 60 * 60
    assert min(framerates
               ) >= 1 / factor, 'Framerates must be at least 1 frame per hour.'
    best_frame_count = np.inf
    best_grouping: typing.Optional[typing.List] = None
    best_frame_rates: typing.Optional[typing.List] = None

    # We try every possible grouping of framerates to minimize the number
    # of frames we decode. There is likely a better way to do this,
    # but this seems to do the job for now.
    for grouping in partition(list(set(framerates))):
        current_frame_rates = [
            # pylint: disable=no-member
            functools.reduce(np.lcm,
                             (np.array(group) * factor).round().astype(int)) /
            factor for group in grouping
        ]
        current_frame_count = sum(current_frame_rates)
        if current_frame_count < best_frame_count:
            best_frame_count = current_frame_count
            best_frame_rates = current_frame_rates
            best_grouping = grouping

    assert best_frame_rates is not None
    assert best_grouping is not None
    return {
        framerate:
        tuple(name for name, rate in id_rates.items() if rate in group)
        for framerate, group in zip(best_frame_rates, best_grouping)
    }


def get_isometric_transforms(image: ImageInputType, require_color=True):
    image = to_image_array(image, require_color=require_color)
    return dict(
        r0=image,
        fv=np.ascontiguousarray(image[::-1, :]),
        fh=np.ascontiguousarray(image[:, ::-1]),
        r180=np.ascontiguousarray(image[::-1, ::-1]),
        r90=np.ascontiguousarray(image.transpose(1, 0, 2)[::-1, :, :]),
        r90fv=np.ascontiguousarray(image.transpose(1, 0, 2)),
        r90fh=np.ascontiguousarray(image.transpose(1, 0, 2)[::-1, ::-1]),
        r270=np.ascontiguousarray(image.transpose(1, 0, 2)[:, ::-1]))


def get_isometric_dct_transforms(dct: np.ndarray):
    # pylint: disable=invalid-name
    T1 = np.empty_like(dct)
    T1[::2] = 1
    T1[1::2] = -1

    # pylint: disable=invalid-name
    T2 = np.empty_like(dct)
    T2[::2, ::2] = 1
    T2[1::2, 1::2] = 1
    T2[::2, 1::2] = -1
    T2[1::2, ::2] = -1
    return dict(
        r0=dct,
        fv=dct * T1,
        fh=dct * T1.T,
        r180=dct * T2,
        r90=dct.T * T1,
        r90fv=dct.T,
        r90fh=dct.T * T2,
        r270=dct.T * T1.T)


def read(filepath_or_buffer: ImageInputType, timeout=None):
    """Read a file into an image object

    Args:
        filepath_or_buffer: The path to the file or any object
            with a `read` method (such as `io.BytesIO`)
        timeout: If filepath_or_buffer is a URL, the timeout to
            use for making the HTTP request.
    """
    if PIL is not None and isinstance(filepath_or_buffer, PIL.Image.Image):
        return np.array(filepath_or_buffer.convert("RGB"))
    if isinstance(filepath_or_buffer, (io.BytesIO, client.HTTPResponse)):
        image = np.asarray(
            bytearray(filepath_or_buffer.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    elif (isinstance(filepath_or_buffer, str)
          and validators.url(filepath_or_buffer)):
        return read(request.urlopen(filepath_or_buffer, timeout=timeout))
    else:
        if not os.path.isfile(filepath_or_buffer):
            raise FileNotFoundError('Could not find image at path: ' +
                                    filepath_or_buffer)
        image = cv2.imread(filepath_or_buffer)
    if image is None:
        raise ValueError(f'An error occurred reading {filepath_or_buffer}.')
    # We use cvtColor here instead of just ret[..., ::-1]
    # in order to ensure that we provide a contiguous
    # array for later processing. Some hashers use ctypes
    # to pass the array and non-contiguous arrays can lead
    # to erroneous results.
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _get_frame_types(filepath):
    """Get the frame types for the frames in a video.

    Args:
        filepath: Path to the target file

    Returns:
        A list of dictionaries with pict_type
        (values are 'I', 'P', or 'B') and
        coded_picture_number (which represents the
        frame).
    """
    args = [
        'ffprobe', '-select_streams', 'v', '-i', filepath, '-print_format',
        'json', '-show_entries', 'frame=pict_type,coded_picture_number'
    ]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        raise ValueError("{out}: {err}".format(out=str(out), err=str(err)))
    frames = json.loads(out.decode('utf-8'))['frames']
    frames.sort(key=lambda f: f['coded_picture_number'])
    return frames


def _get_keyframes(filepath):
    """Get the keyframes for a video.

    Args:
        filepath: Path to the target file

    Returns:
        A list of frame indexes.
    """
    args = [
        'ffprobe', '-select_streams', 'v', '-i', filepath, '-print_format',
        'json', '-show_entries', 'frame=pict_type,coded_picture_number'
    ]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        raise ValueError("{out}: {err}".format(out=str(out), err=str(err)))
    data = json.loads(out.decode('utf-8'))['frames']
    frames = [f['coded_picture_number'] for f in data if f['pict_type'] == 'I']
    frames = list(set(frames))
    frames.sort()
    return frames


# pylint: disable=too-many-branches,too-many-locals,too-many-statements
def read_video_to_generator(
        filepath,
        frames_per_second: typing.Optional[typing.Union[str, float]] = None,
        errors='raise'):
    # pylint: disable=no-member
    if cv2.__version__ < '4.1.1' and filepath.lower().endswith('gif'):
        message = 'Versions of OpenCV < 4.1.1 may read GIF files improperly. Upgrade recommended.'
        if errors == 'raise':
            raise ValueError(message)
        warnings.warn(message=message)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f'Could not find {filepath}.')
    cap = cv2.VideoCapture(filename=filepath, apiPreference=cv2.CAP_FFMPEG)
    try:
        # The purpose of the following block is largely to create a
        # frame_indexes (iterator or list) that indicates which
        # frames we should be returning to the user and then
        # yielding those frames as we come across them.
        file_frames_per_second = cap.get(cv2.CAP_PROP_FPS)
        if file_frames_per_second == 0:
            if errors == "raise":
                raise ValueError("Video file has framerate of 0fps.")
            # The known case where this occurs is for GIFs, where
            # 0 fps is typically inferred as 10 fps.
            file_frames_per_second = 10
            if errors == "warn":
                warnings.warn(
                    message=
                    "Video file has framerate of 0 fps. Guessing framerate of 10fps."
                )
        if frames_per_second is None:
            frames_per_second = file_frames_per_second
        seconds_between_desired_frames = None if (
            frames_per_second is not None
            and isinstance(frames_per_second,
                           str)) else 1 / frames_per_second  # type: ignore
        seconds_between_grabbed_frames = 1 / file_frames_per_second
        grabbed_frame_count = 0
        if frames_per_second == 'keyframes':
            frame_indexes: typing.Union[range, typing.List[int], typing.
                                        Iterator[int]] = _get_keyframes(
                                            filepath)
            # The repeat flag is used to handle the case where the
            # desired sampling rate is higher than the file's frame
            # rate. In this case, we will need to repeat frames in
            # order to provide the least-surprising behavior that
            # we can.
            repeat = False
        else:
            frame_indexes = itertools.count(
                0, file_frames_per_second / frames_per_second)
            repeat = file_frames_per_second < frames_per_second
        for frame_index in frame_indexes:
            while grabbed_frame_count < frame_index:
                # We need to skip this frame.
                success = cap.grab()
                if not success:
                    break
                grabbed_frame_count += 1
            success, frame = cap.read()
            grabbed_frame_count += 1
            if not success:
                # The video is over or an error has occurred.
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_timestamp = frame_index / file_frames_per_second
            yield frame, grabbed_frame_count - 1, current_timestamp
            if repeat:
                next_desired_timestamp = current_timestamp + seconds_between_desired_frames
                next_timestamp = current_timestamp + seconds_between_grabbed_frames
                while next_desired_timestamp < next_timestamp:
                    yield (frame, grabbed_frame_count - 1,
                           next_desired_timestamp)
                    next_desired_timestamp += seconds_between_desired_frames
    # pylint: disable=broad-except
    except Exception as e:
        if errors not in ['warn', 'ignore']:
            raise e
        if errors == 'warn':
            warnings.warn(
                message=
                f'An error occurred while reading {filepath}. Processing may be truncated.'
            )
    finally:
        cap.release()


def read_video_into_queue(*args, video_queue, terminate, **kwargs):
    # We're inside a thread now and the queue is being read elsewhere.
    try:
        for frame, frame_index, timestamp in read_video_to_generator(
                *args, **kwargs):
            if not terminate.isSet():
                video_queue.put((frame, frame_index, timestamp))
            else:
                break
    finally:
        video_queue.put((None, None, None))


def read_video(
        filepath,
        frames_per_second: typing.Optional[typing.Union[str, float]] = None,
        max_queue_size=128,
        use_queue=True,
        errors='raise'):
    """Provides a generator of RGB frames, frame indexes, and timestamps from a
    video. This function requires you to have installed ffmpeg.

    Args:
        filepath: Path to the video file
        frames_per_second: How many frames to provide for
            each second of video. If None, all frames
            are provided. If frames_per_second is "keyframes",
            we use ffmpeg to select I frames from the video.
        max_queue_size: The maximum number of frames to load in the queue
        use_queue: Whether to use a queue of frames during processing
        errors: Whether to 'raise', 'warn', or 'ignore' errors

    Yields:
        (frame, frame_index, timestamp) tuples
    """
    if use_queue:
        video_queue = queue.Queue(
            maxsize=max_queue_size
        )  # type: queue.Queue[typing.Tuple[np.ndarray, int, float]]
        terminate = threading.Event()
        thread = threading.Thread(
            target=read_video_into_queue,
            kwargs={
                'frames_per_second': frames_per_second,
                'video_queue': video_queue,
                'filepath': filepath,
                'errors': errors,
                'terminate': terminate
            })
        thread.start()
        try:
            while True:
                frame, frame_index, timestamp = video_queue.get()
                video_queue.task_done()
                if frame is None:
                    break
                yield (frame, frame_index, timestamp)
        finally:
            # Set the termination flag for the
            # background thread.
            terminate.set()
            try:
                # Unblock the thread, in the event
                # that it is waiting.
                video_queue.get_nowait()
            except queue.Empty:
                # It doesn't matter if it's empty.
                pass
            # Wait for the background thread to terminate.
            thread.join()
    else:
        for frame, frame_index, timestamp in read_video_to_generator(
                filepath=filepath,
                frames_per_second=frames_per_second,
                errors=errors):
            yield (frame, frame_index, timestamp)


def compute_synchronized_video_hashes(filepath: str,
                                      hashers: dict,
                                      framerates=None,
                                      hash_format='base64',
                                      use_queue=True):
    """Compute the video hashes for a group of hashers with synchronized
    frame processing wherever possible.

    Args:
        filepath: Path to video file.
        hashers: A dictionary mapping hasher names to video hasher objects
        hash_format: The format in which to return the hashes
        use_queue: Whether to use queued video frames
    """
    if framerates is None:
        framerates = get_common_framerates({
            k: h.frames_per_second
            for k, h in hashers.items() if h.frames_per_second is not None
        })
    else:
        assert all(
            any(hasher_name in hasher_names
                for hasher_names in framerates.values())
            for hasher_name, hasher in hashers.items()
            if hasher.frames_per_second is not None
        ), 'Provided framerates do not have an entry for all required hashers.'

    results = {
        hasher_name: {
            'state':
            None,
            'hash':
            None,
            'relative_framerate':
            next(framerate / hasher.frames_per_second
                 for framerate, hasher_names in framerates.items()
                 if hasher_name in hasher_names)
        }
        for hasher_name, hasher in hashers.items()
        if hasher.frames_per_second is not None
    }
    for current_framerate, current_hasher_names in framerates.items():
        for frame_index, (frame, grabbed_frame_index,
                          frame_timestamp) in enumerate(
                              read_video(
                                  filepath=filepath,
                                  frames_per_second=current_framerate,
                                  use_queue=use_queue)):
            for hasher_name in current_hasher_names:
                config = results[hasher_name]
                hasher = hashers[hasher_name]
                assert config['relative_framerate'] is not None
                if frame_index % config['relative_framerate'] == 0:
                    config['state'] = hasher.process_frame(
                        frame=frame,
                        frame_index=grabbed_frame_index,
                        frame_timestamp=frame_timestamp,
                        state=config['state'])
        for hasher_name in current_hasher_names:
            config = results[hasher_name]
            hasher = hashers[hasher_name]
            current_hash = hasher.hash_from_final_state(state=config['state'])
            if hash_format == 'vector':
                config['hash'] = current_hash
            else:
                if not hasher.returns_multiple:
                    config['hash'] = hasher.vector_to_string(
                        current_hash, hash_format=hash_format)
                else:
                    config['hash'] = [
                        hasher.vector_to_string(h, hash_format=hash_format)
                        for h in current_hash
                    ]
            config['state'] = None
    hashes = {
        hasher_name: config['hash']
        for hasher_name, config in results.items()
    }
    for hasher_name, hasher in hashers.items():
        if hasher.frames_per_second is None:
            # This is a custom hasher that we just pass a video path to.
            hashes[hasher_name] = hasher.compute(filepath)
    return hashes


def unletterbox(image) -> typing.Optional[
        typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]]]:
    """Obtain bounds on an image that remove the black bars
    on the top, right, bottom, and left side of an image.

    Args:
        image: The image from which to remove letterboxing.

    Returns:
        A pair of coordinates bounds of the form (x1, x2)
        and (y1, y2) representing the left, right, top, and
        bottom bounds.
    """
    adj = image.mean(axis=2) > 2
    if adj.all():
        bounds = (0, image.shape[1] + 1), (0, image.shape[0])
    else:
        y = np.where(adj.sum(axis=1) > 0.1 * image.shape[0])[0]
        x = np.where(adj.sum(axis=0) > 0.1 * image.shape[1])[0]
        if len(y) <= 1 or len(x) <= 1:
            return None
        x1, x2 = x[[0, -1]]
        y1, y2 = y[[0, -1]]
        bounds = (x1, x2 + 1), (y1, y2 + 1)
    return bounds
