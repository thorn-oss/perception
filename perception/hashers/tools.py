import os
import io
import math
import json
import base64
import typing
import hashlib
import warnings
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


def read(filepath_or_buffer: ImageInputType):
    """Read a file into an image object

    Args:
        filepath_or_buffer: The path to the file or any object
            with a `read` method (such as `io.BytesIO`)
    """
    if PIL is not None and isinstance(filepath_or_buffer, PIL.Image.Image):
        return np.array(filepath_or_buffer.convert("RGB"))
    if isinstance(filepath_or_buffer, (io.BytesIO, client.HTTPResponse)):
        image = np.asarray(
            bytearray(filepath_or_buffer.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    elif (isinstance(filepath_or_buffer, str)
          and validators.url(filepath_or_buffer)):
        return read(request.urlopen(filepath_or_buffer))
    else:
        assert os.path.isfile(filepath_or_buffer), \
            'Could not find image at path: ' + filepath_or_buffer
        image = cv2.imread(filepath_or_buffer)
    if image is None:
        raise ValueError(f'An error occurred reading {image}.')
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


def read_video(
        filepath,
        frames_per_second: typing.Optional[typing.Union[str, int]] = None,
        errors='raise'):
    """Provides a generator of RGB frames, frame indexes, and timestamps from a
    video. This function requires you to have installed ffmpeg.

    Args:
        filepath: Path to the video file
        frames_per_second: How many frames to provide for
            each second of video. If None, all frames
            are provided. If frames_per_second is "keyframes",
            we use ffmpeg to select I frames from the video.

    Yields:
        (frame, frame_index, timestamp) tuples
    """
    # pylint: disable=no-member
    if cv2.__version__ < '4.1.1' and filepath.lower().endswith('gif'):
        warnings.warn(
            message=
            'Versions of OpenCV < 4.1.1 may read GIF files improperly. Upgrade recommended.'
        )
    cap = cv2.VideoCapture(filepath)
    try:
        n_frames, video_frames_per_second = cap.get(
            cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        if frames_per_second is None:
            frames_per_second = video_frames_per_second
        if frames_per_second == 'keyframes':
            frame_indexes: typing.Union[range, typing.List[int], typing.
                                        Iterator[int]] = _get_keyframes(
                                            filepath)
        else:
            if n_frames < 1:
                frame_indexes = itertools.count(
                    0, int(video_frames_per_second // frames_per_second))
            else:
                frame_indexes = range(
                    0, int(n_frames),
                    max(1, int(video_frames_per_second // frames_per_second)))

        for frame_index in frame_indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = cap.read()
            if not success:
                # The video is over or an error has occurred.
                break
            yield cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            ), frame_index, frame_index / video_frames_per_second
    # pylint: disable=broad-except
    except Exception as e:
        if errors == 'raise':
            cap.release()
            raise e
        if errors == 'warn':
            warnings.warn(
                message=
                f'An error occurred while reading {filepath}. Processing may be truncated.'
            )
    cap.release()
