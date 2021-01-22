# pylint: disable=too-many-locals
import os
import io
import math
import json
import shlex
import queue
import base64
import typing
import hashlib
import warnings
import logging
import fractions
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

LOGGER = logging.getLogger(__name__)

ImageInputType = typing.Union[str, np.ndarray, 'PIL.Image.Image', io.BytesIO]

SIZES = {'float32': 32, 'uint8': 8, 'bool': 1}

# Map codec names to the CUDA-accelerated version. Obtain
# from ffmpeg -codecs after building using CUDA.
CUDA_CODECS = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "mjpeg": "mjpeg_cuvid",
    "mpeg1video": "mpeg1_cuvid",
    "mpeg2video": "mpeg2_cuvid",
    "mpeg4": "mpeg4_cuvid",
    "vc1": "vc1_cuvid",
    "vp8": "vp8_cuvid",
    "vp9": "vp9_cuvid",
}

FramesWithIndexesAndTimestamps = typing.Generator[
    typing.Tuple[np.ndarray, typing.Optional[int], typing.
                 Optional[float]], None, None]


def get_ffprobe():
    return os.environ.get("PERCEPTION_FFPROBE_BINARY", "ffprobe")


def get_ffmpeg():
    return os.environ.get("PERCEPTION_FFMPEG_BINARY", "ffmpeg")


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
        hash_string: The input hash string
        dtype: The data type of the hash
        hash_length: The length of the hash vector
        hash_format: The input format of the hash (base64 or hex)
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


def hex_to_b64(hash_string: str,
               dtype: str,
               hash_length: int,
               verify_length: bool = True):
    """Convert a hex-encoded hash to base64.

    Args:
        hash_string: The input base64 hash string
        dtype: The data type of the hash
        hash_length: The length of the hash vector
        verify_length: Whether to verify the string length
    """
    return vector_to_string(
        string_to_vector(
            hash_string,
            hash_length=hash_length,
            hash_format='hex',
            dtype=dtype,
            verify_length=verify_length),
        dtype=dtype,
        hash_format='base64')


def b64_to_hex(hash_string: str,
               dtype: str,
               hash_length: int,
               verify_length: bool = True):
    """Convert a base64-encoded hash to hex.

    Args:
        hash_string: The input hex hash string
        dtype: The data type of the hash
        hash_length: The length of the hash vector
        verify_length: Whether to verify the string length
    """
    return vector_to_string(
        string_to_vector(
            hash_string,
            hash_length=hash_length,
            hash_format='base64',
            dtype=dtype,
            verify_length=verify_length),
        dtype=dtype,
        hash_format='hex')


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


def _get_keyframes(filepath):
    """Get the keyframes for a video.

    Args:
        filepath: Path to the target file

    Returns:
        A list of frame indexes.
    """
    args = [
        get_ffprobe(), '-select_streams', 'v', '-i', f"'{filepath}'",
        '-print_format', 'json', '-show_entries',
        'frame=pict_type,coded_picture_number'
    ]
    with subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        out, err = p.communicate()
        if p.returncode != 0:
            raise ValueError("{out}: {err}".format(out=str(out), err=str(err)))
        data = json.loads(out.decode('utf-8'))['frames']
        frames = [
            f['coded_picture_number'] for f in data if f['pict_type'] == 'I'
        ]
        frames = list(set(frames))
        frames.sort()
    return frames


def get_video_properties(filepath):
    cmd = f"""
    {get_ffprobe()} -select_streams v:0 -i '{filepath}'
    -print_format json -show_entries stream=width,height,avg_frame_rate,codec_name,start_time
    """
    with subprocess.Popen(
            shlex.split(cmd), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE) as p:
        out, err = p.communicate()
        if p.returncode != 0:
            raise ValueError("{out}: {err}".format(out=str(out), err=str(err)))
        data = json.loads(out.decode("utf-8"))["streams"][0]
        numerator, denominator = tuple(
            map(int, data["avg_frame_rate"].split("/")[:2]))
        avg_frame_rate: typing.Optional[fractions.Fraction]
        if numerator > 0 and denominator > 0:
            avg_frame_rate = fractions.Fraction(
                numerator=numerator, denominator=denominator)
        else:
            avg_frame_rate = None
        return data["width"], data["height"], avg_frame_rate, data[
            "codec_name"], float(data.get("start_time", "0"))


# pylint: disable=too-many-branches,too-many-statements,too-many-arguments
def read_video_to_generator_ffmpeg(
        filepath,
        frames_per_second: typing.Optional[typing.Union[str, float]] = None,
        errors="raise",
        max_duration: float = None,
        max_size: int = None,
        interp: str = None,
        frame_rounding: str = "up",
        draw_timestamps=False,
        use_cuda=False) -> FramesWithIndexesAndTimestamps:
    """This is used by :code:`read_video` when :code:`use_ffmpeg` is True. It
    differs from :code:`read_video_to_generator` in that it uses FFMPEG instead of
    OpenCV and, optionally, allows for CUDA acceleration. CUDA acceleration
    can be faster for larger videos (>1080p) where downsampling is desired.
    For other videos, CUDA may be slower, but the decoding load will still be
    taken off the CPU, which may still be advantageous. You can specify which
    FFMPEG binary to use by setting PERCEPTION_FFMPEG_BINARY.

    Args:
        filepath: See read_video
        frames_per_second: See read_video
        errors: See read_video
        max_duration: See read_video
        max_size: See read_video
        interp: The interpolation method to use. When not using CUDA, you must choose one
            of the `interpolation options <https://ffmpeg.org/ffmpeg-scaler.html#sws_005fflags>`_
            (default: area). When using CUDA, you must choose from the
            `interp_algo options <http://underpop.online.fr/f/ffmpeg/help/scale_005fnpp.htm.gz>`_
            (default: super).
        frame_rounding: The frame rounding method.
        draw_timestamps: Draw original timestamps onto the frames (for debugging only)
        use_cuda: Whether to enable CUDA acceleration. Requires a
            CUDA-accelerated version of ffmpeg.

    To build FFMPEG with CUDA, do the following in a Docker
    container based on nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04. The
    FFMPEG binary will be ffmpeg/ffmpeg.

    .. code-block:: bash

        git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
        cd nv-codec-headers
        sudo make install
        cd ..
        git clone --branch release/4.3 https://git.ffmpeg.org/ffmpeg.git
        cd ffmpeg
        sudo apt-get update && sudo apt-get -y install yasm
        export PATH=$PATH:/usr/local/cuda/bin
        ./configure --enable-cuda-nvcc --enable-cuvid --enable-nvenc --enable-nvdec \
                    --enable-libnpp --enable-nonfree --extra-cflags=-I/usr/local/cuda/include \
                    --extra-ldflags=-L/usr/local/cuda/lib64
        make -j 10

    Returns:
        See :code:`read_video`
    """
    if interp is None:
        interp = "super" if use_cuda else "area"
    try:
        raw_width, raw_height, avg_frame_rate, codec_name, start_time = get_video_properties(
            filepath)
        start_time_offset = 0.0 if avg_frame_rate is None else float(
            (1 / (2 * avg_frame_rate)))
        LOGGER.debug(
            "raw_width: %s, raw_height: %s, avg_frame_rate: %s, codec_name: %s, start_time: %s",
            raw_width, raw_height, avg_frame_rate, codec_name, start_time)
        channels = 3
        scale = (min(max_size / raw_width, max_size / raw_height, 1)
                 if max_size is not None else 1)
        width, height = map(lambda d: int(round(scale * d)),
                            [raw_width, raw_height])
        # If there is no average frame rate, the offset tends to be unreliable.
        offset = max(start_time,
                     start_time_offset) if avg_frame_rate is not None else 0
        cmd = (f"{get_ffmpeg()} -hide_banner -an -vsync 0 -loglevel fatal "
               f"-itsoffset -{offset}")
        filters = []
        if draw_timestamps:
            pattern = "%{pts}-%{frame_num}"
            filters.append(f"drawtext=fontsize={int(raw_height * 0.1)}:"
                           f"fontcolor=yellow:text={pattern}"
                           ":x=(w-text_w):y=(h-text_h)")
        # Add frame rate filters.
        if frames_per_second is None:
            seconds_per_frame = float(
                1 / avg_frame_rate) if avg_frame_rate is not None else None
        elif frames_per_second == "keyframes":
            seconds_per_frame = None
            filters.append(r"select=eq(pict_type\,I)")
        else:
            assert isinstance(
                frames_per_second,
                (float, int)), f"Invalid framerate: {frames_per_second}"
            seconds_per_frame = 1 / frames_per_second
            filters.append(f"fps={frames_per_second}:"
                           f"round={frame_rounding}:"
                           f"start_time={offset}")
        # Add resizing filters.
        if use_cuda and codec_name in CUDA_CODECS:
            cuda_codec = CUDA_CODECS[codec_name]
            cmd += f" -hwaccel cuda -c:v {cuda_codec}"
            filters.append("hwupload_cuda")
            if scale != 1:
                filters.append(
                    f"scale_npp={width}:{height}:interp_algo={interp}")
            filters.extend([
                "hwdownload",
                "format=nv12",
            ])
        elif scale != 1:
            filters.append(f"scale={width}:{height}:flags={interp}")
        cmd += f" -i '{filepath}'"
        if filters:
            cmd += " -vf '{fstring}'".format(fstring=",".join(filters))
        cmd += " -pix_fmt rgb24 -f image2pipe -vcodec rawvideo -"
        LOGGER.debug("running ffmpeg with: %s", cmd)
        framebytes = width * height * channels
        bufsize = framebytes * int(
            os.environ.get("PERCEPTION_FFMPEG_BUFSIZE", "5"))
        with subprocess.Popen(
                shlex.split(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=bufsize) as p:
            assert p.stdout is not None, "Could not launch subprocess pipe."
            timestamp: typing.Optional[float] = 0
            frame_index: typing.Optional[int] = 0
            while True:
                batch = p.stdout.read(bufsize)
                if not batch:
                    break
                for image in np.frombuffer(
                        batch, dtype="uint8").reshape((-1, height, width,
                                                       channels)):
                    if frames_per_second != "keyframes":
                        yield (image, frame_index, timestamp)
                        if seconds_per_frame is not None:
                            assert timestamp is not None
                            timestamp += seconds_per_frame
                            frame_index = math.ceil(
                                avg_frame_rate * timestamp
                            ) if avg_frame_rate is not None else None
                        else:
                            timestamp = None
                            frame_index = None
                    else:
                        # Obtaining the keyframe indexes with ffprobe is very slow (slower
                        # than reading the video sometimes). We don't *have* to do it
                        # when using ffmpeg, so we don't. The OpenCV approach *does*
                        # get the keyframe indexes, but only because they're required
                        # in order to select them.
                        yield (image, None, None)
                    if (max_duration is not None and timestamp is not None
                            and timestamp > max_duration):
                        break
            stdout, stderr = p.communicate()
            if p.returncode != 0:
                raise ValueError(
                    f"Error parsing video: {stdout.decode('utf-8')} {stderr.decode('utf-8')}"
                )
    # pylint: disable=broad-except
    except Exception as e:
        if errors not in ["warn", "ignore"]:
            raise e
        if errors == "warn":
            warnings.warn(
                message=
                f"An error occurred while reading {filepath}. Processing may be truncated."
            )


# pylint: disable=too-many-branches,too-many-locals,too-many-statements
def read_video_to_generator(
        filepath,
        frames_per_second: typing.Optional[typing.Union[str, float]] = None,
        errors='raise',
        max_duration: float = None,
        max_size: int = None) -> FramesWithIndexesAndTimestamps:
    """This is used by :code:`read_video` when :code:`use_ffmpeg` is False (default).

    Args:
        filepath: See :code:`read_video`.
        frames_per_second: See :code:`read_video`.
        errors: See :code:`read_video`.
        max_duration: See :code:`read_video`.
        max_size: See :code:`read_video`.

    Returns:
        See :code:`read_video`.
    """
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
                0, max(1, file_frames_per_second / frames_per_second))
            repeat = file_frames_per_second < frames_per_second
        input_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        input_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if max_size is not None:
            scale = min(max_size / max(input_width, input_height), 1)
        else:
            scale = 1
        target_size: typing.Optional[typing.Tuple[int, int]]
        if scale < 1:
            target_size = (int(scale * input_width), int(scale * input_height))
        else:
            target_size = None
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
            if target_size is not None:
                frame = cv2.resize(
                    frame, target_size, interpolation=cv2.INTER_NEAREST)
            current_timestamp = frame_index / file_frames_per_second
            yield frame, grabbed_frame_count - 1, current_timestamp
            if max_duration is not None and current_timestamp > max_duration:
                break
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


def read_video_into_queue(*args, video_queue, terminate, func, **kwargs):
    # We're inside a thread now and the queue is being read elsewhere.
    try:
        for frame, frame_index, timestamp in func(*args, **kwargs):
            if not terminate.isSet():
                video_queue.put((frame, frame_index, timestamp))
            else:
                break
    finally:
        video_queue.put((None, None, None))


# pylint: disable=too-many-arguments
def read_video(
        filepath,
        frames_per_second: typing.Optional[typing.Union[str, float]] = None,
        max_queue_size=128,
        use_queue=True,
        errors='raise',
        use_ffmpeg=False,
        **kwargs) -> FramesWithIndexesAndTimestamps:
    """Provides a generator of RGB frames, frame indexes, and timestamps from a
    video. This function requires you to have installed ffmpeg. All other
    arguments passed to read_video_to_generator.

    Args:
        filepath: Path to the video file
        frames_per_second: How many frames to provide for
            each second of video. If None, all frames
            are provided. If frames_per_second is "keyframes",
            we use ffmpeg to select I frames from the video.
        max_queue_size: The maximum number of frames to load in the queue
        use_queue: Whether to use a queue of frames during processing
        max_duration: The maximum length of the video to hash.
        max_size: The maximum size of frames to queue
        errors: Whether to 'raise', 'warn', or 'ignore' errors
        use_ffmpeg: Whether to use the FFMPEG CLI to read videos. If True, other
            kwargs (e.g., :code:`use_cuda`) are passed to
            :code:`read_video_to_generator_ffmpeg`.

    Yields:
        (frame, frame_index, timestamp) tuples
    """
    for ffmpeg_kwarg in [
            "interp", "frame_rounding", "draw_timestamps", "use_cuda"
    ]:
        if not use_ffmpeg and ffmpeg_kwarg in kwargs:
            warnings.warn(
                f"{ffmpeg_kwarg} is ignored when use_ffmpeg is False.",
                UserWarning)
            del kwargs[ffmpeg_kwarg]
    generator: typing.Callable[..., FramesWithIndexesAndTimestamps]
    if use_ffmpeg:
        generator = read_video_to_generator_ffmpeg
    else:
        generator = read_video_to_generator
    frame_index: typing.Optional[int]
    timestamp: typing.Optional[float]
    if use_queue:
        video_queue = queue.Queue(
            maxsize=max_queue_size
        )  # type: queue.Queue[typing.Tuple[np.ndarray, int, float]]
        terminate = threading.Event()
        thread = threading.Thread(
            target=read_video_into_queue,
            kwargs={
                'frames_per_second': frames_per_second,
                'func': generator,
                'video_queue': video_queue,
                'filepath': filepath,
                'errors': errors,
                'terminate': terminate,
                **kwargs
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

                # Do it twice for the edge case
                # where the queue is completely
                # full and the end sentinel is
                # blocking.
                video_queue.get_nowait()
            except queue.Empty:
                # It doesn't matter if it's empty.
                pass
            # Wait for the background thread to terminate.
            thread.join()
    else:
        for frame, frame_index, timestamp in generator(
                filepath=filepath,
                frames_per_second=frames_per_second,
                errors=errors,
                **kwargs):
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
    """Return bounds of non-trivial region of image or None.

    Unletterboxing is cropping an image such that trivial edge regions
    are removed. Trivial in this context means that the majority of
    the values in that row or column are zero or very close to
    zero. This is why we don't use the terms "non-blank" or
    "non-empty."

    In order to do unletterboxing, this function returns bounds in the
    form (x1, x2), (y1, y2) where:

    - x1 is the index of the first column where over 10% of the pixels
      have means (average of R, G, B) > 2.
    - x2 is the index of the last column where over 10% of the pixels
      have means > 2.
    - y1 is the index of the first row where over 10% of the pixels
      have means > 2.
    - y2 is the index of the last row where over 10% of the pixels
      have means > 2.

    If there are zero columns or zero rows where over 10% of the
    pixels have means > 2, this function returns `None`.

    Note that in the case(s) of a single column and/or row of
    non-trivial pixels that it is possible for x1 = x2 and/or y1 = y2.

    Consider these examples to understand edge cases.  Given two
    images, `L` (entire left and bottom edges are 1, all other pixels
    0) and `U` (left, bottom and right edges 1, all other pixels 0),
    `unletterbox(L)` would return the bounds of the single bottom-left
    pixel and `unletterbox(U)` would return the bounds of the entire
    bottom row.

    Consider `U1` which is the same as `U` but with the bottom two
    rows all 1s. `unletterbox(U1)` returns the bounds of the bottom
    two rows.

    Args:
        image: The image from which to remove letterboxing.

    Returns:
        A pair of coordinates bounds of the form (x1, x2)
        and (y1, y2) representing the left, right, top, and
        bottom bounds.

    """
    # adj should be thought of as a boolean at each pixel indicating
    # whether or not that pixel is non-trivial (True) or not (False).
    adj = image.mean(axis=2) > 2

    if adj.all():
        return (0, image.shape[1] + 1), (0, image.shape[0] + 1)

    y = np.where(adj.sum(axis=1) > 0.1 * image.shape[1])[0]
    x = np.where(adj.sum(axis=0) > 0.1 * image.shape[0])[0]

    if len(y) == 0 or len(x) == 0:
        return None

    if len(y) == 1:
        y1 = y2 = y[0]
    else:
        y1, y2 = y[[0, -1]]
    if len(x) == 1:
        x1 = x2 = x[0]
    else:
        x1, x2 = x[[0, -1]]
    bounds = (x1, x2 + 1), (y1, y2 + 1)

    return bounds
