# pylint: disable=too-many-arguments,too-many-branches

import os
import typing

import cv2
import ffmpeg

from ..hashers.tools import read_video


def probe(filepath):
    """Get the output of ffprobe."""
    return ffmpeg.probe(filepath)


def sanitize_output_filepath(input_filepath, output_filepath, output_ext=None):
    """Get a suitable output filepath with an extension based on
    an input filepath.

    Args:
        input_filepath: The filepath for the source file.
        output_filepath: The filepath for the output file.
        output_ext: A new extension to add (e.g., '.gif')
    """
    _, input_ext = os.path.splitext(input_filepath)
    if not output_filepath.lower().endswith(output_ext or input_ext):
        output_filepath += output_ext or input_ext
    return output_filepath


def get_simple_transform(width: typing.Union[str, int] = -1,
                         height: typing.Union[str, int] = -1,
                         pad: str = None,
                         codec: str = None,
                         clip_pct: typing.Tuple[float, float] = None,
                         clip_s: typing.Tuple[float, float] = None,
                         sar=None,
                         fps=None,
                         output_ext=None):
    """Resize to a specific size and re-encode.

    Args:
        width: The target width (-1 to maintain aspect ratio)
        height: The target height (-1 to maintain aspect ratio)
        pad: An ffmpeg pad argument provided as a string.
        codec: The codec for encoding the video.
        fps: The new frame rate for the video.
        clip_pct: The video start and end in percentages of video duration.
        clip_s: The video start and end in seconds (used over clip_pct if both
            are provided).
        sar: Whether to make all videos have a common sample aspect
            ratio (i.e., for all square pixels, set this to '1/1').
        output_ext: The extension to use when re-encoding (used to select
            video format). It should include the leading '.'.
    """

    def transform(input_filepath, output_filepath):
        output_filepath = sanitize_output_filepath(input_filepath,
                                                   output_filepath, output_ext)
        data = None
        if codec is None:
            data = (data or probe(input_filepath))
            output_codec = [
                s for s in data['streams'] if s['codec_type'] == 'video'
            ][0]['codec_name']
        else:
            output_codec = codec
        format_kwargs = {'codec:v': output_codec}
        if clip_pct is not None or clip_s is not None:
            pct_start, pct_end, pos_start, pos_end = None, None, None, None
            if clip_pct is not None:
                pct_start, pct_end = clip_pct
            if clip_s is not None:
                pos_start, pos_end = clip_s
            if pct_start is not None:
                assert 0 <= pct_start <= 1, 'Start position must be between 0 and 1.'
            if pct_end is not None:
                assert 0 <= pct_end <= 1, 'End position must be between 0 and 1.'
            if pct_start is not None and pct_end is not None:
                assert pct_start < pct_end, 'End must be greater than start.'
            if (pct_start is not None
                    and pos_start is None) or (pct_end is not None
                                               and pos_end is None):
                # We only want to get the duration for the video if we need
                # it.
                data = data or probe(input_filepath)
                duration = float(data['streams'][0]['duration'])
            if pct_start is not None or pos_start is not None:
                format_kwargs[
                    'ss'] = pos_start or pct_start * duration  # type: ignore
            if pct_end is not None or pos_end is not None:
                format_kwargs[
                    't'] = pos_end or pct_end * duration  # type: ignore
        stream = ffmpeg.input(input_filepath)
        if not (width == -1 and height == -1):
            stream = stream.filter('scale', width, height)
        if pad is not None:
            stream = stream.filter('pad', *pad.split(':'))
        if fps is not None:
            stream = stream.filter('fps', fps)
        if sar is not None:
            stream = stream.filter('setsar', sar)
        stream = stream.output(output_filepath, **format_kwargs) \
                    .overwrite_output()
        ffmpeg.run(stream)
        if os.path.isfile(output_filepath):
            return output_filepath
        return None

    return transform


def get_slideshow_transform(frame_input_rate,
                            frame_output_rate,
                            max_frames=None,
                            offset=0):
    """Get a slideshow transform to create slideshows from
    videos.

    Args:
        frame_input_rate: The rate at which frames will be sampled
            from the source video (e.g., a rate of 1 means we collect
            one frame per second of the input video).
        frame_output_rate: The rate at which the sampled frames are played
            in the slideshow (e.g., a rate of 0.5 means each frame will
            appear for 2 seconds).
        max_frames: The maximum number of frames to write.
        offset: The number of seconds to wait before beginning the slide show.
    """

    def transform(input_filepath, output_filepath):
        output_filepath = sanitize_output_filepath(
            input_filepath, output_filepath, output_ext='.mov')
        writer = None
        frame_count = 0
        try:
            for frame, _, timestamp in read_video(
                    filepath=input_filepath,
                    frames_per_second=frame_input_rate):
                if timestamp < offset:
                    continue
                if writer is None:
                    writer = cv2.VideoWriter(
                        filename=output_filepath,
                        fourcc=cv2.VideoWriter_fourcc(*'mjpg'),
                        fps=frame_output_rate,
                        frameSize=tuple(frame.shape[:2][::-1]),
                        isColor=True)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    break
        finally:
            if writer is not None:
                writer.release()
        if os.path.isfile(output_filepath):
            return output_filepath
        return None

    return transform


def get_black_frame_padding_transform(duration_s=0, duration_pct=0):
    """Get a transform that adds black frames at the start and end
    of a video.

    Args:
        duration_s: The duration of the black frames in seconds.
        duration_pct: The duration of the black frames
            as a percentage of video duration. If both duration_s
            and duration_pct are provided, the maximum value
            is used.
    """

    def transform(input_filepath, output_filepath):
        output_filepath = sanitize_output_filepath(input_filepath,
                                                   output_filepath)
        stream = next(
            stream for stream in probe(input_filepath)['streams']
            if stream['codec_type'] == 'video')
        assert stream['sample_aspect_ratio'] == '1:1', 'SAR is not 1:1.'
        width = stream['width']
        height = stream['height']
        duration = max(duration_s, duration_pct * float(stream['duration']))
        ffmpeg.input(input_filepath).output(
            output_filepath,
            vf=("color=c=black:s={width}x{height}:d={duration} [pre] ; "
                "color=c=black:s={width}x{height}:d={duration} [post] ; "
                "[pre] [in] [post] concat=n=3").format(
                    width=width, height=height,
                    duration=duration)).overwrite_output().run()
        if os.path.isfile(output_filepath):
            return output_filepath
        return None

    return transform
