import concurrent.futures
import typing
import warnings
from abc import ABC, abstractmethod
from logging import warning
from typing import Optional

import numpy as np
import scipy.spatial
import tqdm

from perception.hashers import tools


class Hasher(ABC):
    """All hashers implement a common set of methods from
    the Hasher base class.
    """

    #: The metric to use when computing distance between two hashes. All hashers
    #: must supply this parameter.
    distance_metric: str

    #: The numpy type to use when converting from string to array form.
    #: All hashers must supply this parameter.
    dtype: str

    #: Indicates the length of the hash vector
    hash_length: int

    #: Whether or not this hash returns multiple values
    returns_multiple: bool = False

    #: Indicates whether the hashes can be computed in parallel
    allow_parallel: bool = True

    def string_to_vector(self, hash_string: str, hash_format: str = "base64"):
        """Convert hash string to vector.

        Args:
            hash_string: The input hash string
            hash_format: One of 'base64' or 'hex'
        """
        return tools.string_to_vector(
            hash_string,
            dtype=self.dtype,
            hash_length=self.hash_length,
            hash_format=hash_format,
        )

    def vector_to_string(
        self, vector: np.ndarray, hash_format: str = "base64"
    ) -> typing.Optional[str]:
        """Convert vector to hash string.

        Args:
            vector: Input vector
            hash_format: One of 'base64' or 'hex'
        """
        return tools.vector_to_string(vector, dtype=self.dtype, hash_format=hash_format)

    def compute_distance(
        self,
        hash1: typing.Union[np.ndarray, str],
        hash2: typing.Union[np.ndarray, str],
        hash_format="base64",
    ):
        """Compute the distance between two hashes.

        Args:
            hash1: The first hash or vector
            hash2: The second hash or vector
            hash_format: If either or both of the hashes are hash strings,
                what format the string is encoded in.
        """
        hash1 = (
            self.string_to_vector(hash1, hash_format=hash_format)
            if isinstance(hash1, str)
            else hash1
        )  # makes mypy happy
        hash2 = (
            self.string_to_vector(hash2, hash_format=hash_format)
            if isinstance(hash2, str)
            else hash2
        )

        if self.distance_metric == "sqeuclidean":
            return scipy.spatial.distance.sqeuclidean(
                hash1.astype("float32"), hash2.astype("float32")
            )
        if self.distance_metric == "euclidean":
            return scipy.spatial.distance.euclidean(
                hash1.astype("float32"), hash2.astype("float32")
            )
        if self.distance_metric == "hamming":
            return scipy.spatial.distance.hamming(hash1, hash2)
        if self.distance_metric == "cosine":
            return scipy.spatial.distance.cosine(
                hash1.astype("float32"), hash2.astype("float32")
            )
        if self.distance_metric == "custom":
            return self._compute_distance(hash1, hash2)
        raise NotImplementedError(
            f"Distance metric: {self.distance_metric} not supported."
        )

    def _compute_distance(self, vector1, vector2):
        raise ValueError("Called a custom distance function but it is not implemented.")

    @typing.no_type_check
    def compute_parallel(
        self,
        filepaths: typing.List[str],
        progress: Optional["tqdm.tqdm"] = None,
        progress_desc: Optional[str] = None,
        max_workers: int = 5,
        isometric: bool = False,
    ):
        """Compute hashes in a parallelized fashion.

        Args:
            filepaths: A list of paths to images or videos (depending on the hasher).
            progress: A tqdm-like wrapper for reporting progress. If None,
                progress is not reported.
            progress_desc: The title of the progress bar.
            max_workers: The maximum number of workers
            isometric: Whether to compute all eight isometric transforms for
                each image.
        """
        if not self.allow_parallel and max_workers != 1:
            warnings.warn(
                message="This hash cannot be used in parallel. Setting max_workers to 1.",
                category=UserWarning,
            )
            max_workers = 1
        assert all(
            isinstance(p, str) for p in filepaths
        ), "All images should be provided as paths."

        if isinstance(self, VideoHasher) and isometric:
            raise ValueError("Computing isometric hashes for videos is not supported.")

        # We can use a with statement to ensure threads are cleaned up promptly
        records = []
        if isinstance(self, VideoHasher):
            executor_class = concurrent.futures.ProcessPoolExecutor
        else:
            executor_class = concurrent.futures.ThreadPoolExecutor
        with executor_class(max_workers=max_workers) as executor:
            # Start the load operations and mark each future with its filepath
            compute: typing.Callable = (
                self.compute_isometric if isometric else self.compute
            )
            future_to_path: dict = {
                executor.submit(compute, path): path for path in filepaths
            }
            generator = concurrent.futures.as_completed(future_to_path)
            if progress is not None:
                generator = progress(
                    generator, total=len(filepaths), desc=progress_desc
                )
            for future in generator:
                path = future_to_path[future]
                try:
                    hash_value = future.result()
                except Exception as exc:
                    records.append({"filepath": path, "hash": None, "error": str(exc)})
                else:
                    records.append(
                        {"filepath": path, "hash": hash_value, "error": None}
                    )
        return records


class ImageHasher(Hasher):
    @abstractmethod
    def _compute(self, image: np.ndarray) -> np.ndarray:
        """Compute hash from an image.

        Args:
            image: A numpy array representing an image as
                of shape (H, W, 3) where channels are ordered
                as RGB or a filepath to an image.
        """

    def compute_isometric_from_hash(self, hash_string_or_vector, hash_format="base64"):
        """For supported hashes, obtain the hashes for the dihedral transformations
        of the original image. They are provided in the following order:

        - Vertical flip
        - Horizontal flip
        - 180 degree rotation
        - 90 degree rotation
        - 90 degree rotation and vertical flip
        - 90 degree rotation and horizontal flip
        - 270 degree rotation

        Args:
            hash_string_or_vector: The hash string or vector
            hash_format: One 'base64' or 'hex'
        """
        if not hasattr(self, "_compute_isometric_from_hash"):
            raise NotImplementedError("This hasher does not support hash rotation.")
        rotations = self._compute_isometric_from_hash(  # type: ignore
            hash_string_or_vector
            if isinstance(hash_string_or_vector, np.ndarray)
            else self.string_to_vector(hash_string_or_vector, hash_format=hash_format)
        )
        return {
            transform_name: self.vector_to_string(vector, hash_format=hash_format)
            for transform_name, vector in rotations.items()
        }

    def compute_isometric(self, image: tools.ImageInputType):
        image = tools.to_image_array(image)
        if hasattr(self, "_compute_isometric"):
            hashes = self._compute_isometric(image)  # type: ignore
        elif hasattr(self, "_compute_isometric_from_hash"):
            hashes = self._compute_isometric_from_hash(  # type: ignore
                self._compute(image)
            )
        else:
            transforms = tools.get_isometric_transforms(image)
            for name, transform in transforms.items():
                transforms[name] = self._compute(transform)
            hashes = transforms
        return {
            transform_name: self.vector_to_string(vector)
            for transform_name, vector in hashes.items()
        }

    def compute(
        self, image: tools.ImageInputType, hash_format="base64"
    ) -> typing.Union[
        np.ndarray, typing.Optional[str], typing.List[typing.Optional[str]]
    ]:
        """Compute a hash from an image.

        Args:
            image: An image represented as a filepath, a PIL image object,
                or as an np.ndarray object. If it is an np.ndarray object,
                it must be in RGB color order (note the OpenCV default is
                BGR).
            hash_format: One 'base64', 'hex', or 'vector'
        """
        vector = self._compute(tools.to_image_array(image))
        if hash_format == "vector":
            # Take care of this separately because we took out `vector`
            # as valid return type to vector_to_string().
            # The .tolist() might seem unnecessary for the
            # ndarray `vector` but downstream expects a list and it
            # stays consistent with original, so keeping for now.
            # return (vector.tolist() if self.returns_multiple
            #        else vector)
            return vector  # should iterate the same as vector.tolist()
        if self.returns_multiple:
            return [self.vector_to_string(v, hash_format=hash_format) for v in vector]
        return self.vector_to_string(vector, hash_format=hash_format)

    def compute_with_quality(
        self, image: tools.ImageInputType, hash_format="base64"
    ) -> typing.Tuple[
        typing.Union[
            np.ndarray, typing.Optional[str], typing.List[typing.Optional[str]]
        ],
        int,
    ]:
        """Compute hash and hash quality from image.

        Args:
            image: An image represented as a filepath, a PIL image object,
                or as an np.ndarray object. If it is an np.ndarray object,
                it must be in RGB color order (note the OpenCV default is
                BGR).
            hash_format: One 'base64', 'hex', or 'vector'

        Returns:
            A tuple of (hash, quality)
        """
        vector, quality = self._compute_with_quality(tools.to_image_array(image))
        if hash_format == "vector":
            return vector, quality
        if self.returns_multiple:
            return (
                [self.vector_to_string(v, hash_format=hash_format) for v in vector],
                quality,
            )
        return (self.vector_to_string(vector, hash_format=hash_format), quality)

    def _compute_with_quality(self, image: np.ndarray) -> typing.Tuple[np.ndarray, int]:
        return self._compute(image), tools.compute_quality(image)


class VideoHasher(Hasher):

    #: The frame rate at which videos are read
    frames_per_second: float = 1

    @abstractmethod
    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: typing.Optional[int],
        frame_timestamp: typing.Optional[float],
        state: Optional[dict] = None,
    ) -> dict:
        """Called for each frame in the video. For all
        but the first frame, a state is provided recording the state from
        the previous frame.

        Args:
            frame: The current frame as an RGB ndarray
            frame_index: The current frame index
            frame_timestamp: The current frame timestamp
            state: The state from the last call to process_frame
        """

    @abstractmethod
    def hash_from_final_state(self, state: dict) -> np.ndarray:
        """Called after all frames have been processed. Returns the final
        feature vector.

        Args:
            state: The state dictionary at the end of processing.
        """

    def compute_with_timestamps(
        self, filepath, errors="raise", hash_format="base64", **kwargs
    ):
        scenes: typing.List[dict] = []
        hashes = self.compute(filepath, errors, hash_format, scenes, **kwargs)
        return [
            {
                "hash": hashes[i],
                "start_timestamp": scene.get("start_timestamp"),
                "end_timestamp": scene.get("end_timestamp"),
                "frame_index": scene.get("frame_index"),
            }
            for i, scene in enumerate(scenes)
        ]

    def compute(
        self,
        filepath,
        errors="raise",
        hash_format="base64",
        scenes=None,
        **kwargs,
    ):
        """Compute a hash for a video at a given filepath. All
        other arguments are passed to perception.hashers.tools.read_video.

        Args:
            filepath: Path to video file
            errors: One of "raise", "ignore", or "warn". Passed
                to perception.hashers.tools.read_video.
            hash_format: One of "vector", "base64", or "hex"
            max_duration: The maximum length of the video to hash.
            max_size: The maximum size of frames to queue
            scenes: An array used to pass scene info back to wrapper
                functions
        """
        frame_timestamp, state = None, None
        # Iterate through the video, aggregating scene info in the state
        # dict
        for frame, frame_index, frame_timestamp in tools.read_video(
            filepath=filepath,
            frames_per_second=self.frames_per_second,
            errors=errors,
            **kwargs,
        ):
            state = self.process_frame(
                frame=frame,
                frame_index=frame_index,
                frame_timestamp=frame_timestamp,
                state=state,
            )

        if state is None:
            if errors == "raise":
                raise ValueError(
                    f"Video processing failed for {filepath}, State is None."
                )
            if errors == "warn":
                warning(f"Video processing failed for {filepath}, State is None.")

            return None

        # Persist the final timestamp in the state to allow us to pass along
        # duration
        state["end"] = frame_timestamp
        vectors = self.hash_from_final_state(state=state)
        if scenes is not None:
            scenes += state.get("scenes", [])
        if hash_format == "vector":
            # Take care of this separately because we took out `vector`
            # as valid return type to vector_to_string().
            # The .tolist() might seem unnecessary for the
            # ndarray `vector` but downstream expects a list and it
            # stays consistent with original, so keeping for now.
            # return (vector.tolist() if self.returns_multiple
            #        else vector)
            return vectors  # should iterate the same as vector.tolist()
        if self.returns_multiple:
            return [self.vector_to_string(v, hash_format=hash_format) for v in vectors]
        return self.vector_to_string(vectors, hash_format=hash_format)
