# pylint: disable=too-many-arguments,too-many-locals

import os
import json
import base64
import typing
import warnings
import urllib.parse
import urllib.request

try:
    import tqdm  # pylint: disable=unused-import
except ImportError:  # pragma: no cover
    tqdm = None
from scipy import spatial
import numpy as np

from . import hashers as perception_hashers
from .utils import flatten
try:
    from . import extensions  # type: ignore
except ImportError:
    warnings.warn(
        'C extensions were not built. Some metrics will be computed more slowly. '
        'Please install from wheels or set up a compiler prior to installation '
        'from source to use extensions.')
    extensions = None


def deduplicate_hashes(
        hashes: typing.List[typing.Tuple[str, typing.Union[str, np.ndarray]]],
        threshold: float,
        hash_format: str = 'base64',
        hasher: perception_hashers.ImageHasher = None,
        hash_length: int = None,
        hash_dtype: str = None,
        distance_metric: str = None,
        progress: 'tqdm.tqdm' = None) -> typing.List[typing.Tuple[str, str]]:
    """Find duplicates using a list of precomputed hashes.

    Args:
        hashes: A list of (id, hash) tuples
        threshold: A distance threshold
        hasher: A hasher to use for computing distances
        progress: A tqdm object for reporting progress

    Returns:
        A list of duplicated id pairs. To use, you can just remove the
        first entry of each pair from your dataset. The pairs are provided
        in the event that you wish to apply further analysis.
    """
    assert (
        hash_length is not None and
        hash_dtype is not None and
        distance_metric is not None
    ) or (
        hasher is not None
    ), \
        (
            'You must provide either `hasher` or all of '
            '`hash_length`, `hash_dtype`, and `distance_metric`.'
        )
    if hasher is not None:
        assert all(k is None for k in [hash_length, hash_dtype, distance_metric]), \
            'If hasher is provided, hash_length, hash_dtype, and distance_metric must all be None.'
        hash_length = hasher.hash_length
        hash_dtype = hasher.dtype
        distance_metric = hasher.distance_metric
    assert hash_length is not None
    assert isinstance(hash_dtype, str)
    assert isinstance(distance_metric, str)
    vectors = np.array([
        perception_hashers.tools.string_to_vector(
            hash_string=hash_string_or_vector,
            hash_format=hash_format,
            hash_length=hash_length,
            dtype=hash_dtype)
        if isinstance(hash_string_or_vector, str) else hash_string_or_vector
        for _, hash_string_or_vector in hashes
    ])
    files = np.array([identifier for identifier, _ in hashes])
    pairs: typing.List[typing.Tuple[str, str]] = []
    n_hashes = len(vectors)
    iterator = range(n_hashes)
    if progress is not None:
        iterator = progress(iterator, total=n_hashes, desc='Deduplicating.')
    start_idx = 0
    end_idx = None
    if distance_metric != 'euclidean' or 'int' not in hash_dtype or extensions is None:
        distances = spatial.distance.pdist(vectors, metric=distance_metric)
        for hash_index in iterator:
            if end_idx is not None:
                start_idx = end_idx
            end_idx = start_idx + (n_hashes - hash_index - 1)
            current_distances = distances[start_idx:end_idx]
            duplicated_files = files[hash_index +
                                     1:][current_distances < threshold]
            current_file = files[hash_index]
            # We have to make sure the two files are not the same file
            # because it can happen for highly symmetric images when
            # we are including isometric hashes.
            pairs.extend([(current_file, duplicated_file)
                          for duplicated_file in duplicated_files
                          if duplicated_file != current_file])
    else:
        duplicated = extensions.compute_euclidean_pairwise_duplicates(
            vectors.astype('int32'), threshold=threshold).astype('bool')
        for hash_index in iterator:
            if end_idx is not None:
                start_idx = end_idx
            end_idx = start_idx + (n_hashes - hash_index - 1)
            current_duplicated = duplicated[start_idx:end_idx]
            current_file = files[hash_index]
            duplicated_files = files[hash_index + 1:][current_duplicated]
            pairs.extend([(current_file, duplicated_file)
                          for duplicated_file in duplicated_files
                          if duplicated_file != current_file])
    return list(set(pairs))


# pylint: disable=too-many-locals
def deduplicate(
        files: typing.List[str],
        hashers: typing.List[typing.Tuple[perception_hashers.
                                          ImageHasher, float]],
        isometric: bool = False,
        progress: 'tqdm.tqdm' = None) -> typing.List[typing.Tuple[str, str]]:
    """Find duplicates in a list of files.

    Args:
        files: A list of filepaths.
        hashers: A list of tuples of the form (hasher, threshold)
        isometric: Whether to compare the rotated versions of the images
        progress: A tqdm progress indicator

    Returns:
        A list of duplicated file pairs. To use, you can just remove the
        first entry of each pair from your dataset. The pairs are provided
        in the event that you wish to apply further analysis.
    """
    files_dedup = set(files)
    if len(files_dedup) != len(files):
        warnings.warn(
            message=
            "Duplicate file paths were provided. These will be automatically removed.",
            category=UserWarning)
        files = list(files_dedup)
    pairs: typing.List[typing.Tuple[str, str]] = []
    for hasher_idx, (hasher, threshold) in enumerate(hashers):
        hash_dicts = hasher.compute_parallel(
            filepaths=files,
            progress=progress,
            progress_desc=
            f"Computing hashes for hash {hasher_idx+1} of {len(hashers)}.",
            isometric=isometric)
        hash_list = sorted(hash_dicts, key=lambda h: h['filepath'])
        if isometric:
            hash_list = flatten([
                list(row['hash'].values()) for row in hash_dicts
                if row['error'] is None
            ])
            files_for_hashes = flatten([[row['filepath']] * 8
                                        for row in hash_dicts
                                        if row['error'] is None])
        elif hasher.returns_multiple:
            hash_list = flatten(
                [row['hash'] for row in hash_dicts if row['error'] is None])
            files_for_hashes = flatten([[row['filepath']] * 8
                                        for row in hash_dicts
                                        if row['error'] is None])
        else:
            hash_list = [
                row['hash'] for row in hash_dicts if row['error'] is None
            ]
            files_for_hashes = [
                row['filepath'] for row in hash_dicts if row['error'] is None
            ]
        pairs.extend(
            deduplicate_hashes(
                hashes=list(zip(files_for_hashes, hash_list)),
                hasher=hasher,
                threshold=threshold,
                progress=progress))
    return list(set(pairs))


# pylint: disable=too-few-public-methods
class SaferMatcher:
    """An object for matching hashes with the known CSAM hashes in the
    Safer matching service.
    Please contact `info@getsafer.io <mailto:info@getsafer.io>`_
    for details on obtaining credentials and information on how match
    responses are provided.

    Here's a minimalist example:

    .. code-block:: python

        from perception import hashers, tools

        hasher = hashers.PHash(hash_size=16)
        matches = hashers.tools.SaferMatcher(
            api_key='YOUR_API_KEY',
            username='YOUR_USERNAME', # You only need to provide
            password='YOUR_PASSWORD', # an API key OR username/password.
            url='MATCHING_SERVICE_URL'
        )

    For authentication, you must provide the API key OR username and password pair.
    If neither is provided, the function will attempt to find them as environment
    variables with names :code:`SAFER_MATCHING_SERVICE_API_KEY`,
    :code:`SAFER_MATCHING_SERVICE_USERNAME`, and :code:`SAFER_MATCHING_SERVICE_PASSWORD`,
    respectively. You must also provide the URL endpoint for the matching service,
    either as a keyword argument or as a :code:`SAFER_MATCHING_SERVICE_URL`
    environment variable.

    Args:
        api_key: A base64 encoded set of matching service credentials
        username: Matching service username
        password: Matching service password
        url: Safer matching service URL
        hasher: A hasher to use for matching
        hasher_api_id: The hasher ID for finding matches.
        quality_threshold: The quality threshold filter to use
    """

    def __init__(self,
                 api_key: str = None,
                 username: str = None,
                 password: str = None,
                 url: str = None,
                 hasher: perception_hashers.ImageHasher = None,
                 hasher_api_id: str = None,
                 quality_threshold: int = 90):
        if (username is None and password is None and api_key is None and
                os.environ.get('SAFER_MATCHING_SERVICE_USERNAME') is not None
                and
                os.environ.get('SAFER_MATCHING_SERVICE_PASSWORD') is not None):
            username = os.environ['SAFER_MATCHING_SERVICE_USERNAME']
            password = os.environ['SAFER_MATCHING_SERVICE_PASSWORD']
        if username is not None and password is not None:
            credentials = ('%s:%s' % (username, password))
            api_key = base64.b64encode(
                credentials.encode('ascii')).decode('ascii')
        if api_key is None:
            api_key = os.environ.get('SAFER_MATCHING_SERVICE_API_KEY')
            if api_key is None:
                raise ValueError(
                    'You must provide one of (1) API key, (2) API key provided as '
                    '`SAFER_MATCHING_SERVICE_API_KEY` env var, (3) username and password or '
                    '(4) username and password as `SAFER_MATCHING_SERVICE_USERNAME` and '
                    '`SAFER_MATCHING_SERVICE_PASSWORD` env vars.')
        if url is None:
            url = os.environ.get('SAFER_MATCHING_SERVICE_URL')
            if url is None:
                raise ValueError(
                    'You must provide either the url or the SAFER_MATCHING_SERVICE_URL env var.'
                )
        if urllib.parse.urlparse(url).scheme != 'https' and not os.environ.get(
                'SAFER_MATCHING_SERVICE_DEV_ALLOW_HTTP'):
            raise ValueError(
                'You must provide an url that begins with `https://`.')
        self.api_key = api_key
        self.url = url
        if hasher is None:
            hasher = perception_hashers.PHash(hash_size=16, highfreq_factor=4)
        if hasher_api_id is None:
            hasher_api_id = 'phash'
        self.hasher = hasher
        self.hasher_api_id = hasher_api_id
        self.quality_threshold = quality_threshold

    def match(self, images: typing.List[
            typing.Union[str, typing.Tuple[perception_hashers.tools.
                                           ImageInputType, str]]]) -> dict:
        """Match hashes with the Safer matching service.

        Args:
            images: A list of image filepaths or (image_like, image_id) tuples.

        Returns:
            A dictionary of matches. See Safer matching service documentation (
            contact Thorn for a copy).
        """
        raw_hashes = [
            self.hasher.compute_with_quality(
                image if isinstance(image, str) else image[0])
            for image in images
        ]
        hashes = [{
            'id':
            image if isinstance(image, str) else image[1],
            self.hasher_api_id:
            hash_string,
            'md5':
            perception_hashers.tools.compute_md5(image) if isinstance(
                image, str) else
            (perception_hashers.tools.compute_md5(image[0]) if isinstance(
                image[0], str) else None)
        } for image, (hash_string, quality) in zip(images, raw_hashes)
                  if quality > self.quality_threshold]
        for hash_dict in hashes:
            # We cannot include an md5 key if we don't
            # have the md5.
            if hash_dict['md5'] is None:
                del hash_dict['md5']
        if not hashes:
            warnings.warn(
                message="No images of sufficient quality were found.",
                category=UserWarning)
            return {}
        body = {'hashes': hashes, 'version': 'v2'}
        headers = {
            'Authorization': f'Basic {self.api_key}',
            'Content-Type': 'application/json'
        }
        req = urllib.request.Request(
            url=self.url,
            data=str(json.dumps(body)).encode('utf-8'),
            headers=headers,
            method='POST')
        res = urllib.request.urlopen(req)
        return json.loads(res.read().decode('utf-8'))
