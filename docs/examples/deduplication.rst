Media Deduplication
*******************

Perceptual hashes can be used to deduplicate sets of images. Below we provide two examples (one simple, one larger scale).

**For most use cases, we recommend using PHash with** :code:`hash_size=16` **and
with 0.2 as the distance threshold as in the example below.** You may wish to adjust
this threshold up or down based on your tolerance for false negatives / positives.

In practice, deduplicating in memory on your machine by the methods below may be impractical.
For larger-scale applications, you may wish to use tools like
`FAISS <https://github.com/facebookresearch/faiss>`_,
`Annoy <https://github.com/spotify/annoy>`_, or databases with
functionality for querying based on distance such as
`MemSQL <https://docs.memsql.com/sql-reference/v6.8/euclidean_distance/>`_.

For the supported hashers, below are our recommended thresholds with expected false positive rates of <1%.

======================  ===========
hasher                  threshold
======================  ===========
ahash (hash_size=16)    0.008
blockmean               0.008
dhash (hash_size=16)    0.07
marrhildreth            0.1
pdq                     0.2
phash (hash_size=16)    0.2
wavelet (hash_size=16)  0.02
======================  ===========

Simple example
==============

In this example, we download a ZIP file containing 18 images. One of the images is duplicated
twice and another image is duplicated once.

.. code-block:: python

    import os
    import glob
    import zipfile
    import urllib.request

    import tabulate
    import pandas as pd

    from perception import tools, hashers

    urllib.request.urlretrieve(
        "https://thorn-perception.s3.amazonaws.com/thorn-perceptual-deduplication-example.zip",
        "thorn-perceptual-deduplication-example.zip"
    )

    with zipfile.ZipFile('thorn-perceptual-deduplication-example.zip') as f:
        f.extractall('.')
        
    filepaths = glob.glob('thorn-perceptual-deduplication-example/*.jpg')
    duplicate_pairs = tools.deduplicate(files=filepaths, hashers=[(hashers.PHash(hash_size=16), 0.2)])
    print(tabulate.tabulate(pd.DataFrame(duplicate_pairs), showindex=False, headers=['file1', 'file2'], tablefmt='rst'))
    
    # Now we can do whatever we want with the duplicates. We could just delete
    # the first entry in each pair or manually verify the pairs to ensure they
    # are, in fact duplicates.


===============================================  ===============================================
file1                                            file2
===============================================  ===============================================
thorn-perceptual-deduplication-example/309b.jpg  thorn-perceptual-deduplication-example/309.jpg
thorn-perceptual-deduplication-example/309b.jpg  thorn-perceptual-deduplication-example/309a.jpg
thorn-perceptual-deduplication-example/309a.jpg  thorn-perceptual-deduplication-example/309.jpg
thorn-perceptual-deduplication-example/315a.jpg  thorn-perceptual-deduplication-example/315.jpg
===============================================  ===============================================

Real-world example
==================

In the example below, we use the 
`Caltech 256 Categories <http://www.vision.caltech.edu/Image_Datasets/Caltech256>`_ dataset. Like
most other public image datasets, it contains a handful of duplicates in some categories.

The code below will:

1. Download the dataset
2. Group all the filepaths by category (the dataset is provided in folders)
3. Within each group, find duplicates using PHash. We will compare not just the
   original images, but also the 8 isometric transformations for each image.

.. code-block:: python

    import os
    import tarfile
    from glob import glob
    import urllib.request

    import tqdm

    from perception import hashers, tools

    urllib.request.urlretrieve(
        "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar",
        "256_ObjectCategories.tar"
    )
    with tarfile.open('256_ObjectCategories.tar') as tfile:
        tfile.extractall()

    files = glob('256_ObjectCategories/**/*.jpg')

    # To reduce the number of pairwise comparisons,
    # we can deduplicate within each image category
    # (i.e., we don't need to compare images of 
    # butterflies with images of chess boards).
    filepath_group = [
        (
            filepath,
            os.path.normpath(filepath).split(os.sep)[-2]
        ) for filepath in files
    ]
    groups = list(set([group for _, group in filepath_group]))
    
    # We consider any pair of images with a PHash distance of < 0.2 as
    # as a duplicate.
    comparison_hashers = [(hashers.PHash(hash_size=16), 0.2)]

    duplicate_pairs = []

    for current_group in groups:
        current_filepaths = [
            filepath for filepath, group in filepath_group if group == current_group
        ]
        current_duplicate_pairs = tools.deduplicate(
            files=current_filepaths,
            hashers=comparison_hashers,
            isometric=True,
            progress=tqdm.tqdm
        )
        duplicate_pairs.extend(current_duplicate_pairs)

    # Now we can do whatever we want with the duplicates. We could just delete
    # the first entry in each pair or manually verify the pairs to ensure they
    # are, in fact duplicates.

Video deduplication
===================

Video deduplication requires more thought depending on your tolerance for false positives and
how important temporal relationships are. Below is one example approach for deduplicating a
group of videos by taking frames from each video that are sufficiently different from each other
(to avoid keeping too many) and then using them all to find
pairs of videos that have matching frames.

.. code-block:: python

    import urllib.request
    import zipfile

    import glob
    import tqdm

    import perception.hashers

    # Download some example videos.
    urllib.request.urlretrieve(
        "https://thorn-perception.s3.amazonaws.com/thorn-perceptual-video-deduplication-example.zip",
        "thorn-perceptual-video-deduplication-example.zip"
    )

    with zipfile.ZipFile('thorn-perceptual-video-deduplication-example.zip') as f:
        f.extractall('.')

    # By default, this will use TMK L1 with PHashU8.
    hasher = perception.hashers.SimpleSceneDetection(max_scene_length=5)

    # Set a threshold for matching frames within videos and across videos.
    filepaths = glob.glob('thorn-perceptual-video-deduplication-example/*.m4v') + \
                glob.glob('thorn-perceptual-video-deduplication-example/*.gif')

    # Returns a list of dicts with a "filepath" and "hash" key. "hash" contains a
    # list of hashes.
    hashes = hasher.compute_parallel(filepaths=filepaths, progress=tqdm.tqdm)


    # Flatten the hashes into a list of (filepath, hash) tuples.
    hashes_flattened = perception.tools.flatten([
        [(hash_group['filepath'], hash_string) for hash_string in hash_group['hash']]
        for hash_group in hashes
    ])

    duplicates = perception.tools.deduplicate_hashes(
        hashes=hashes_flattened,
        threshold=50,
        hasher=hasher
    )