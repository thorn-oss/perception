Benchmarking
************

This package provides a fair amount of infrastructure for benchmarking different hashers to evaluate their performance.

Image Hashing
=============

The below example does the following:

- Download a benchmarking dataset (we provide a dataset with images that have compatible licensing for this example)
- Load the dataset. If you are using your own datasets, you may wish to call `deduplicate` on it to ensure no duplicates are included.
- Transform the dataset to generate synthetic images.
- Define a new custom hasher that we want to evaluate.
  It's not very good -- but demonstrates how you can evaluate your own custom hash functions.
- Compute all the hashes.
- Report metrics for each image category / hasher / transformation combination.

.. code-block:: python

    import os
    import glob
    import zipfile
    import urllib.request

    import cv2
    import imgaug
    import tabulate # Optional: Only used for generating tables for the Sphinx documentation
    import numpy as np

    from perception import benchmarking, hashers

    urllib.request.urlretrieve(
        "https://thorn-perception.s3.amazonaws.com/thorn-perceptual-benchmark-v0.zip",
        "thorn-perceptual-benchmark-v0.zip"
    )

    with zipfile.ZipFile('thorn-perceptual-benchmark-v0.zip') as f:
        f.extractall('.')
    
    # Load the dataset
    dataset = benchmarking.BenchmarkImageDataset.from_tuples(files=[
        (filepath, filepath.split(os.path.sep)[-2]) for filepath in glob.glob(
            os.path.join('thorn-perceptual-benchmark-v0', '**', '*.jpg')
        )
    ])

    # Define the transforms we want to use for
    # evaluation hash quality.
    def watermark(image):
        fontScale = 5
        thickness = 5
        text = "TEXT"
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        targetWidth = 0.2*image.shape[1]
        (textWidth, textHeight), _ = cv2.getTextSize(
            text="TEST",
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness
        )
        fontScaleCorr = targetWidth / textWidth
        textHeight *= fontScaleCorr
        textWidth *= fontScaleCorr
        fontScale *= fontScaleCorr

        org = ( textHeight, image.shape[0] - textHeight )
        org = tuple(map(int, org))
        color = (0, 0, 0, 200)
        placeholder = cv2.putText(
            img=np.zeros(image.shape[:2] + (4, ), dtype='uint8'),
            text="TEST",
            org=org,
            color=color,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness
        ).astype('float32')
        augmented = (
            (image.astype('float32')[..., :3]*(255 - placeholder[..., 3:]) + placeholder[..., :3]*placeholder[..., 3:])
        ) / 255
        return augmented.astype('uint8')

    def vignette(image):
        height, width = image.shape[:2]
        a = cv2.getGaussianKernel(height, height/2)
        b = cv2.getGaussianKernel(width, width/2)
        c = (b.T*a)[..., np.newaxis]
        d = c/c.max()
        e = image*d
        return e.astype('uint8')

    transforms={
        'watermark': watermark,
        'blur2': imgaug.augmenters.GaussianBlur(sigma=2.0),
        'vignette': vignette,
        'gamma2': imgaug.augmenters.GammaContrast(gamma=2),
        'jpeg95': imgaug.augmenters.JpegCompression(95),
        'pad0.2': imgaug.augmenters.Pad(percent=((0.2, 0.2), (0, 0), (0.2, 0.2), (0, 0)), keep_size=False),
        'crop0.05': imgaug.augmenters.Crop(percent=((0.05, 0.05), (0.05, 0.05), (0.05, 0.05), (0.05, 0.05)), keep_size=False),
        'noise0.2': imgaug.augmenters.AdditiveGaussianNoise(scale=0.2*255),
        'rotate4': imgaug.augmenters.Affine(rotate=4),
        'noop': imgaug.augmenters.Resize({"longer-side": 256, "shorter-side": "keep-aspect-ratio"}),
    }

    # Compute the transformed versions of the images.
    # This takes a while but you can reload the
    # generated dataset without recomputing it (see next line).
    transformed = dataset.transform(
        transforms=transforms,
        storage_dir='transformed',
        errors="raise"
    )
    # We don't actually have to do this, but it shows
    # how to reload the transformed dataset later.
    transformed = benchmarking.BenchmarkImageTransforms.load(
        path_to_zip_or_directory='transformed', verify_md5=False
    )

    # Create a new hash that we want to evaluate.
    # perception will handle most of the plumbing but
    # we do have to specify a few things.
    class ShrinkHash(hashers.Hasher):
        """This is a simple hash to demonstrate how you
        can create your own hasher and compare it to others.
        It just shrinks images to 8x8 pixels and then flattens
        the result.
        """
        
        # We have to let perception know
        # the shape and type of our hash.
        hash_length = 64
        dtype = 'uint8'
        
        # We need to specify how distance is
        # computed between hashes.
        distance_metric = 'euclidean'
        
        def _compute(self, image):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, dsize=(8, 8))
            return resized.flatten()

    hashers_dict = {
        'ahash': hashers.AverageHash(hash_size=16),
        'dhash': hashers.DHash(hash_size=16),
        'pdq': hashers.PDQHash(),
        'phash': hashers.PHash(hash_size=16),
        'marrhildreth': hashers.MarrHildreth(),
        'wavelet': hashers.WaveletHash(hash_size=16),
        'blockmean': hashers.BlockMean(),
        'shrinkhash': ShrinkHash()
    }

    # Compute the hashes
    hashes = transformed.compute_hashes(hashers=hashers_dict)

    # Get performance metrics (i.e., recall) for each hash function based on
    # a false positive rate tolerance threshold. Here we use 0.01%
    fpr_threshold = 1e-4
    
    # The metrics are just pandas dataframes. We use tabulate here to obtain the tables
    # formatted for the documentation.
    metrics = hashes.compute_threshold_recall(fpr_threshold=fpr_threshold).reset_index()
    print(tabulate.tabulate(metrics, showindex=False, headers=metrics.columns, tablefmt='rst'))

    metrics_by_transform = hashes.compute_threshold_recall(grouping=['transform_name'], fpr_threshold=fpr_threshold).reset_index()
    print(tabulate.tabulate(metrics_by_transform, showindex=False, headers=metrics_by_transform.columns, tablefmt='rst'))

    metrics_simple = hashes.compute_threshold_recall(grouping=[], fpr_threshold=fpr_threshold).reset_index()
    print(tabulate.tabulate(metrics_simple, showindex=False, headers=metrics_simple.columns, tablefmt='rst'))



===========  ================  =============  ============  ========  =====  =============
category     transform_name    hasher_name       threshold    recall    fpr    n_exemplars
===========  ================  =============  ============  ========  =====  =============
paintings    blur2             ahash            0.0117188     66.062      0           2204
paintings    blur2             blockmean        0.0134298     87.432      0           2204
paintings    blur2             dhash            0.132812     100          0           2204
paintings    blur2             marrhildreth     0.126736     100          0           2204
paintings    blur2             pdq              0.117188     100          0           2204
paintings    blur2             phash            0.09375      100          0           2204
paintings    blur2             shrinkhash      61.441         43.829      0           2204
paintings    blur2             wavelet          0.015625      65.926      0           2204
paintings    crop0.05          ahash            0.0078125      0.227      0           2204
paintings    crop0.05          blockmean        0.0144628      0.408      0           2204
paintings    crop0.05          dhash            0.222656      11.298      0           2204
paintings    crop0.05          marrhildreth     0.215278       3.857      0           2204
paintings    crop0.05          pdq              0.265625      11.298      0           2204
paintings    crop0.05          phash            0.234375       8.757      0           2204
paintings    crop0.05          shrinkhash      95.5667         2.314      0           2204
paintings    crop0.05          wavelet          0.015625       0.318      0           2204
paintings    gamma2            ahash            0.0078125      2.586      0           2204
paintings    gamma2            blockmean        0.00826446     2.269      0           2204
paintings    gamma2            dhash            0.175781      98.82       0           2204
paintings    gamma2            marrhildreth     0.163194      99.501      0           2204
paintings    gamma2            pdq              0.164062     100          0           2204
paintings    gamma2            phash            0.164062     100          0           2204
paintings    gamma2            shrinkhash     180.69           0.045      0           2204
paintings    gamma2            wavelet          0.015625      18.603      0           2204
paintings    jpeg95            ahash            0.0117188     29.9        0           2204
paintings    jpeg95            blockmean        0.0134298     38.612      0           2204
paintings    jpeg95            dhash            0.191406      92.604      0           2204
paintings    jpeg95            marrhildreth     0.166667      85.844      0           2204
paintings    jpeg95            pdq              0.25         100          0           2204
paintings    jpeg95            phash            0.25         100          0           2204
paintings    jpeg95            shrinkhash      66.7008        46.597      0           2204
paintings    jpeg95            wavelet          0.015625      19.419      0           2204
paintings    noise0.2          ahash            0.0078125      6.352      0           2204
paintings    noise0.2          blockmean        0.0154959     21.779      0           2204
paintings    noise0.2          dhash            0.238281      90.699      0           2204
paintings    noise0.2          marrhildreth     0.166667      72.096      0           2204
paintings    noise0.2          pdq              0.28125       99.501      0           2204
paintings    noise0.2          phash            0.273438      99.909      0           2204
paintings    noise0.2          shrinkhash     154.729          0.635      0           2204
paintings    noise0.2          wavelet          0.0078125      1.407      0           2204
paintings    noop              ahash            0            100          0           2204
paintings    noop              blockmean        0            100          0           2204
paintings    noop              dhash            0            100          0           2204
paintings    noop              marrhildreth     0            100          0           2204
paintings    noop              pdq              0            100          0           2204
paintings    noop              phash            0            100          0           2204
paintings    noop              shrinkhash       0            100          0           2204
paintings    noop              wavelet          0            100          0           2204
paintings    pad0.2            ahash            0.0820312      0.045      0           2204
paintings    pad0.2            blockmean        0.0950413      0.045      0           2204
paintings    pad0.2            dhash            0.214844       1.27       0           2204
paintings    pad0.2            marrhildreth     0.220486       0.045      0           2204
paintings    pad0.2            pdq              0.296875       2.586      0           2204
paintings    pad0.2            phash            0.28125        3.448      0           2204
paintings    pad0.2            shrinkhash     153.981          0.227      0           2204
paintings    pad0.2            wavelet          0.109375       0          0           2204
paintings    rotate4           ahash            0.0429688      4.083      0           2204
paintings    rotate4           blockmean        0.0392562      3.448      0           2204
paintings    rotate4           dhash            0.210938      40.245      0           2204
paintings    rotate4           marrhildreth     0.229167      64.201      0           2204
paintings    rotate4           pdq              0.28125       61.388      0           2204
paintings    rotate4           phash            0.265625      66.924      0           2204
paintings    rotate4           shrinkhash      69.4622         2.858      0           2204
paintings    rotate4           wavelet          0.0390625      0.635      0           2204
paintings    vignette          ahash            0.046875       7.623      0           2204
paintings    vignette          blockmean        0.0485537      8.53       0           2204
paintings    vignette          dhash            0.125         34.256      0           2204
paintings    vignette          marrhildreth     0.177083      77.813      0           2204
paintings    vignette          pdq              0.132812     100          0           2204
paintings    vignette          phash            0.132812     100          0           2204
paintings    vignette          shrinkhash     103.015          3.312      0           2204
paintings    vignette          wavelet          0.0546875      5.172      0           2204
paintings    watermark         ahash            0.0078125     31.307      0           2204
paintings    watermark         blockmean        0.0134298     47.55       0           2204
paintings    watermark         dhash            0.0664062    100          0           2204
paintings    watermark         marrhildreth     0.0711806    100          0           2204
paintings    watermark         pdq              0.28125       99.138      0           2204
paintings    watermark         phash            0.289062      99.682      0           2204
paintings    watermark         shrinkhash     104.723         75.635      0           2204
paintings    watermark         wavelet          0.015625      51.18       0           2204
photographs  blur2             ahash            0.0195312     80.788      0           1650
photographs  blur2             blockmean        0.0330579     97.818      0           1650
photographs  blur2             dhash            0.0898438     96.303      0           1650
photographs  blur2             marrhildreth     0.102431      96.97       0           1650
photographs  blur2             pdq              0.304688      99.939      0           1650
photographs  blur2             phash            0.179688     100          0           1650
photographs  blur2             shrinkhash     116.09          42.303      0           1650
photographs  blur2             wavelet          0.0234375     78.303      0           1650
photographs  crop0.05          ahash            0.0117188      0.242      0           1650
photographs  crop0.05          blockmean        0.0278926      0.848      0           1650
photographs  crop0.05          dhash            0.101562       1.333      0           1650
photographs  crop0.05          marrhildreth     0.175347       3.152      0           1650
photographs  crop0.05          pdq              0.320312      38.485      0           1650
photographs  crop0.05          phash            0.335938      73.394      0           1650
photographs  crop0.05          shrinkhash     128.222          1.212      0           1650
photographs  crop0.05          wavelet          0.0234375      0.424      0           1650
photographs  gamma2            ahash            0.0195312     10.606      0           1650
photographs  gamma2            blockmean        0.0278926     18.242      0           1650
photographs  gamma2            dhash            0.105469      91.636      0           1650
photographs  gamma2            marrhildreth     0.121528      92.303      0           1650
photographs  gamma2            pdq              0.195312     100          0           1650
photographs  gamma2            phash            0.234375     100          0           1650
photographs  gamma2            shrinkhash     121.569          0.545      0           1650
photographs  gamma2            wavelet          0.0234375     19.152      0           1650
photographs  jpeg95            ahash            0.0117188     33.576      0           1650
photographs  jpeg95            blockmean        0.0299587     84.424      0           1650
photographs  jpeg95            dhash            0.117188      77.273      0           1650
photographs  jpeg95            marrhildreth     0.109375      73.333      0           1650
photographs  jpeg95            pdq              0.4375        99.939      0           1650
photographs  jpeg95            phash            0.335938      99.879      0           1650
photographs  jpeg95            shrinkhash     124.78          83.758      0           1650
photographs  jpeg95            wavelet          0.0234375     44.727      0           1650
photographs  noise0.2          ahash            0.0195312     34.909      0           1650
photographs  noise0.2          blockmean        0.036157      72.121      0           1650
photographs  noise0.2          dhash            0.167969      69.03       0           1650
photographs  noise0.2          marrhildreth     0.119792      56.182      0           1650
photographs  noise0.2          pdq              0.34375       99.758      0           1650
photographs  noise0.2          phash            0.320312      99.818      0           1650
photographs  noise0.2          shrinkhash     190.137         24          0           1650
photographs  noise0.2          wavelet          0.0234375     23.03       0           1650
photographs  noop              ahash            0            100          0           1650
photographs  noop              blockmean        0            100          0           1650
photographs  noop              dhash            0            100          0           1650
photographs  noop              marrhildreth     0            100          0           1650
photographs  noop              pdq              0            100          0           1650
photographs  noop              phash            0            100          0           1650
photographs  noop              shrinkhash       0            100          0           1650
photographs  noop              wavelet          0            100          0           1650
photographs  pad0.2            ahash            0.046875       0.121      0           1650
photographs  pad0.2            blockmean        0.0588843      0.061      0           1650
photographs  pad0.2            dhash            0.109375       0.667      0           1650
photographs  pad0.2            marrhildreth     0.190972       0.182      0           1650
photographs  pad0.2            pdq              0.289062       1.515      0           1650
photographs  pad0.2            phash            0.296875       4.606      0           1650
photographs  pad0.2            shrinkhash     164.593          0.121      0           1650
photographs  pad0.2            wavelet          0.0820312      0          0           1650
photographs  rotate4           ahash            0.03125        2.545      0           1650
photographs  rotate4           blockmean        0.0382231      4.242      0           1650
photographs  rotate4           dhash            0.0976562      3.333      0           1650
photographs  rotate4           marrhildreth     0.159722       7.394      0           1650
photographs  rotate4           pdq              0.3125        78.121      0           1650
photographs  rotate4           phash            0.320312      92.182      0           1650
photographs  rotate4           shrinkhash     132.944          4.788      0           1650
photographs  rotate4           wavelet          0.015625       0.182      0           1650
photographs  vignette          ahash            0.03125        9.152      0           1650
photographs  vignette          blockmean        0.0330579     10.242      0           1650
photographs  vignette          dhash            0.0742188     24.606      0           1650
photographs  vignette          marrhildreth     0.0954861     38.606      0           1650
photographs  vignette          pdq              0.117188     100          0           1650
photographs  vignette          phash            0.125        100          0           1650
photographs  vignette          shrinkhash     133.364         10.727      0           1650
photographs  vignette          wavelet          0.0234375      4.424      0           1650
photographs  watermark         ahash            0.0195312     48          0           1650
photographs  watermark         blockmean        0.0258264     59.697      0           1650
photographs  watermark         dhash            0.078125     100          0           1650
photographs  watermark         marrhildreth     0.114583      98.242      0           1650
photographs  watermark         pdq              0.351562      99.879      0           1650
photographs  watermark         phash            0.320312      99.758      0           1650
photographs  watermark         shrinkhash     142.317         78.242      0           1650
photographs  watermark         wavelet          0.0234375     51.515      0           1650
===========  ================  =============  ============  ========  =====  =============

================  =============  ============  ========  =====  =============
transform_name    hasher_name       threshold    recall    fpr    n_exemplars
================  =============  ============  ========  =====  =============
blur2             ahash            0.0117188     62.247      0           3854
blur2             blockmean        0.0134298     82.045      0           3854
blur2             dhash            0.0898438     98.054      0           3854
blur2             marrhildreth     0.102431      98.651      0           3854
blur2             pdq              0.304688      99.974      0           3854
blur2             phash            0.179688     100          0           3854
blur2             shrinkhash      61.441         28.23       0           3854
blur2             wavelet          0.015625      59.964      0           3854
crop0.05          ahash            0.0078125      0.208      0           3854
crop0.05          blockmean        0.0144628      0.337      0           3854
crop0.05          dhash            0.101562       0.597      0           3854
crop0.05          marrhildreth     0.175347       1.635      0           3854
crop0.05          pdq              0.265625      11.598      0           3854
crop0.05          phash            0.234375       9.185      0           3854
crop0.05          shrinkhash      95.5667         1.427      0           3854
crop0.05          wavelet          0.015625       0.259      0           3854
gamma2            ahash            0.0078125      2.647      0           3854
gamma2            blockmean        0.00826446     2.335      0           3854
gamma2            dhash            0.105469      91.048      0           3854
gamma2            marrhildreth     0.121528      95.381      0           3854
gamma2            pdq              0.195312     100          0           3854
gamma2            phash            0.234375     100          0           3854
gamma2            shrinkhash     112.911          0.182      0           3854
gamma2            wavelet          0.015625      15.153      0           3854
jpeg95            ahash            0.0117188     31.474      0           3854
jpeg95            blockmean        0.0134298     39.673      0           3854
jpeg95            dhash            0.117188      64.037      0           3854
jpeg95            marrhildreth     0.109375      66.762      0           3854
jpeg95            pdq              0.273438      99.87       0           3854
jpeg95            phash            0.335938      99.948      0           3854
jpeg95            shrinkhash      66.7008        33.083      0           3854
jpeg95            wavelet          0.015625      21.069      0           3854
noise0.2          ahash            0.0078125      7.421      0           3854
noise0.2          blockmean        0.0154959     23.638      0           3854
noise0.2          dhash            0.167969      63.83       0           3854
noise0.2          marrhildreth     0.119792      46.341      0           3854
noise0.2          pdq              0.28125       99.559      0           3854
noise0.2          phash            0.273438      99.87       0           3854
noise0.2          shrinkhash     154.729          0.934      0           3854
noise0.2          wavelet          0.0078125      1.635      0           3854
noop              ahash            0            100          0           3854
noop              blockmean        0            100          0           3854
noop              dhash            0            100          0           3854
noop              marrhildreth     0            100          0           3854
noop              pdq              0            100          0           3854
noop              phash            0            100          0           3854
noop              shrinkhash       0            100          0           3854
noop              wavelet          0            100          0           3854
pad0.2            ahash            0.046875       0.052      0           3854
pad0.2            blockmean        0.0588843      0.026      0           3854
pad0.2            dhash            0.109375       0.285      0           3854
pad0.2            marrhildreth     0.190972       0.104      0           3854
pad0.2            pdq              0.289062       1.738      0           3854
pad0.2            phash            0.28125        3.269      0           3854
pad0.2            shrinkhash     136.11           0.078      0           3854
pad0.2            wavelet          0.0820312      0          0           3854
rotate4           ahash            0.03125        1.946      0           3854
rotate4           blockmean        0.0382231      3.503      0           3854
rotate4           dhash            0.0976562      1.583      0           3854
rotate4           marrhildreth     0.159722       6.046      0           3854
rotate4           pdq              0.28125       60.042      0           3854
rotate4           phash            0.265625      65.646      0           3854
rotate4           shrinkhash      69.4622         1.92       0           3854
rotate4           wavelet          0.015625       0.078      0           3854
vignette          ahash            0.03125        5.475      0           3854
vignette          blockmean        0.0330579      6.461      0           3854
vignette          dhash            0.0742188     14.011      0           3854
vignette          marrhildreth     0.0954861     30.436      0           3854
vignette          pdq              0.132812     100          0           3854
vignette          phash            0.132812     100          0           3854
vignette          shrinkhash     103.015          4.515      0           3854
vignette          wavelet          0.0234375      2.024      0           3854
watermark         ahash            0.0078125     28.464      0           3854
watermark         blockmean        0.0134298     43.15       0           3854
watermark         dhash            0.078125     100          0           3854
watermark         marrhildreth     0.114583      99.248      0           3854
watermark         pdq              0.28125       99.325      0           3854
watermark         phash            0.289062      99.481      0           3854
watermark         shrinkhash     104.666         70.239      0           3854
watermark         wavelet          0.015625      46.653      0           3854
================  =============  ============  ========  =====  =============

=============  ===========  ========  ===========  =============
hasher_name      threshold    recall          fpr    n_exemplars
=============  ===========  ========  ===========  =============
ahash           0.0078125     20.005  0                    38540
blockmean       0.00826446    22.003  0                    38540
dhash           0.0898438     46.798  6.07681e-05          38540
marrhildreth    0.102431      52.377  9.97855e-05          38540
pdq             0.265625      75.846  6.93433e-05          38540
phash           0.273438      80.106  6.56685e-05          38540
shrinkhash     60.1166        19.538  0                    38540
wavelet         0.0078125     16.168  0                    38540
=============  ===========  ========  ===========  =============

Video Hashing
=============

The below example does the following:

- Download a benchmarking dataset. Here we use the `Charades <https://prior.allenai.org/projects/charades>`_ dataset which contain over 9,000 videos.
- Load the dataset.
- Transform the dataset to generate synthetically altered videos. Our hashers are responsible for
  matching the altered videos with the originals.
- Define some hashers we want to evaluate.
- Compute all the hashes.
- Report metrics for each video category / hasher / transformation combination to see how well our hashers
  can match the altered videos to the original ("no-op" videos).

.. code-block:: python

    import os
    import zipfile
    import urllib.request


    import pandas as pd

    import perception.benchmarking
    import perception.hashers

    if not os.path.isdir('Charades_v1_480'):
        # Download the dataset since it appears we do not have it. Note that
        # these are large files (> 13GB).
        urllib.request.urlretrieve(
            url='http://ai2-website.s3.amazonaws.com/data/Charades_v1_480.zip',
            filename='Charades_v1_480.zip'
        )
        with zipfile.ZipFile('Charades_v1_480.zip') as zfile:
            zfile.extractall('.')
        urllib.request.urlretrieve(
            url='http://ai2-website.s3.amazonaws.com/data/Charades.zip',
            filename='Charades.zip'
        )
        with zipfile.ZipFile('Charades.zip') as zfile:
            zfile.extractall('.')


    # These are files that we've identified as having identical subsequences, typically
    # when a person is out of frame and the backgrounds are the same.
    duplicates = [
        ('0HVVN.mp4', 'UZRQD.mp4'), ('ZIOET.mp4', 'YGXX6.mp4'), ('82XPD.mp4', 'E7QDZ.mp4'),
        ('FQDS1.mp4', 'AIOTI.mp4'), ('PBV4T.mp4', 'XXYWL.mp4'), ('M0P0H.mp4', 'STY6W.mp4'),
        ('3Q92U.mp4', 'GHPO3.mp4'), ('NFIQM.mp4', 'I2DHG.mp4'), ('PIRMO.mp4', '0GFE8.mp4'),
        ('LRPBA.mp4', '9VK0J.mp4'), ('UI0QG.mp4', 'FHXKQ.mp4'), ('Y05U8.mp4', '4RVZB.mp4'),
        ('J6TVB.mp4', '2ZBL5.mp4'), ('A8T8V.mp4', 'IGOQK.mp4'), ('H8QM1.mp4', 'QYMWC.mp4'),
        ('O45BC.mp4', 'ZS7X6.mp4'), ('NOP6W.mp4', 'F7KFE.mp4'), ('4MPPQ.mp4', 'A3M94.mp4'),
        ('L8FFR.mp4', 'M8MP0.mp4'), ('EHYXP.mp4', 'O8PO3.mp4'), ('MGBLJ.mp4', 'RIEG6.mp4'),
        ('53FPM.mp4', 'BLFEV.mp4'), ('UIIF3.mp4', 'TKEKQ.mp4'), ('GVX7E.mp4', '7GPSY.mp4'),
        ('T7HZB.mp4', '6KGZA.mp4'), ('65M4K.mp4', 'UDGP2.mp4'), ('6SS4H.mp4', 'CK6OL.mp4'),
        ('OVHFT.mp4', 'GG1X2.mp4'), ('VEHER.mp4', 'XBPEJ.mp4'), ('WN38A.mp4', '2QI8F.mp4'),
        ('UMXKN.mp4', 'EOKJ0.mp4'), ('OSIKP.mp4', 'WT2C0.mp4'), ('H5V2Y.mp4', 'ZXN6A.mp4'),
        ('XS6PF.mp4', '1WJ6O.mp4'), ('S2XJW.mp4', 'YH0BX.mp4'), ('UO607.mp4', 'Z5JZD.mp4'),
        ('XN64E.mp4', 'CSRZM.mp4'), ('YXI7M.mp4', 'IKQLJ.mp4'), ('1B9C8.mp4', '004QE.mp4'),
        ('V1SQH.mp4', '48WOM.mp4'), ('107YZ.mp4', 'I049A.mp4'), ('3S6WL.mp4', 'SC5YW.mp4'),
        ('OY50Q.mp4', '5T607.mp4'), ('XKH7W.mp4', '028CE.mp4'), ('X8XQE.mp4', 'J0VXY.mp4'),
        ('STB0G.mp4', 'J0VXY.mp4'), ('UNXLF.mp4', 'J0VXY.mp4'), ('56PK0.mp4', 'M1TZR.mp4'),
        ('FVITB.mp4', 'R0M34.mp4'), ('BPZE3.mp4', 'R0M34.mp4'), ('VS7DA.mp4', '1X0M3.mp4'),
        ('I7MEA.mp4', 'YMM1Z.mp4'), ('9N76L.mp4', '0LDP7.mp4'), ('AXS82.mp4', 'W8WRK.mp4'),
        ('8TSU4.mp4', 'MXATD.mp4'), ('80FWF.mp4', '18HFG.mp4'), ('RO3A2.mp4', 'V4HY4.mp4'),
        ('HU409.mp4', 'BDWIX.mp4'), ('3YY88.mp4', 'EHHRS.mp4'), ('65RS3.mp4', 'SLIH4.mp4'),
        ('LR0L8.mp4', 'Y665P.mp4')
    ]

    blacklist = [fp1 for fp1, fp2 in duplicates]
    df = pd.concat([pd.read_csv('Charades/Charades_v1_test.csv'), pd.read_csv('Charades/Charades_v1_train.csv')])
    df = df[~(df['id'] + '.mp4').isin(blacklist)]
    df['filepath'] = df['id'].apply(lambda video_id: os.path.join('Charades_v1_480', video_id + '.mp4'))
    assert df['filepath'].apply(os.path.isfile).all(), 'Some video files are missing.'
    dataset = perception.benchmarking.BenchmarkVideoDataset.from_tuples(files=df[['filepath', 'scene']].itertuples(index=False))

    if not os.path.isdir('benchmarking_videos'):
        # We haven't computed the transforms yet, so we do that
        # now. Below, we create the following files for each of
        # the videos in our dataset. Note that the only required
        # transform is `noop` (see documentation for
        # perception.bencharmking.BenchmarkVideoDataset.transform).
        #
        # noop: This is the base video we'll actually use in benchmarking, rather
        #       than using the raw video. It is the same as the raw video but downsampled
        #       to a size that is reasonable for hashing (240p). This is because all
        #       of our hashers downsample to a size smaller than this anyway, so there
        #       is no benefit to a higher resolution. Also, we limit the length to the
        #       first five minutes of the video, which speeds everything up significantly.
        # shrink: Shrink the noop video down to 70% of its original size.
        # clip0.2: Clip the first 20% and last 20% of the noop video off.
        # slideshow: Create a slideshow version of the video that grabs frames periodically
        #            from the original.
        # black_frames: Add black frames before and after the start of the video.
        # gif: Create a GIF from the video (similar to slideshow but with re-encoding)
        # black_padding: Add black bars to the top and bottom of the video.
        pad_width = 240
        pad_height = 320
        transforms = {
            'noop': perception.benchmarking.video_transforms.get_simple_transform(
                width='ceil(min(240/max(iw, ih), 1)*iw/2)*2',
                height='ceil(min(240/max(iw, ih), 1)*ih/2)*2',
                codec='h264',
                output_ext='.m4v',
                sar='1/1',
                clip_s=(None, 60*5)
            ),
            'shrink': perception.benchmarking.video_transforms.get_simple_transform(
                width='ceil(0.7*iw/2)*2',
                height='ceil(0.7*ih/2)*2'
            ),
            'clip0.2': perception.benchmarking.video_transforms.get_simple_transform(clip_pct=(0.2, 0.8)),
            'slideshow': perception.benchmarking.video_transforms.get_slideshow_transform(
                frame_input_rate=1/2.5, frame_output_rate=0.5, max_frames=10, offset=1.3),
            'black_frames': perception.benchmarking.video_transforms.get_black_frame_padding_transform(0.5, 0.05),
            'gif': perception.benchmarking.video_transforms.get_simple_transform(
                output_ext='.gif', codec='gif', clip_s=(1.2, 10.2), fps=1/2.5
            ),
            'black_padding': perception.benchmarking.video_transforms.get_simple_transform(
                width=f'(iw*sar)*min({pad_width}/(iw*sar),{pad_height}/ih)', height=f'ih*min({pad_width}/(iw*sar),{pad_height}/ih)',
                pad=f'{pad_width}:{pad_height}:({pad_width}-iw*min({pad_width}/iw,{pad_height}/ih))/2:({pad_height}-ih*min({pad_width}/iw,{pad_height}/ih))/2'
            )
        }

        # Save the transforms for later.
        transformed = dataset.transform(transforms=transforms, storage_dir='benchmarking_videos')

    transformed = perception.benchmarking.BenchmarkVideoTransforms.load('benchmarking_videos', verify_md5=False)

    phashu8 = perception.hashers.PHashU8(exclude_first_term=False, freq_shift=1, hash_size=12)
    hashers = {
        'phashu8_framewise': perception.hashers.FramewiseHasher(
            frames_per_second=1, frame_hasher=phashu8, interframe_threshold=50, quality_threshold=90),
        'phashu8_tmkl1': perception.hashers.SimpleSceneDetection(
            base_hasher=perception.hashers.TMKL1(
                frames_per_second=5, frame_hasher=phashu8,
                distance_metric='euclidean', dtype='uint8',
                norm=None, quality_threshold=90),
            max_scene_length=1,
            interscene_threshold=50
        )
    }
    if not os.path.isfile('hashes.csv'):
        # We haven't computed the hashes, so we do that now.
        hashes = transformed.compute_hashes(hashers=hashers, max_workers=0)
        # Save the hashes for later. It took a long time after all!
        hashes.save('hashes.csv')

    hashes = perception.benchmarking.BenchmarkHashes.load('hashes.csv')

    hashes.compute_threshold_recall(fpr_threshold=0.001, grouping=['transform_name'])


================  =================  ===========  ========  ===========  =============
transform_name    hasher_name          threshold    recall          fpr    n_exemplars
================  =================  ===========  ========  ===========  =============
black_frames      phashu8_framewise      51.0979    88.163  0.000933489         277865
black_frames      phashu8_tmkl1          55.7584    99.918  0.000821862         403415
black_padding     phashu8_framewise      74.6391     7.689  0                   276585
black_padding     phashu8_tmkl1          53.8702    99.887  0.000924784         411664
clip0.2           phashu8_framewise      54.8635    90.772  0.000904977         223591
clip0.2           phashu8_tmkl1          59.1693    99.753  0.000926021         323870
gif               phashu8_framewise      55.4437    68.314  0.000913103          82038
gif               phashu8_tmkl1          63.773     82.926  0.000993172          32140
noop              phashu8_framewise       0        100      0                   281976
noop              phashu8_tmkl1           0        100      0                   408673
shrink            phashu8_framewise      24.7184   100      0                   280617
shrink            phashu8_tmkl1          52.8678    99.866  0.000926307         399357
slideshow         phashu8_framewise      56.9825    99.712  0.000926689         164361
slideshow         phashu8_tmkl1          63.4271    95.131  0.000988576          71668
================  =================  ===========  ========  ===========  =============