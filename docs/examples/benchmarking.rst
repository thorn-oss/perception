Benchmarking
************

This package provides a fair amount of infrastructure for benchmarking different hashers to evaluate their performance.
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
    dataset = benchmarking.BenchmarkDataset.from_tuples(files=[
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
    transformed = benchmarking.BenchmarkTransforms.load(
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
