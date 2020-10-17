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
    from perception.hashers.image.pdq import PDQHash

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
    class ShrinkHash(hashers.ImageHasher):
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
        'pdq': PDQHash(),
        'phash': hashers.PHash(hash_size=16),
        'marrhildreth': hashers.MarrHildreth(),
        'wavelet': hashers.WaveletHash(hash_size=16),
        'blockmean': hashers.BlockMean(),
        'shrinkhash': ShrinkHash()
    }

    # Compute the hashes
    hashes = transformed.compute_hashes(hashers=hashers_dict)

    # Get performance metrics (i.e., recall) for each hash function based on
    # a minimum precision threshold. Here we use 99.99%.
    precision_threshold = 99.99
    
    # The metrics are just pandas dataframes. We use tabulate here to obtain the tables
    # formatted for the documentation.
    metrics = hashes.compute_threshold_recall(precision_threshold=precision_threshold).reset_index()
    print(tabulate.tabulate(metrics, showindex=False, headers=metrics.columns, tablefmt='rst'))

    metrics_by_transform = hashes.compute_threshold_recall(grouping=['transform_name'], precision_threshold=precision_threshold).reset_index()
    print(tabulate.tabulate(metrics_by_transform, showindex=False, headers=metrics_by_transform.columns, tablefmt='rst'))

    metrics_simple = hashes.compute_threshold_recall(grouping=[], precision_threshold=precision_threshold).reset_index()
    print(tabulate.tabulate(metrics_simple, showindex=False, headers=metrics_simple.columns, tablefmt='rst'))



===========  ================  =============  ============  ========  ===========  =============
category     transform_name    hasher_name       threshold    recall    precision    n_exemplars
===========  ================  =============  ============  ========  ===========  =============
paintings    blur2             ahash            0.0078125     51.724          100           2204
paintings    blur2             blockmean        0.0123967     85.753          100           2204
paintings    blur2             dhash            0.105469     100              100           2204
paintings    blur2             marrhildreth     0.0989583    100              100           2204
paintings    blur2             pdq              0.117188     100              100           2204
paintings    blur2             phash            0.0390625    100              100           2204
paintings    blur2             shrinkhash      60.8112        43.33           100           2204
paintings    blur2             wavelet          0.0117188     66.379          100           2204
paintings    crop0.05          ahash            0.00390625     0.045          100           2204
paintings    crop0.05          blockmean        0.0123967      0.227          100           2204
paintings    crop0.05          dhash            0.210938       7.577          100           2204
paintings    crop0.05          marrhildreth     0.213542       3.584          100           2204
paintings    crop0.05          pdq              0.257812       8.439          100           2204
paintings    crop0.05          phash            0.226562       6.76           100           2204
paintings    crop0.05          shrinkhash      95.0053         2.269          100           2204
paintings    crop0.05          wavelet          0.0078125      0              nan           2204
paintings    gamma2            ahash            0.00390625     0.998          100           2204
paintings    gamma2            blockmean        0.0072314      1.724          100           2204
paintings    gamma2            dhash            0.167969      98.639          100           2204
paintings    gamma2            marrhildreth     0.159722      99.41           100           2204
paintings    gamma2            pdq              0.164062     100              100           2204
paintings    gamma2            phash            0.164062     100              100           2204
paintings    gamma2            shrinkhash      46.5296         0              nan           2204
paintings    gamma2            wavelet          0.0117188     18.512          100           2204
paintings    jpeg95            ahash            0.00390625     4.22           100           2204
paintings    jpeg95            blockmean        0.0134298     28.811          100           2204
paintings    jpeg95            dhash            0.191406      94.782          100           2204
paintings    jpeg95            marrhildreth     0.168403      82.985          100           2204
paintings    jpeg95            pdq              0.257812     100              100           2204
paintings    jpeg95            phash            0.234375     100              100           2204
paintings    jpeg95            shrinkhash      66.053         55.172          100           2204
paintings    jpeg95            wavelet          0              0              nan           2204
paintings    noise0.2          ahash            0.00390625     2.677          100           2204
paintings    noise0.2          blockmean        0.00826446     6.987          100           2204
paintings    noise0.2          dhash            0.25          93.648          100           2204
paintings    noise0.2          marrhildreth     0.170139      73.911          100           2204
paintings    noise0.2          pdq              0.257812      99.229          100           2204
paintings    noise0.2          phash            0.257812     100              100           2204
paintings    noise0.2          shrinkhash     169.387          3.312          100           2204
paintings    noise0.2          wavelet          0.0078125      1.407          100           2204
paintings    noop              ahash            0            100              100           2204
paintings    noop              blockmean        0            100              100           2204
paintings    noop              dhash            0            100              100           2204
paintings    noop              marrhildreth     0            100              100           2204
paintings    noop              pdq              0            100              100           2204
paintings    noop              phash            0            100              100           2204
paintings    noop              shrinkhash       0            100              100           2204
paintings    noop              wavelet          0            100              100           2204
paintings    pad0.2            ahash            0.0703125      0              nan           2204
paintings    pad0.2            blockmean        0.0795455      0              nan           2204
paintings    pad0.2            dhash            0.210938       1.089          100           2204
paintings    pad0.2            marrhildreth     0.177083       0              nan           2204
paintings    pad0.2            pdq              0.289062       1.86           100           2204
paintings    pad0.2            phash            0.273438       2.541          100           2204
paintings    pad0.2            shrinkhash     146.325          0.181          100           2204
paintings    pad0.2            wavelet          0.109375       0              nan           2204
paintings    resize0.5         ahash            0.0078125     76.089          100           2204
paintings    resize0.5         blockmean        0.0144628     98.185          100           2204
paintings    resize0.5         dhash            0.0976562    100              100           2204
paintings    resize0.5         marrhildreth     0.154514      99.819          100           2204
paintings    resize0.5         pdq              0.1875       100              100           2204
paintings    resize0.5         phash            0.09375      100              100           2204
paintings    resize0.5         shrinkhash      56.9034        76.27           100           2204
paintings    resize0.5         wavelet          0.0117188     84.71           100           2204
paintings    rotate4           ahash            0.0390625      2.949          100           2204
paintings    rotate4           blockmean        0.0382231      2.949          100           2204
paintings    rotate4           dhash            0.207031      36.298          100           2204
paintings    rotate4           marrhildreth     0.227431      61.978          100           2204
paintings    rotate4           pdq              0.273438      56.08           100           2204
paintings    rotate4           phash            0.257812      61.615          100           2204
paintings    rotate4           shrinkhash      69.1737         2.813          100           2204
paintings    rotate4           wavelet          0.03125        0.136          100           2204
paintings    vignette          ahash            0.0429688      6.171          100           2204
paintings    vignette          blockmean        0.0475207      8.122          100           2204
paintings    vignette          dhash            0.121094      32.305          100           2204
paintings    vignette          marrhildreth     0.177083      77.904          100           2204
paintings    vignette          pdq              0.132812     100              100           2204
paintings    vignette          phash            0.132812     100              100           2204
paintings    vignette          shrinkhash     102.186          3.267          100           2204
paintings    vignette          wavelet          0.046875       3.085          100           2204
paintings    watermark         ahash            0.00390625    20.054          100           2204
paintings    watermark         blockmean        0.0123967     45.145          100           2204
paintings    watermark         dhash            0.0585938    100              100           2204
paintings    watermark         marrhildreth     0.0625       100              100           2204
paintings    watermark         pdq              0.273438      98.866          100           2204
paintings    watermark         phash            0.28125       99.456          100           2204
paintings    watermark         shrinkhash     104.398         75.998          100           2204
paintings    watermark         wavelet          0.0117188     51.27           100           2204
photographs  blur2             ahash            0.015625      76.727          100           1650
photographs  blur2             blockmean        0.0330579     98              100           1650
photographs  blur2             dhash            0.0859375     98.97           100           1650
photographs  blur2             marrhildreth     0.107639      97.576          100           1650
photographs  blur2             pdq              0.304688     100              100           1650
photographs  blur2             phash            0.179688     100              100           1650
photographs  blur2             shrinkhash     117.627         44              100           1650
photographs  blur2             wavelet          0.0195312     79.879          100           1650
photographs  crop0.05          ahash            0.0078125      0.182          100           1650
photographs  crop0.05          blockmean        0.0258264      0.788          100           1650
photographs  crop0.05          dhash            0.0976562      1.091          100           1650
photographs  crop0.05          marrhildreth     0.173611       3.152          100           1650
photographs  crop0.05          pdq              0.304688      30.606          100           1650
photographs  crop0.05          phash            0.320312      63.697          100           1650
photographs  crop0.05          shrinkhash     125.94           1.152          100           1650
photographs  crop0.05          wavelet          0.015625       0.182          100           1650
photographs  gamma2            ahash            0.015625       8.182          100           1650
photographs  gamma2            blockmean        0.0268595     17.212          100           1650
photographs  gamma2            dhash            0.101562      90.303          100           1650
photographs  gamma2            marrhildreth     0.105903      90.909          100           1650
photographs  gamma2            pdq              0.210938     100              100           1650
photographs  gamma2            phash            0.234375     100              100           1650
photographs  gamma2            shrinkhash     119.683          0.545          100           1650
photographs  gamma2            wavelet          0.0195312     18.424          100           1650
photographs  jpeg95            ahash            0.0117188     29.879          100           1650
photographs  jpeg95            blockmean        0.0278926     76.788          100           1650
photographs  jpeg95            dhash            0.121094      84.182          100           1650
photographs  jpeg95            marrhildreth     0.104167      69.576          100           1650
photographs  jpeg95            pdq              0.296875      99.879          100           1650
photographs  jpeg95            phash            0.28125       99.879          100           1650
photographs  jpeg95            shrinkhash     131.031         89.212          100           1650
photographs  jpeg95            wavelet          0.0195312     40.242          100           1650
photographs  noise0.2          ahash            0.015625      27.636          100           1650
photographs  noise0.2          blockmean        0.036157      75.091          100           1650
photographs  noise0.2          dhash            0.121094      54.121          100           1650
photographs  noise0.2          marrhildreth     0.0989583     46.364          100           1650
photographs  noise0.2          pdq              0.296875      99.697          100           1650
photographs  noise0.2          phash            0.304688      99.818          100           1650
photographs  noise0.2          shrinkhash     210.661         57.576          100           1650
photographs  noise0.2          wavelet          0.0234375     27.03           100           1650
photographs  noop              ahash            0            100              100           1650
photographs  noop              blockmean        0            100              100           1650
photographs  noop              dhash            0            100              100           1650
photographs  noop              marrhildreth     0            100              100           1650
photographs  noop              pdq              0            100              100           1650
photographs  noop              phash            0            100              100           1650
photographs  noop              shrinkhash       0            100              100           1650
photographs  noop              wavelet          0            100              100           1650
photographs  pad0.2            ahash            0.0429688      0.061          100           1650
photographs  pad0.2            blockmean        0.0320248      0              nan           1650
photographs  pad0.2            dhash            0.105469       0.545          100           1650
photographs  pad0.2            marrhildreth     0.177083       0.121          100           1650
photographs  pad0.2            pdq              0.28125        1.455          100           1650
photographs  pad0.2            phash            0.289062       3.515          100           1650
photographs  pad0.2            shrinkhash     114.721          0.061          100           1650
photographs  pad0.2            wavelet          0.0820312      0              nan           1650
photographs  resize0.5         ahash            0.015625      87.697          100           1650
photographs  resize0.5         blockmean        0.0330579     99.152          100           1650
photographs  resize0.5         dhash            0.0898438     98.485          100           1650
photographs  resize0.5         marrhildreth     0.111111      95.394          100           1650
photographs  resize0.5         pdq              0.328125      99.818          100           1650
photographs  resize0.5         phash            0.234375     100              100           1650
photographs  resize0.5         shrinkhash     132.117         80.242          100           1650
photographs  resize0.5         wavelet          0.0195312     88.97           100           1650
photographs  rotate4           ahash            0.0273438      1.818          100           1650
photographs  rotate4           blockmean        0.0371901      3.879          100           1650
photographs  rotate4           dhash            0.09375        2.97           100           1650
photographs  rotate4           marrhildreth     0.149306       4.606          100           1650
photographs  rotate4           pdq              0.304688      73.394          100           1650
photographs  rotate4           phash            0.3125        89.818          100           1650
photographs  rotate4           shrinkhash     130.211          4.424          100           1650
photographs  rotate4           wavelet          0.0078125      0.061          100           1650
photographs  vignette          ahash            0.0273438      8.242          100           1650
photographs  vignette          blockmean        0.0320248     10              100           1650
photographs  vignette          dhash            0.0703125     22              100           1650
photographs  vignette          marrhildreth     0.0954861     38.727          100           1650
photographs  vignette          pdq              0.117188     100              100           1650
photographs  vignette          phash            0.125        100              100           1650
photographs  vignette          shrinkhash     138.989         11.939          100           1650
photographs  vignette          wavelet          0.0195312      4.242          100           1650
photographs  watermark         ahash            0.015625      42.667          100           1650
photographs  watermark         blockmean        0.0247934     60.788          100           1650
photographs  watermark         dhash            0.078125     100              100           1650
photographs  watermark         marrhildreth     0.112847      98.727          100           1650
photographs  watermark         pdq              0.3125        99.818          100           1650
photographs  watermark         phash            0.3125        99.758          100           1650
photographs  watermark         shrinkhash     142.046         79.576          100           1650
photographs  watermark         wavelet          0.0195312     53.455          100           1650
===========  ================  =============  ============  ========  ===========  =============

================  =============  ============  ========  ===========  =============
transform_name    hasher_name       threshold    recall    precision    n_exemplars
================  =============  ============  ========  ===========  =============
blur2             ahash            0.0078125     49.014          100           3854
blur2             blockmean        0.0123967     80.773          100           3854
blur2             dhash            0.0859375     99.196          100           3854
blur2             marrhildreth     0.107639      98.962          100           3854
blur2             pdq              0.234375      99.948          100           3854
blur2             phash            0.179688     100              100           3854
blur2             shrinkhash      60.8112        28.412          100           3854
blur2             wavelet          0.0117188     62.247          100           3854
crop0.05          ahash            0.00390625     0.052          100           3854
crop0.05          blockmean        0.0123967      0.208          100           3854
crop0.05          dhash            0.0976562      0.493          100           3854
crop0.05          marrhildreth     0.173611       1.635          100           3854
crop0.05          pdq              0.257812       9.03           100           3854
crop0.05          phash            0.226562       7.058          100           3854
crop0.05          shrinkhash      95.0053         1.427          100           3854
crop0.05          wavelet          0.0078125      0              nan           3854
gamma2            ahash            0.00390625     0.934          100           3854
gamma2            blockmean        0.0072314      1.713          100           3854
gamma2            dhash            0.101562      90.036          100           3854
gamma2            marrhildreth     0.105903      94.24           100           3854
gamma2            pdq              0.210938     100              100           3854
gamma2            phash            0.234375     100              100           3854
gamma2            shrinkhash     108.457          0.156          100           3854
gamma2            wavelet          0.0117188     14.997          100           3854
jpeg95            ahash            0.00390625     5.319          100           3854
jpeg95            blockmean        0.0134298     32.045          100           3854
jpeg95            dhash            0.121094      74.079          100           3854
jpeg95            marrhildreth     0.104167      59.263          100           3854
jpeg95            pdq              0.257812      99.896          100           3854
jpeg95            phash            0.234375      99.896          100           3854
jpeg95            shrinkhash      66.053         40.296          100           3854
jpeg95            wavelet          0.00390625     3.71           100           3854
noise0.2          ahash            0.00390625     2.984          100           3854
noise0.2          blockmean        0.00826446     8.563          100           3854
noise0.2          dhash            0.121094      40.088          100           3854
noise0.2          marrhildreth     0.0989583     33.083          100           3854
noise0.2          pdq              0.257812      99.222          100           3854
noise0.2          phash            0.273438      99.896          100           3854
noise0.2          shrinkhash     169.387          4.385          100           3854
noise0.2          wavelet          0.0078125      1.894          100           3854
noop              ahash            0            100              100           3854
noop              blockmean        0            100              100           3854
noop              dhash            0            100              100           3854
noop              marrhildreth     0            100              100           3854
noop              pdq              0            100              100           3854
noop              phash            0            100              100           3854
noop              shrinkhash       0            100              100           3854
noop              wavelet          0            100              100           3854
pad0.2            ahash            0.0429688      0.026          100           3854
pad0.2            blockmean        0.0320248      0              nan           3854
pad0.2            dhash            0.105469       0.234          100           3854
pad0.2            marrhildreth     0.177083       0.052          100           3854
pad0.2            pdq              0.28125        1.349          100           3854
pad0.2            phash            0.273438       2.387          100           3854
pad0.2            shrinkhash     114.721          0.052          100           3854
pad0.2            wavelet          0.0820312      0              nan           3854
resize0.5         ahash            0.0078125     70.784          100           3854
resize0.5         blockmean        0.0144628     95.226          100           3854
resize0.5         dhash            0.0898438     99.299          100           3854
resize0.5         marrhildreth     0.112847      97.846          100           3854
resize0.5         pdq              0.265625      99.844          100           3854
resize0.5         phash            0.234375     100              100           3854
resize0.5         shrinkhash      56.9034        51.453          100           3854
resize0.5         wavelet          0.0117188     80.747          100           3854
rotate4           ahash            0.0273438      1.297          100           3854
rotate4           blockmean        0.0371901      3.036          100           3854
rotate4           dhash            0.09375        1.401          100           3854
rotate4           marrhildreth     0.149306       3.762          100           3854
rotate4           pdq              0.273438      54.489          100           3854
rotate4           phash            0.257812      59.626          100           3854
rotate4           shrinkhash      69.1737         1.894          100           3854
rotate4           wavelet          0.0078125      0.026          100           3854
vignette          ahash            0.0273438      4.67           100           3854
vignette          blockmean        0.0320248      6.098          100           3854
vignette          dhash            0.0703125     12.195          100           3854
vignette          marrhildreth     0.0954861     30.54           100           3854
vignette          pdq              0.132812     100              100           3854
vignette          phash            0.132812     100              100           3854
vignette          shrinkhash     103.005          4.541          100           3854
vignette          wavelet          0.0195312      1.946          100           3854
watermark         ahash            0.00390625    18.5            100           3854
watermark         blockmean        0.0123967     41.593          100           3854
watermark         dhash            0.078125     100              100           3854
watermark         marrhildreth     0.112847      99.455          100           3854
watermark         pdq              0.273438      99.014          100           3854
watermark         phash            0.28125       99.377          100           3854
watermark         shrinkhash     104.398         71.199          100           3854
watermark         wavelet          0.0117188     46.912          100           3854
================  =============  ============  ========  ===========  =============

=============  ===========  ========  ===========  =============
hasher_name      threshold    recall    precision    n_exemplars
=============  ===========  ========  ===========  =============
ahash           0.00390625    17.578     100               42394
blockmean       0.00826446    27.714     100               42394
dhash           0.0859375     51.981      99.9952          42394
marrhildreth    0.100694      55.942      99.9957          42394
pdq             0.257812      77.181      99.9969          42394
phash           0.273438      81.967      99.9942          42394
shrinkhash     56.9034        22.378     100               42394
wavelet         0.00390625    18.467     100               42394
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
        ('LR0L8.mp4', 'Y665P.mp4'), ('DVPL2.mp4', 'EI5M3.mp4'), ('0EGNU.mp4', 'CU3JE.mp4'),
        ('94KP4.mp4', '94KP4.mp4'), ('79QDP.mp4', '79QDP.mp4'), ('GKBX9.mp4', 'GKBX9.mp4'),
        ('RX6R8.mp4', 'RX6R8.mp4'), ('PMVT7.mp4', 'PMVT7.mp4'), ('XNXW6.mp4', 'XNXW6.mp4'),
        ('I005F.mp4', 'I005F.mp4'), ('TF95Y.mp4', 'TF95Y.mp4'), ('79QDP.mp4', '79QDP.mp4'),
        ('LQGMM.mp4', 'LQGMM.mp4'), ('QCAUL.mp4', 'QCAUL.mp4'), ('GFVSV.mp4', 'GFVSV.mp4'),
        ('4UYGY.mp4', '4UYGY.mp4'), ('BYDSE.mp4', 'BYDSE.mp4'), ('PV3KQ.mp4', 'PV3KQ.mp4'),
        ('1X0M3.mp4', '1X0M3.mp4'), ('T5FHD.mp4', 'T5FHD.mp4'), ('QRHJJ.mp4', 'QRHJJ.mp4'),
        ('JYBGS.mp4', 'JYBGS.mp4'), ('N2XCF.mp4', 'N2XCF.mp4'), ('OZPA9.mp4', 'OZPA9.mp4'),
        ('297S4.mp4', '297S4.mp4'), ('LHU7D.mp4', 'LHU7D.mp4'), ('TSKZL.mp4', 'TSKZL.mp4'),
        ('BCONW.mp4', 'BCONW.mp4'), ('KBPDM.mp4', 'KBPDM.mp4'), ('7FTBS.mp4', '7FTBS.mp4'),
        ('099Y1.mp4', '099Y1.mp4'), ('S2RIQ.mp4', 'S2RIQ.mp4'), ('22FJU.mp4', '22FJU.mp4'),
        ('99UA6.mp4', '99UA6.mp4'), ('WJ13E.mp4', 'WJ13E.mp4'), ('5OLVC.mp4', '5OLVC.mp4'),
        ('YQ6Z6.mp4', 'YQ6Z6.mp4'), ('T5MLJ.mp4', 'T5MLJ.mp4'), ('0VOQC.mp4', '0VOQC.mp4'),
        ('S2RIQ.mp4', 'S2RIQ.mp4'), ('2VNXF.mp4', '2VNXF.mp4'), ('G87XG.mp4', 'G87XG.mp4'),
        ('RRS54.mp4', 'RRS54.mp4'), ('TXJK7.mp4', 'TXJK7.mp4'), ('G4KE3.mp4', 'G4KE3.mp4'),
        ('3SNSC.mp4', '3SNSC.mp4'), ('U2FA5.mp4', 'U2FA5.mp4'), ('9AFQ7.mp4', '9AFQ7.mp4')
    ]

    blacklist = [fp1 for fp1, fp2 in duplicates]
    df = pd.concat([pd.read_csv('Charades/Charades_v1_test.csv'), pd.read_csv('Charades/Charades_v1_train.csv')])
    df = df[~(df['id'] + '.mp4').isin(blacklist)]
    df['filepath'] = df['id'].apply(lambda video_id: os.path.join('Charades_v1_480', video_id + '.mp4'))
    assert df['filepath'].apply(os.path.isfile).all(), 'Some video files are missing.'
    dataset = perception.benchmarking.BenchmarkVideoDataset.from_tuples(
        files=df[['filepath', 'scene']].itertuples(index=False)
    )

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
        hashes = transformed.compute_hashes(hashers=hashers, max_workers=5)
        # Save the hashes for later. It took a long time after all!
        hashes.save('hashes.csv')

    hashes = perception.benchmarking.BenchmarkHashes.load('hashes.csv')

    hashes.compute_threshold_recall(precision_threshold=99.9, grouping=['transform_name'])


================  =================  ===========  ========  ===========  =============
transform_name    hasher_name          threshold    recall    precision    n_exemplars
================  =================  ===========  ========  ===========  =============
black_frames      phashu8_framewise      51.0979    88.12       99.9069         278644
black_frames      phashu8_tmkl1          55.7584    99.918      99.9079         403768
black_padding     phashu8_framewise      74.6391     7.662     100              277399
black_padding     phashu8_tmkl1          53.8702    99.898      99.9079         406899
clip0.2           phashu8_framewise      54.8635    90.741      99.9098         224264
clip0.2           phashu8_tmkl1          59.0424    99.724      99.9077         324251
gif               phashu8_framewise      55.4437    68.21       99.9088          82232
gif               phashu8_tmkl1          55.4887    81.029      99.9103          39757
noop              phashu8_framewise       0        100         100              282658
noop              phashu8_tmkl1           0        100         100              408871
shrink            phashu8_framewise      24.7184   100         100              281731
shrink            phashu8_tmkl1          49.8999    99.836      99.9078         400650
slideshow         phashu8_framewise      56.9825    99.713      99.9076         172829
slideshow         phashu8_tmkl1          56.8683    95.934      99.9035          90684
================  =================  ===========  ========  ===========  =============