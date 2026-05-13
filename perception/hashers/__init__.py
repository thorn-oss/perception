from .hasher import ImageHasher, VideoHasher
from .image._average import AverageHash
from .image._dhash import DHash
from .image._opencv import BlockMean, ColorMoment, MarrHildreth
from .image._phash import PHash, PHashF, PHashU8
from .image._wavelet import WaveletHash
from .video._framewise import FramewiseHasher
from .video._tmk import TMKL1, TMKL2

__all__ = [
    "ImageHasher",
    "VideoHasher",
    "AverageHash",
    "PHash",
    "WaveletHash",
    "MarrHildreth",
    "BlockMean",
    "ColorMoment",
    "DHash",
    "FramewiseHasher",
    "TMKL1",
    "TMKL2",
    "PHashU8",
    "PHashF",
]

try:
    from .image._pdq import PDQHash as PDQHash, PDQHashF as PDQHashF
except ImportError:
    pass
else:
    __all__.extend(["PDQHash", "PDQHashF"])
