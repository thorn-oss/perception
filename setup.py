# -*- coding: utf-8 -*-
from setuptools import setup

packages = [
    "perception",
    "perception.approximate_deduplication",
    "perception.benchmarking",
    "perception.hashers",
    "perception.hashers.image",
    "perception.hashers.video",
    "perception.testing",
]

package_data = {"": ["*"], "perception.testing": ["images/*", "logos/*", "videos/*"]}

extras_require = {
    "benchmarking": [
        "matplotlib",
        "scipy",
        "albumentations",
        "tabulate",
        "scikit-learn",
        "ffmpeg-python",
    ],
    "experimental": ["networkit", "faiss-cpu"],
    "matching": ["aiohttp", "python-json-logger"],
}

setup_kwargs = {
    "name": "Perception",
    "version": "0.0.0",
    "description": "Perception provides flexible, well-documented, and comprehensively tested tooling for perceptual hashing research, development, and production use.",
    "long_description": "# perception ![ci](https://github.com/thorn-oss/perception/workflows/ci/badge.svg)\n\n`perception` provides flexible, well-documented, and comprehensively tested tooling for perceptual hashing research, development, and production use. See [the documentation](https://perception.thorn.engineering/en/latest/) for details.\n\n## Background\n\n`perception` was initially developed at [Thorn](https://www.thorn.org) as part of our work to eliminate child sexual abuse material from the internet. For more information on the issue, check out [our CEO's TED talk](https://www.thorn.org/blog/time-is-now-eliminate-csam/).\n\n## Getting Started\n\n### Installation\n\n`pip install perception`\n\n### Hashing\n\nHashing with different functions is simple with `perception`.\n\n```python\nfrom perception import hashers\n\nfile1, file2 = 'test1.jpg', 'test2.jpg'\nhasher = hashers.PHash()\nhash1, hash2 = hasher.compute(file1), hasher.compute(file2)\ndistance = hasher.compute_distance(hash1, hash2)\n```\n\n### Examples\n\nSee below for end-to-end examples for common use cases for perceptual hashes.\n\n- [Detecting child sexual abuse material](https://perception.thorn.engineering/en/latest/examples/detecting_csam.html)\n- [Deduplicating media](https://perception.thorn.engineering/en/latest/examples/deduplication.html)\n- [Benchmarking perceptual hashes](https://perception.thorn.engineering/en/latest/examples/benchmarking.html)\n\n## Supported Hashing Algorithms\n\n`perception` currently ships with:\n\n- pHash (DCT hash) (`perception.hashers.PHash`)\n- Facebook's PDQ Hash (`perception.hashers.PDQ`)\n- dHash (difference hash) (`perception.hashers.DHash`)\n- aHash (average hash) (`perception.hashers.AverageHash`)\n- Marr-Hildreth (`perception.hashers.MarrHildreth`)\n- Color Moment (`perception.hashers.ColorMoment`)\n- Block Mean (`perception.hashers.BlockMean`)\n- wHash (wavelet hash) (`perception.hashers.WaveletHash`)\n\n## Contributing\n\nTo work on the project, start by doing the following.\n\n```bash\n# Install local dependencies for\n# code completion, etc.\nmake init\n\n- To do a (close to) comprehensive check before committing code, you can use `make precommit`.\n\nTo implement new features, please first file an issue proposing your change for discussion.\n\nTo report problems, please file an issue with sample code, expected results, actual results, and a complete traceback.\n\n## Alternatives\n\nThere are other packages worth checking out to see if they meet your needs for perceptual hashing. Here are some\nexamples.\n\n- [dedupe](https://github.com/dedupeio/dedupe)\n- [imagededup](https://idealo.github.io/imagededup/)\n- [ImageHash](https://github.com/JohannesBuchner/imagehash)\n- [PhotoHash](https://github.com/bunchesofdonald/photohash)\n```\n",
    "author": "Thorn",
    "author_email": "info@wearethorn.org",
    "maintainer": "None",
    "maintainer_email": "None",
    "url": "None",
    "packages": packages,
    "package_data": package_data,
    "extras_require": extras_require,
    "python_requires": ">=3.10,<4.0",
}
from build import *

build(setup_kwargs)

setup(**setup_kwargs)
