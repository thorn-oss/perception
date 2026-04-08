# perception ![ci](https://github.com/thorn-oss/perception/workflows/ci/badge.svg)

`perception` provides flexible, well-documented, and comprehensively tested tooling for perceptual hashing research, development, and production use. See [the documentation](https://perception.thorn.engineering/en/latest/) for details.

## Background

`perception` was initially developed at [Thorn](https://www.thorn.org) as part of our work to eliminate child sexual abuse material from the internet. For more information on the issue, check out [our CEO's TED talk](https://www.thorn.org/blog/time-is-now-eliminate-csam/).

## Getting Started

### Installation

`pip install perception`

#### Optional extras

`perception` provides optional extras for additional functionality:

- `benchmarking` – tools for benchmarking perceptual hashes
- `matching` – async matching utilities
- `pdq` – Facebook's PDQ hash support

**Note for `benchmarking` extra users:** The `benchmarking` extra depends on
`albumentations`, which in turn requires `opencv-python-headless`. However,
`perception` already depends on `opencv-contrib-python-headless` (needed for
contrib modules such as `cv2.img_hash` and `cv2.SIFT_create`). Installing both
OpenCV distributions simultaneously causes file-level conflicts.

If you are using [uv](https://docs.astral.sh/uv/), this is handled
automatically:

```bash
uv pip install "perception[benchmarking]"
```

If you are using plain `pip`, install the extra and then force-reinstall the
contrib variant to remove the conflicting headless package:

```bash
pip install "perception[benchmarking]"
pip install --force-reinstall --no-deps opencv-contrib-python-headless
```

### Hashing

Hashing with different functions is simple with `perception`.

```python
from perception import hashers

file1, file2 = 'test1.jpg', 'test2.jpg'
hasher = hashers.PHash()
hash1, hash2 = hasher.compute(file1), hasher.compute(file2)
distance = hasher.compute_distance(hash1, hash2)
```

### Examples

See below for end-to-end examples for common use cases for perceptual hashes.

- [Detecting child sexual abuse material](https://perception.thorn.engineering/en/latest/examples/detecting_csam.html)
- [Deduplicating media](https://perception.thorn.engineering/en/latest/examples/deduplication.html)
- [Benchmarking perceptual hashes](https://perception.thorn.engineering/en/latest/examples/benchmarking.html)

## Supported Hashing Algorithms

`perception` currently ships with:

- pHash (DCT hash) (`perception.hashers.PHash`)
- Facebook's PDQ Hash (`perception.hashers.PDQ`)
- dHash (difference hash) (`perception.hashers.DHash`)
- aHash (average hash) (`perception.hashers.AverageHash`)
- Marr-Hildreth (`perception.hashers.MarrHildreth`)
- Color Moment (`perception.hashers.ColorMoment`)
- Block Mean (`perception.hashers.BlockMean`)
- wHash (wavelet hash) (`perception.hashers.WaveletHash`)

## Contributing

To work on the project, start by doing the following.

```bash
# Install local dependencies for code completion,
# testing, and linting.
make init
```

To do a (close to) comprehensive check before committing code, use `make precommit`.

To implement new features, please first file an issue proposing your change for discussion.

To report problems, please file an issue with sample code, expected results, actual results, and a complete traceback.

## Alternatives

There are other packages worth checking out to see if they meet your needs for perceptual hashing. Here are some
examples.

- [dedupe](https://github.com/dedupeio/dedupe)
- [imagededup](https://idealo.github.io/imagededup/)
- [ImageHash](https://github.com/JohannesBuchner/imagehash)
- [PhotoHash](https://github.com/bunchesofdonald/photohash)
