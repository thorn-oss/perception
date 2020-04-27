# perception ![ci](https://github.com/thorn-oss/perception/workflows/ci/badge.svg)

`perception` provides flexible, well-documented, and comprehensively tested tooling for perceptual hashing research, development, and production use. See [the documentation](https://perception.thorn.engineering/en/latest/) for details.

## Background
`perception` was initially developed at [Thorn](https://www.thorn.org) as part of our work to eliminate child sexual abuse material from the internet. For more information on the issue, check out [our CEO's TED talk](https://www.thorn.org/blog/time-is-now-eliminate-csam/).

## Getting Started

### Installation
`pip install opencv-python perception`

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
# Install local dependencies for
# code completion, etc.
make init

# Build the Docker container to run
# tests and such.
make build
```

- You can get a JupyterLab server running to experiment with using `make lab-server`.
- To do a (close to) comprehensive check before committing code, you can use `make precommit`.
- To view the documentation, use `make documentation-server`.

To implement new features, please first file an issue proposing your change for discussion.

To report problems, please file an issue with sample code, expected results, actual results, and a complete traceback.

## Alternatives
There are other packages worth checking out to see if they meet your needs for perceptual hashing. Here are some
examples.

- [dedupe](https://github.com/dedupeio/dedupe)
- [imagededup](https://idealo.github.io/imagededup/)
- [ImageHash](https://github.com/JohannesBuchner/imagehash)
- [PhotoHash](https://github.com/bunchesofdonald/photohash)
