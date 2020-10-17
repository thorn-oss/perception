# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2020-10-17
This release switches from using false positive rates in benchmarking to reporting precision, which is more intuitive.

### Breaking changes
All references to fpr_threshold now refer to precision_threshold.

### Bug fixes
The PDQHash hasher now correctly returns the hash vector instead of the (vector, quality) tuple.

## [0.3.0] - 2020-04-27
This release adds significantly more support for video.

### Breaking changes
- Previously, `read_video` returned `(frame, index, timestamp)` tuples where `index` reflected the index of the yielded frame (i.e., it always increased by exactly 1). It now reflects the index of the frame in the original video. This means that, if the requested framerate is higher than the encoded video framerate, this index may repeat the same value, indicating that we have repeated the same frame.

### Enhancements
- We now include a `SimpleSceneDetection` hasher that can wrap other video hashers using scene detection.
- `compute_metrics` is much faster now for integer-valued hashes that use a euclidean distance metric.
- We now include an unsigned 8-bit integer version of `PHash`, called `PHashU8`. This provides a useful framewise hasher for averaging across frames (e.g., using TMK) while being more compact than `PHashF`.
- We include more thorough support for benchmarking video hashes.

### Bug fixes
- When using `hasher.vector_to_string` with hashers that return multiple hashes, the `hash_format` argument was not respected.
- The `compute_threshold_recall` and `show_histograms` functions did not work properly when `grouping=[]`.

## [0.2.0] - 2019-12-20
This release adds more support for hashing videos (including TMK L2 and TMK L2). As part of that, it also includes a re-factor to separate `benchmarking.BenchmarkDataset` and `benchmarking.BenchmarkTransforms` into image and video variants.

## [0.1.0] - 2019-11-04
Initial release