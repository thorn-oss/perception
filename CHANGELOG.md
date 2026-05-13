# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-05-13
This release moves heavyweight dependencies behind optional extras so they are not installed for users who only need core hashing functionality, and standardizes the error users see when an extra is missing.

### Breaking changes
- `faiss-cpu`, `networkit`, and `networkx` are no longer core dependencies. They are pulled in by the new `approximate-deduplication` extra (`pip install perception[approximate-deduplication]`), which is required to use `perception.approximate_deduplication` or `perception.local_descriptor_deduplication`.
- `pandas` is no longer a core dependency. It is pulled in by the `approximate-deduplication` and `benchmarking` extras (the only modules that use it). Code that imports `perception.benchmarking`, `perception.approximate_deduplication`, `perception.local_descriptor_deduplication`, or `perception.testing` should install the appropriate extra.

### Enhancements
- All optional-dependency import sites — across the `approximate-deduplication`, `benchmarking`, `matching`, and `pdq` extras — now raise a uniform, actionable `ImportError` pointing at the correct `pip install perception[<extra>]` command when the relevant extra is not installed. This is implemented via a single helper, `perception._optional.import_optional`.
- `typing_extensions` is now an explicit core dependency (it was previously transitive via `faiss-cpu` / `pandas`).

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
