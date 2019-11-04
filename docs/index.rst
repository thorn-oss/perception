perception
==========

:code:`perception` provides flexible, well-documented, and comprehensively tested tooling for perceptual hashing
research, development, and production use. It provides a common wrapper around existing, popular perceptual hashes
(such as those implemented by `ImageHash <https://pypi.org/project/ImageHash/>`_)
along with tools to compare their performance and use them for common tasks.

Perceptual hashes are used to create compact image "fingerprints" which are invariant to small alterations to
the original image. Typically, the representations are compact enough that they are irreversible, which makes
them useful for deduplication and detecting abusive content while preserving the privacy of content owners.

Installation
************

You can install :code:`perception` using pip. You must install OpenCV separately (e.g., with :code:`pip install opencv-python`).

.. code-block:: bash
    
    # Install from PyPi
    pip install perception

    # Install from GitHub
    pip install git+https://github.com/thorn-oss/perception.git#egg=perception

To install with the necessary dependencies for benchmarking, use:

.. code-block:: bash

    # Install from PyPi
    pip install perception[benchmarking]

    # Install from GitHub
    pip install opencv-python git+https://github.com/thorn-oss/perception.git#egg=perception[benchmarking]

Getting Started
***************

Please see the examples for code snippets for common use cases.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   examples/index
   api/index
    
