# pylint: disable=invalid-name

import cv2
import numpy as np


def apply_watermark(watermark, alpha: float = 1., size: float = 1.):
    """Apply a watermark to the bottom right of
    images. Based on the work provided at
    https://www.pyimagesearch.com/2016/04/25/watermarking-images-with-opencv-and-python/

    Args:
        watermark: The watermark to overlay
        alpha: The strength of the overlay
        size: The maximum proportion of the image
            taken by the watermark.
    """
    assert watermark.shape[-1] == 4, "Watermark must have an alpha channel."

    # Why do we have to do this? It's not clear. But the process doesn't work
    # without it.
    (B, G, R, A) = cv2.split(watermark)
    B = cv2.bitwise_and(B, B, mask=A)
    G = cv2.bitwise_and(G, G, mask=A)
    R = cv2.bitwise_and(R, R, mask=A)
    watermark = cv2.merge([B, G, R, A])

    def transform(image):
        # Add alpha channel
        (h, w) = image.shape[:2]
        wh, ww = watermark.shape[:2]
        scale = size * min(h / wh, w / ww)
        image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])
        # Construct an overlay that is the same size as the input.
        overlay = np.zeros((h, w, 4), dtype="uint8")
        scaled = cv2.resize(watermark, (int(scale * ww), int(scale * wh)))
        sh, sw = scaled.shape[:2]
        overlay[max(h - sh, 0):, max(w - sw, 0):w] = scaled
        # Blend the two images together using transparent overlays
        output = image.copy()
        cv2.addWeighted(overlay, alpha, output, 1.0, 0, output)
        return cv2.cvtColor(output, cv2.COLOR_RGBA2RGB)

    return transform
