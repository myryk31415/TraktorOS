"""Utilities for finding the horizon line"""

import cv2
import numpy as np


def detect_horizon_y_middle(image):
    """
    Detect the horizon and return the y-coordinate at the horizontal
    center of the image.

    Parameters
    ----------
        image: the image
    Returns
    -------
    int
        y-coordinate of the horizon at the middle of the image
    """

    image_grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    msg = "`image_grayscaled` must be a grayscale image of shape (height, width)"
    assert image_grayscaled.ndim == 2, msg

    # blur to reduce noise
    image_blurred = cv2.GaussianBlur(image_grayscaled, (3, 3), 0)

    # Otsu threshold
    _, image_thresholded = cv2.threshold(
        image_blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    image_thresholded = image_thresholded - 1

    # morphological closing
    image_closed = cv2.morphologyEx(
        image_thresholded, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
    )

    height, width = image_grayscaled.shape
    x_mid = width // 2

    # horizon = last sky pixel in center column
    horizon_y_mid = max(np.where(image_closed[:, x_mid] == 0)[0])

    return int(horizon_y_mid)
