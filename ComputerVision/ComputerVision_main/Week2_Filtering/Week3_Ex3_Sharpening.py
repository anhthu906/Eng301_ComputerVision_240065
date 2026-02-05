import cv2
import numpy as np


class SharpenProcessor:
    def __init__(self):
        pass

    def apply_sharpen(self, bgr_img):

        if bgr_img is None:
            return None

        # Sharpening kernel
        kernel = np.array([
            [0, -1,  0],
            [-1, 5, -1],
            [0, -1,  0]
        ])

        sharpened = cv2.filter2D(bgr_img, -1, kernel)

        return sharpened
