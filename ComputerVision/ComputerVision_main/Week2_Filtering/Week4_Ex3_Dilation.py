import cv2
import numpy as np


class DilationProcessor:

    def __init__(self):
        pass

    def apply_dilation(self, bgr_img, kernel_size=(5, 5), iterations=2):

        if bgr_img is None:
            return None

        kernel = np.ones(kernel_size, np.uint8)

        dilated = cv2.dilate(bgr_img, kernel, iterations=iterations)

        return dilated
