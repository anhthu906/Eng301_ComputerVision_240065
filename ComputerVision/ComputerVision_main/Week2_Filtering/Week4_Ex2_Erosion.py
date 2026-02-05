import cv2
import numpy as np


class ErosionProcessor:

    def __init__(self):
        pass

    def apply_erosion(self, bgr_img, kernel_size=(5, 5), iterations=2):

        if bgr_img is None:
            return None

        # Create structuring element
        kernel = np.ones(kernel_size, np.uint8)

        eroded = cv2.erode(bgr_img, kernel, iterations=iterations)

        return eroded
