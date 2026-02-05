import cv2
import numpy as np

class LaplaciandetectionProcessor:
    def __init__(self):
        pass

    def laplacian_edges(self, gray_img):
        """
        Input: grayscale image
        Output: edge image using Laplacian
        """
        if gray_img is None:
            return None

        # Apply Laplacian (second-order derivative)
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F, ksize=3)

        # Take absolute value and convert to uint8
        laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))

        return laplacian
