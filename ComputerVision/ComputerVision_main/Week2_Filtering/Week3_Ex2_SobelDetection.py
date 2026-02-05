import cv2
import numpy as np

class SobelDetectionProcessor:
    def __init__(self):
        pass

    def sobel_edges(self, bgr_img):
        """
        Input: BGR image
        Output: edge magnitude image
        """
        if bgr_img is None:
            return None

        sobelx = cv2.Sobel(bgr_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(bgr_img, cv2.CV_64F, 0, 1, ksize=3)

        sobel = cv2.magnitude(sobelx, sobely)
        sobel = np.uint8(np.clip(sobel, 0, 255))
        return sobel
