import cv2


class BilateralProcessor:

    def __init__(self):
        pass

    def apply_bilateral_filter(self, bgr_img, d=12, sigmaColor=150, sigmaSpace=150):

        if bgr_img is None:
            return None

        filtered_img = cv2.bilateralFilter(bgr_img, d, sigmaColor, sigmaSpace)

        return filtered_img
