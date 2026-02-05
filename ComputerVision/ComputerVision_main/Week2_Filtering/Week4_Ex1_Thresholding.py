import cv2


class ThresholdProcessor:

    def __init__(self):
        pass

    def apply_binary_threshold(self, img, thresh=127, max_val=255):

        if img is None:
            return None

        # Ensure grayscale (threshold works on single channel)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(img, thresh, max_val, cv2.THRESH_BINARY)

        return binary
