import cv2
class MeanblurProcessor:
    def __init__(self):
        pass

    def apply_mean_filter(self, img, kernel_size=(15, 15)):

        if img is None:
            return None

        filtered_img = cv2.blur(img, ksize=kernel_size)
        return filtered_img
    
    