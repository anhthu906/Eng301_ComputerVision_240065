import time
import cv2
import numpy as np


class ImageProcessor:
    """
    Class for processing images from camera feed
    """
    frame = None  # BGR numpy array
    def __init__(self, frame=None):
        """Initialize image processor"""
        self.frame = frame
        pass
    
    def process_frame(self, bgr_img):
        """
        Process a single frame
        
        Args:
            bgr_img: Input image in BGR format (numpy array)
            
        Returns:
            tuple: (Processed image, process time in ms)
        """
        if bgr_img is None:
            raise ValueError("Input frame is None")

        start_time = time.perf_counter()

        h, w = bgr_img.shape[:2]
        side = int(min(h, w) * 0.5)
        cx, cy = w // 2, h // 2
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        crop = bgr_img[y0:y0+side, x0:x0+side].copy()

        processed = cv2.resize(crop, (256, 256))

        process_time_ms = (time.perf_counter() - start_time) * 1000

        return processed, process_time_ms
    def preprocess(self, bgr_img, use_gray=True, use_blur=False):
        img = bgr_img
        if use_gray:
            img = self.gray_scale(img)
        if use_blur:
            img = self.gaussian_blur(img, ksize=(5, 5))
        return img
    
    def gray_scale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    def gaussian_blur(self, img, kernel_size=(5, 5), sigma=1.5):
        return cv2.GaussianBlur(img, kernel_size, sigma)





    def postprocess(self, result):
        """
        Postprocess results
        
        Args:
            result: Processed result
            
        Returns:
            Postprocessed result
        """
        # TODO: Implement postprocessing
        pass
