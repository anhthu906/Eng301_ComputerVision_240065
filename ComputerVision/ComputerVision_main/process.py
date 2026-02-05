import time
import cv2
import numpy as np
import os
from Week1_Capturering.Week1_CaptureSaveImg import CaptureSaveImgProcessor
from Week2_Filtering.Week2_Ex1_Grayscale import GrayscaleProcessor
from Week2_Filtering.Week2_Ex2_Gaussian import GaussianProcessor
from Week2_Filtering.Week2_Ex3_Meanblur import MeanblurProcessor
from Week2_Filtering.Week3_Ex2_SobelDetection import SobelDetectionProcessor
from Week2_Filtering.Week3_Ex1_LaplacianDetection import LaplaciandetectionProcessor 
from Week2_Filtering.Week3_Ex3_Sharpening import SharpenProcessor
from Week2_Filtering.Week3_Ex4_Bilateral import BilateralProcessor 
from Week2_Filtering.Week4_Ex1_Thresholding import ThresholdProcessor
from Week2_Filtering.Week4_Ex2_Erosion import ErosionProcessor
from Week2_Filtering.Week4_Ex3_Dilation import DilationProcessor


class ImageProcessor:
    """
    Class for processing images from camera feed
    Implements computer vision techniques from ProjectProgress.txt
    """
    
    def __init__(self):
        """Initialize image processor with calibration parameters"""
        self.camera_matrix = None  # Camera calibration matrix
        self.dist_coeffs = None    # Distortion coefficients
        self.homography_matrix = None  # Homography transformation matrix
        self.previous_frame = None  # For motion detection
        self.tracked_objects = []   # For object tracking
        
        self.filters = {
            "grayscale": False,
            "gaussian": True,
            "meanblur": False,
            "sobel": False,
            "laplacian": False,
            "sharpen": False,
            "bilateral": False,
            "threshold": False,
            "erosion": False,
            "dilation": False
    }

    
    def process_frame(self, bgr_img):

        if bgr_img is None:
            raise ValueError("Input frame is None")
 
        img = bgr_img.copy()

        start_time = time.perf_counter()
        results = {}

        save_processor = CaptureSaveImgProcessor()

        if self.filters["grayscale"]:
            img = GrayscaleProcessor().convert_to_grayscale(img)

        if self.filters["gaussian"]:
            img = GaussianProcessor().apply_gaussian_filter(img)

        if self.filters["meanblur"]:
            img = MeanblurProcessor().apply_mean_filter(img)

        if self.filters["sobel"]:
            img = SobelDetectionProcessor().sobel_edges(img)

        if self.filters["laplacian"]:
            img = LaplaciandetectionProcessor().laplacian_edges(img)

        if self.filters["sharpen"]:
            img = SharpenProcessor().apply_sharpen(img)    

        if self.filters["bilateral"]:
            img = BilateralProcessor().apply_bilateral_filter(img)    

        if self.filters["threshold"]:
            img = ThresholdProcessor().apply_binary_threshold(img)    

        if self.filters["erosion"]:
            img = ErosionProcessor().apply_erosion(img)    

        if self.filters["dilation"]:
            img = DilationProcessor().apply_dilation(img)    
            

        # Step 4: Save processed (edge)
        save_processor.capture_and_save_image(img, "processed.bmp")

        process_time_ms = (time.perf_counter() - start_time) * 1000

        return img, results, process_time_ms


    
    def visualize_results(self, bgr_img, results):
        """
        Visualize all processing results on image
        
        Args:
            bgr_img: Original image
            results: Dictionary of results from process_frame
            
        Returns:
            Annotated image
        """

        pass