import numpy as np
import cv2
from einops import rearrange

class CannyDetector:
    def __init__(self, path, download):
        return
    def __call__(self, img, low_threshold=0.4, high_threshold=0.8):
        return cv2.Canny(img, int(low_threshold*255), int(high_threshold*255))
    def to(self, device, dtype):
        return self