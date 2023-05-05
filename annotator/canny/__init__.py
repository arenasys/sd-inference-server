import numpy as np
import cv2
from einops import rearrange

class CannyDetector:
    def __init__(self, path):
        return
    def __call__(self, img, low_threshold=100, high_threshold=200):
        return cv2.Canny(img, low_threshold, high_threshold)
    def to(self, device, dtype):
        return self