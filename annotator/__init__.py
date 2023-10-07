import cv2
import numpy as np

import annotator.hed
import annotator.lineart
import annotator.lineart_anime
import annotator.midas
import annotator.openpose
import annotator.mlsd
import annotator.normalbae

annotators = {
    "Softedge": annotator.hed.HEDdetector,
    "Lineart": annotator.lineart.LineartDetector,
    "Anime": annotator.lineart_anime.LineartAnimeDetector,
    "Depth": annotator.midas.MidasDetector,
    "Pose": annotator.openpose.OpenposeDetector,
    "M-LSD": annotator.mlsd.MLSDdetector,
    "Scribble": annotator.hed.HEDdetectorScribble,
    "Normal": annotator.normalbae.NormalBaeDetector
}

def make_noise_disk(H, W, C, F):
    noise = np.random.uniform(low=0, high=1, size=((H // F) + 2, (W // F) + 2, C))
    noise = cv2.resize(noise, (W + 2 * F, H + 2 * F), interpolation=cv2.INTER_CUBIC)
    noise = noise[F: F + H, F: F + W]
    noise -= np.min(noise)
    noise /= np.max(noise)
    if C == 1:
        noise = noise[:, :, None]
    return noise

def shuffle(img, f=256):
    h, w, _ = img.shape
    x = make_noise_disk(h, w, 1, f) * float(w - 1)
    y = make_noise_disk(h, w, 1, f) * float(h - 1)
    flow = np.concatenate([x, y], axis=2).astype(np.float32)
    return cv2.remap(img, flow, None, cv2.INTER_LINEAR)

def canny(img, low_threshold=0.4, high_threshold=0.8):
    return cv2.Canny(img, int(low_threshold*255), int(high_threshold*255))