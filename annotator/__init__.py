import cv2
import numpy as np
import math

import annotator.hed
import annotator.lineart
import annotator.lineart_anime
import annotator.midas
import annotator.openpose
import annotator.mlsd
import annotator.normalbae
import annotator.anyline

annotators = {
    "Softedge": annotator.hed.HEDdetector,
    "Lineart": annotator.lineart.LineartDetector,
    "Anime": annotator.lineart_anime.LineartAnimeDetector,
    "Depth": annotator.midas.MidasDetector,
    "Pose": annotator.openpose.OpenposeDetector,
    "M-LSD": annotator.mlsd.MLSDdetector,
    "Scribble": annotator.hed.HEDdetectorScribble,
    "Normal": annotator.normalbae.NormalBaeDetector,
    "Anyline": annotator.anyline.AnylineDetector
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

POSE_LIMBS = [
    [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
    [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
    [1, 16], [16, 18], [3, 17], [6, 18]
]

POSE_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
    [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
    [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
]

def unpack_pose(pose):
    candidate = pose["bodies"]["candidate"]
    subset = pose["bodies"]["subset"]

    bodies = []
    for s in range(len(subset)):
        idx = [int(subset[s][i]) for i in range(18)]
        bodies += [[candidate[i] if i >= 0 else None for i in idx]]
    return bodies

def draw_pose(canvas, pose):
    H, W, _ = canvas.shape

    for body in pose:
        for i in range(17):
            limb = [POSE_LIMBS[i][0]-1, POSE_LIMBS[i][1]-1]
            a = body[limb[0]]
            b = body[limb[1]]
            if a == None or b == None:
                continue

            aX, aY = a[0] * float(W), a[1] * float(H)
            bX, bY = b[0] * float(W), b[1] * float(H)

            mX, mY = (aX + bX)/2, (aY + bY)/2

            length = ((aX - bX) ** 2 + (aY - bY) ** 2) ** 0.5
            angle = math.degrees(math.atan2(aY - bY, aX - bX))
            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), 4), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, POSE_COLORS[i])
    
    canvas = (canvas * 0.6).astype(np.uint8)

    for body in pose:
        for i in range(18):
            point = body[i]
            if not point:
                continue
            
            point = [int(point[0] * W), int(point[1] * H)]
            cv2.circle(canvas, point, 4, POSE_COLORS[i], thickness=-1)

    return canvas