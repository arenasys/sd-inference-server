import os
import utils
import numpy as np
import PIL
import torch

SEGMENT_AVAILABLE = False
try:
    from segment_anything import SamPredictor, sam_model_registry
    SEGMENT_AVAILABLE = True
except:
    pass

SEGMENTATION_MODELS = {
    "SAM-ViT-H": ("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth","sam_vit_h.pth", "vit_h"),
    "SAM-ViT-L": ("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth","sam_vit_l.pth", "vit_l"),
    "SAM-ViT-B": ("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth","sam_vit_b.pth", "vit_b")
}

def get_predictor(name, folder, callback):
    if not SEGMENT_AVAILABLE:
        raise Exception("Segment Anything not installed")

    url, file, type = SEGMENTATION_MODELS[name]
    file = os.path.join(folder, file)
    if not os.path.exists(file):
        utils.download(url, file, callback)
    sam = sam_model_registry[type](checkpoint=file)
    return sam

def segment(model, image, points, labels):
    predictor = SamPredictor(model.to(torch.float32))
    predictor.set_image(np.array(image))

    masks, scores, _ = predictor.predict(points, labels, multimask_output=True)

    mask = masks[np.argmax(scores)]*255.0
    mask = mask.reshape(mask.shape[-2], mask.shape[-1]).astype(np.uint8)

    mask_img = PIL.Image.fromarray(mask, 'L')
    inv_img = PIL.Image.fromarray(255-mask, 'L')

    return mask_img, inv_img