import PIL.Image
import torch
import PIL

from ultralytics import YOLO
from ultralytics.utils import LOGGER
LOGGER.disabled = True

import torchvision.ops.boxes as bops
from torchvision.transforms.functional import to_pil_image

class ADetailer(YOLO):
    def merge_bbox(self, bboxs, max_overlap):
        for a in bboxs:
            for b in bboxs:
                if a == b:
                    continue
                box1 = torch.tensor([a], dtype=torch.float)
                box2 = torch.tensor([b], dtype=torch.float)
                overlap = bops.box_iou(box1, box2)
                if overlap > max_overlap:
                    bboxs.remove(a)
                    bboxs.remove(b)
                    bboxs += [[min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]]
                    return self.merge_bbox(bboxs, max_overlap)
        return bboxs

    def predict_masks(self, image, confidence=0.5, mode="rectangle"):
        self.to(torch.float32)
        pred = self(image, conf=confidence)

        preview = pred[0].plot()
        preview = PIL.Image.fromarray(preview[:, :, ::-1])

        masks = []
        if pred[0].masks is None:
            boxes = pred[0].boxes.xyxy.cpu().numpy().tolist()
            #boxes = self.merge_bbox(boxes, max_overlap=0.5)
            for box in boxes:
                mask = PIL.Image.new("L", image.size, 0)
                mask_draw = PIL.ImageDraw.Draw(mask)
                if mode == "rectangle":
                    mask_draw.rectangle(box, fill=255)
                else:
                    mask_draw.ellipse(box, fill=255)
                masks.append(mask)
        else:
            masks = pred[0].masks.data
            masks = [to_pil_image(masks[i], mode="L").resize(image.size) for i in range(masks.shape[0])]

        processed = []
        for i in range(len(masks)):
            conf = pred[0].boxes[i].conf.item()
            cls = pred[0].names[pred[0].boxes[i].cls.item()]
            if conf < confidence:
                continue
            processed += [(cls, masks[i])]
        
        return processed, preview

    def get_device(self):
        return next(self.model.parameters()).device

    def __getattr__(self, name):
        if name == "device":
            return self.get_device()
        return super().__getattr__(name)