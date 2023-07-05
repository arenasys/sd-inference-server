'''
modified by  lihaoweicv
pytorch version
'''

'''
M-LSD
Copyright 2021-present NAVER Corp.
Apache License v2.0
'''

import os
import numpy as np
import cv2
import torch
from  torch.nn import  functional as F


def deccode_output_score_and_ptss(tpMap, topk_n = 200, ksize = 5):
    '''
    tpMap:
    center: tpMap[1, 0, :, :]
    displacement: tpMap[1, 1:5, :, :]
    '''
    b, c, h, w = tpMap.shape
    assert  b==1, 'only support bsize==1'
    displacement = tpMap[:, 1:5, :, :][0]
    center = tpMap[:, 0, :, :]
    heat = torch.sigmoid(center)
    hmax = F.max_pool2d( heat, (ksize, ksize), stride=1, padding=(ksize-1)//2)
    keep = (hmax == heat).float()
    heat = heat * keep
    heat = heat.reshape(-1, )

    scores, indices = torch.topk(heat, topk_n, dim=-1, largest=True)
    yy = torch.floor_divide(indices, w).unsqueeze(-1)
    xx = torch.fmod(indices, w).unsqueeze(-1)
    ptss = torch.cat((yy, xx),dim=-1)

    ptss   = ptss.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    displacement = displacement.detach().cpu().numpy()
    displacement = displacement.transpose((1,2,0))
    return  ptss, scores, displacement


def pred_lines(image, model, input_shape, score_thr, dist_thr, device, dtype):
    h, w, _ = image.shape
    h_ratio, w_ratio = [h / input_shape[0], w / input_shape[1]]

    resized_image = np.concatenate([cv2.resize(image, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA),
                                    np.ones([input_shape[0], input_shape[1], 1])], axis=-1)

    resized_image = resized_image.transpose((2,0,1))
    batch_image = np.expand_dims(resized_image, axis=0).astype('float32')
    batch_image = (batch_image / 127.5) - 1.0

    batch_image = torch.from_numpy(batch_image).to(device, dtype)
    outputs = model(batch_image)
    pts, pts_score, vmap = deccode_output_score_and_ptss(outputs, 200, 3)
    start = vmap[:, :, :2]
    end = vmap[:, :, 2:]
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

    segments_list = []
    for center, score in zip(pts, pts_score):
        y, x = center
        distance = dist_map[y, x]
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])

    lines = 2 * np.array(segments_list)  # 256 > 512
    lines[:, 0] = lines[:, 0] * w_ratio
    lines[:, 1] = lines[:, 1] * h_ratio
    lines[:, 2] = lines[:, 2] * w_ratio
    lines[:, 3] = lines[:, 3] * h_ratio

    return lines