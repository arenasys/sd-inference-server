import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodules import UpSampleBN, UpSampleGN, norm_normalize

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        # hyper-parameter for sampling
        self.sampling_ratio = args.sampling_ratio
        self.importance_ratio = args.importance_ratio

        # feature-map
        self.conv2 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)
        if args.architecture == 'BN':
            self.up1 = UpSampleBN(skip_input=2048 + 176, output_features=1024)
            self.up2 = UpSampleBN(skip_input=1024 + 64, output_features=512)
            self.up3 = UpSampleBN(skip_input=512 + 40, output_features=256)
            self.up4 = UpSampleBN(skip_input=256 + 24, output_features=128)

        elif args.architecture == 'GN':
            self.up1 = UpSampleGN(skip_input=2048 + 176, output_features=1024)
            self.up2 = UpSampleGN(skip_input=1024 + 64, output_features=512)
            self.up3 = UpSampleGN(skip_input=512 + 40, output_features=256)
            self.up4 = UpSampleGN(skip_input=256 + 24, output_features=128)

        else:
            raise Exception('invalid architecture')

        # produces 1/8 res output
        self.out_conv_res8 = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)

        # produces 1/4 res output
        self.out_conv_res4 = nn.Sequential(
            nn.Conv1d(512 + 4, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 4, kernel_size=1),
        )

        # produces 1/2 res output
        self.out_conv_res2 = nn.Sequential(
            nn.Conv1d(256 + 4, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 4, kernel_size=1),
        )

        # produces 1/1 res output
        self.out_conv_res1 = nn.Sequential(
            nn.Conv1d(128 + 4, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 4, kernel_size=1),
        )

    def forward(self, features, gt_norm_mask=None, mode='test'):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        # generate feature-map

        x_d0 = self.conv2(x_block4)                     # x_d0 : [2, 2048, 15, 20]      1/32 res
        x_d1 = self.up1(x_d0, x_block3)                 # x_d1 : [2, 1024, 30, 40]      1/16 res
        x_d2 = self.up2(x_d1, x_block2)                 # x_d2 : [2, 512, 60, 80]       1/8 res
        x_d3 = self.up3(x_d2, x_block1)                 # x_d3: [2, 256, 120, 160]      1/4 res
        x_d4 = self.up4(x_d3, x_block0)                 # x_d4: [2, 128, 240, 320]      1/2 res

        # 1/8 res output
        out_res8 = self.out_conv_res8(x_d2)             # out_res8: [2, 4, 60, 80]      1/8 res output
        out_res8 = norm_normalize(out_res8)             # out_res8: [2, 4, 60, 80]      1/8 res output

        ################################################################################################################
        # out_res4
        ################################################################################################################

        # grid_sample feature-map
        feat_map = F.interpolate(x_d2, scale_factor=2, mode='bilinear', align_corners=True)
        init_pred = F.interpolate(out_res8, scale_factor=2, mode='bilinear', align_corners=True)
        feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 512+4, H, W)
        B, _, H, W = feat_map.shape

        # try all pixels
        out_res4 = self.out_conv_res4(feat_map.view(B, 512 + 4, -1))  # (B, 4, N)
        out_res4 = norm_normalize(out_res4)  # (B, 4, N) - normalized
        out_res4 = out_res4.view(B, 4, H, W)
        samples_pred_res4 = point_coords_res4 = None

        ################################################################################################################
        # out_res2
        ################################################################################################################

        # grid_sample feature-map
        feat_map = F.interpolate(x_d3, scale_factor=2, mode='bilinear', align_corners=True)
        init_pred = F.interpolate(out_res4, scale_factor=2, mode='bilinear', align_corners=True)
        feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 512+4, H, W)
        B, _, H, W = feat_map.shape

        out_res2 = self.out_conv_res2(feat_map.view(B, 256 + 4, -1))  # (B, 4, N)
        out_res2 = norm_normalize(out_res2)  # (B, 4, N) - normalized
        out_res2 = out_res2.view(B, 4, H, W)
        samples_pred_res2 = point_coords_res2 = None

        ################################################################################################################
        # out_res1
        ################################################################################################################

        # grid_sample feature-map
        feat_map = F.interpolate(x_d4, scale_factor=2, mode='bilinear', align_corners=True)
        init_pred = F.interpolate(out_res2, scale_factor=2, mode='bilinear', align_corners=True)
        feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 512+4, H, W)
        B, _, H, W = feat_map.shape

        out_res1 = self.out_conv_res1(feat_map.view(B, 128 + 4, -1))  # (B, 4, N)
        out_res1 = norm_normalize(out_res1)  # (B, 4, N) - normalized
        out_res1 = out_res1.view(B, 4, H, W)
        samples_pred_res1 = point_coords_res1 = None

        return [out_res8, out_res4, out_res2, out_res1], \
               [out_res8, samples_pred_res4, samples_pred_res2, samples_pred_res1], \
               [None, point_coords_res4, point_coords_res2, point_coords_res1]

