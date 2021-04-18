import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from .. import builder
from .single_stage import SingleStage3DDetector
import torch.nn as nn
import torchvision
import numpy as np


@DETECTORS.register_module()
class VoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(VoxelNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)

        # 转换RangeImage部分相关代码
        self.H = 64
        self.W = 1024
        self.fov_up = 3
        self.fov_down = -15.0
        self.pi = torch.tensor(np.pi)
        fov_up = self.fov_up * self.pi / 180.0
        fov_down = self.fov_down * self.pi / 180.0
        fov = abs(fov_up) + abs(fov_down)
        self.uv = torch.zeros((2, self.H, self.W))
        self.uv[1] = torch.arange(0, self.W)
        self.uv.permute((0, 2, 1))[0] = torch.arange(0, self.H)
        self.uv[0] = ((self.H - self.uv[0]) * fov - abs(fov_down) * self.H) / self.H
        self.uv[1] = (self.uv[1] * 2.0 - self.W) * self.pi / (self.W * 4)  # 最后一个 4 用来控制水平范围

        self.fusion = Fusion(5, 3)

    def extract_feat(self, points, img_metas):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        #提取出反射率信息
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      img = None,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """

        # 转换成range提取特征后转回lidar
        batchsize = len(points)
        rangeImage = []
        for i in range(batchsize):
            rangeImage.append(self.lidar_to_range_gpu(points[i]).unsqueeze(0))
        rangeImage = torch.cat(rangeImage, dim=0)
        # 是否加入img信息
        range_feat = self.fusion(rangeImage, img)
        range_ori = torch.cat((rangeImage[:, 0:2], range_feat), dim=1)
        pts_with_range = []
        for i in range(batchsize):
            pts_with_range.append(self.range_to_lidar_gpu(range_ori[i].squeeze(0)))

        x = self.extract_feat(pts_with_range, img_metas)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""

        # 转换成range提取特征后转回lidar
        batchsize = len(points)
        rangeImage = []
        for i in range(batchsize):
            rangeImage.append(self.lidar_to_range_gpu(points[i]).unsqueeze(0))
        rangeImage = torch.cat(rangeImage, dim=0)
        # 是否加入img信息
        # range_feat = self.range_encoder(rangeImage, imgs)      #用自编码器的形式
        range_feat = self.fusion(rangeImage, imgs)
        range_ori = torch.cat((rangeImage[:, 0:2], range_feat), dim=1)
        pts_with_range = []
        for i in range(batchsize):
            pts_with_range.append(self.range_to_lidar_gpu(range_ori[i].squeeze(0)))

        x = self.extract_feat(pts_with_range, img_metas)
        outs = self.bbox_head(x)

        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]

    def lidar_to_range_gpu(self, points):
        device = points.device
        pi = torch.tensor(np.pi).to(device)
        fov_up = self.fov_up * pi / 180.0
        fov_down = self.fov_down * pi / 180.0
        fov = abs(fov_up) + abs(fov_down)

        depth = torch.norm(points, 2, dim=1)

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        yaw = torch.atan2(y, x)
        pitch = torch.asin(z / depth)

        u = 0.5 * (1 - 4 * yaw / pi) * self.W  # 最后一个 4 用来控制水平范围
        v = (1 - (pitch + abs(fov_down)) / fov) * self.H

        zero_tensor = torch.zeros_like(u)
        W_tensor = torch.ones_like(u) * (self.W - 1)
        H_tensor = torch.ones_like(v) * (self.H - 1)

        u = torch.floor(u)
        u = torch.min(u, W_tensor)
        u = torch.max(u, zero_tensor).long()

        v = torch.floor(v)
        v = torch.min(v, H_tensor)
        v = torch.max(v, zero_tensor).long()

        range_image = torch.full((5, self.H, self.W), 0, dtype=torch.float32).to(device)
        range_image[0][v, u] = depth
        range_image[1][v, u] = points[:, 3]
        range_image[2][v, u] = points[:, 0]
        range_image[3][v, u] = points[:, 1]
        range_image[4][v, u] = points[:, 2]
        return range_image

    def range_to_lidar_gpu(self, range_img):
        device = range_img.device
        self.uv = self.uv.to(device)
        lidar_out = torch.zeros((12, self.H, self.W)).to(device)
        lidar_out[0] = range_img[0] * torch.cos(self.uv[0]) * torch.cos(self.uv[1])
        lidar_out[1] = range_img[0] * torch.cos(self.uv[0]) * torch.sin(self.uv[1]) * (-1)
        lidar_out[2] = range_img[0] * torch.sin(self.uv[0])
        lidar_out[3:] = range_img[1:]
        lidar_out = lidar_out.permute((2, 1, 0)).reshape([-1, 12])
        lidar_out = lidar_out[torch.where(lidar_out[:, 0] != 0)]
        return lidar_out

class Attention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Attention, self).__init__()
        self.q_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        # self.range_v = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.channel_back = nn.Sequential(
            nn.Conv2d(out_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, range, img):
        batch_size, C, width, height = range.size()
        query_range = self.q_conv(range).view(batch_size, -1, width * height).permute(0, 2, 1)
        key_img = self.k_conv(img).view(batch_size, -1, width * height)
        correlation = torch.bmm(query_range, key_img)
        attention = self.softmax(correlation)

        value_img = self.v_conv(img).view(batch_size, -1, width * height)
        # value_range = self.range_v(range).view(batch_size, -1, width * height)

        out = torch.bmm(value_img, attention.permute(0, 2, 1))
        # out_range = torch.bmm(value_range, attention.permute(0, 2, 1))
        out = out.view(batch_size, C//2, width, height)
        # out_range = out_range.view(batch_size, C//2, width, height)
        # out = self.gamma * out
        # out = self.channel_back(out + out_range)
        out = self.channel_back(out)
        out = img + out
        return out

class Fusion(nn.Module):
    def __init__(self, range_in_channel, img_in_channel):
        super(Fusion, self).__init__()
        self.range_conv1 = nn.Sequential(
            nn.Conv2d(range_in_channel, 16, kernel_size=3, padding=2, stride=1, dilation=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2))
        )
        self.img_conv1 = nn.Sequential(
            nn.Conv2d(img_in_channel, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((32, 512))
        )
        self.attention1 = Attention(16, 8)

        self.range_conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=2, stride=1, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2))
        )
        self.img_conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2))
        )
        self.attention2 = Attention(32, 16)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

    def forward(self, range, img):
        img = img[:, :, (img.shape[2]) // 3:, :]
        # img = F.interpolate(img, (48, 512))
        x1 = self.range_conv1(range)    #16x24x256
        y1 = self.img_conv1(img)        #16x24x256
        y_att = self.attention1(x1, y1) #16x24x256

        x2 = self.range_conv2(x1)       #32x12x128
        y2 = self.img_conv2(y_att)      #32x12x128
        y_att2 = self.attention2(x2, y2)#32x12x128

        y_up1 = self.up1(y_att2)        #16x24x256
        y_up2 = self.up2(y_up1 + y1)    #8x48x512
        return y_up2