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
        # 融合Img信息部分代码
        # self.img_encoder = ImgFeatExtractor()
        # self.refmap_encoder = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        # self.ref_to_cls = nn.Sequential(
        #     nn.Conv2d(64, 18, kernel_size=1),
        #     nn.BatchNorm2d(18),
        #     nn.ReLU()
        # )

        # 转换RangeImage部分相关代码
        self.H = 48
        self.W = 512
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

        self.range_encoder = RangeEncoder(5, 64, use_img=True)

    def extract_feat(self, points, img_metas):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        #提取出反射率信息
        voxel_features, reflection = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1

        #对反射率地图做scatter
        # ref_map = self.middle_encoder(reflection, coors, batch_size)
        # ref_map = self.refmap_encoder(ref_map)

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
        range_feat = self.range_encoder(rangeImage, img)
        range_ori = torch.cat((rangeImage[:, 0:2], range_feat), dim=1)
        pts_with_range = []
        for i in range(batchsize):
            pts_with_range.append(self.range_to_lidar_gpu(range_ori[i].squeeze(0)))

        x = self.extract_feat(pts_with_range, img_metas)
        #img特征提取
        # img_feat = self.img_encoder(img)
        # img_feat = self.bbox_head([img_feat])

        outs = self.bbox_head(x)
        #从反射率地图中求cls score
        # ref_cls_score = self.ref_to_cls(ref_map)

        #用img求结果与point结果相加
        # for i in range(len(outs)):
        #     outs[i][0] += img_feat[i][0]
        #用反射率加权
        # outs[0][0] *= (ref_cls_score * 2 + 1)

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
        range_feat = self.range_encoder(rangeImage, imgs)
        range_ori = torch.cat((rangeImage[:, 0:2], range_feat), dim=1)
        pts_with_range = []
        for i in range(batchsize):
            pts_with_range.append(self.range_to_lidar_gpu(range_ori[i].squeeze(0)))

        x = self.extract_feat(pts_with_range, img_metas)

        # img特征提取
        # img_feat = self.img_encoder(imgs)
        # img_feat = self.bbox_head([img_feat])

        outs = self.bbox_head(x)
        #从反射率地图中求cls score
        # ref_cls_score = self.ref_to_cls(ref_map)

        # for i in range(len(outs)):
        #     outs[i][0] += img_feat[i][0]
        #用反射率加权
        # outs[0][0] *= (ref_cls_score * 2 + 1)

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
        lidar_out = torch.zeros((8, self.H, self.W)).to(device)
        lidar_out[0] = range_img[0] * torch.cos(self.uv[0]) * torch.cos(self.uv[1])
        lidar_out[1] = range_img[0] * torch.cos(self.uv[0]) * torch.sin(self.uv[1]) * (-1)
        lidar_out[2] = range_img[0] * torch.sin(self.uv[0])
        lidar_out[3:] = range_img[1:]
        lidar_out = lidar_out.permute((2, 1, 0)).reshape([-1, 8])
        lidar_out = lidar_out[torch.where(lidar_out[:, 0] != 0)]
        return lidar_out

class ImgFeatExtractor(nn.Module):
    def __init__(self):
        super(ImgFeatExtractor, self).__init__()
        self.extract_img_feat = torchvision.models.resnet50(pretrained=True).eval().cuda()
        self.argpool = nn.AdaptiveAvgPool2d((496, 432))
        self.Linear = nn.Sequential(
            nn.Linear(512, 384),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.deblocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2), bias=False),
                nn.BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(512, 128, kernel_size=(4, 4), stride=(4, 4), bias=False),
                nn.BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(1024, 128, kernel_size=(8, 8), stride=(8, 8), bias=False),
                nn.BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            )
        ])

    def forward(self, img):
        out = self.argpool(torch.transpose(img, 2, 3))
        out = self.extract_img_feat.conv1(out)
        out = self.extract_img_feat.bn1(out)
        out = self.extract_img_feat.relu(out)
        out = self.extract_img_feat.maxpool(out)

        out1 = self.extract_img_feat.layer1(out)
        out2 = self.extract_img_feat.layer2(out1)
        out3 = self.extract_img_feat.layer3(out2)

        out1 = self.deblocks[0](out1)
        out2 = self.deblocks[1](out2)
        out3 = self.deblocks[2](out3)
        img_feat = torch.cat((out1, out2, out3), dim=1)
        return img_feat

class RangeEncoder(nn.Module):
    def __init__(self, in_channel, out_channel, use_img=False):
        super(RangeEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        self.down4 = Down(256, 256)

        self.up1 = Up(256, 256, 128)
        self.up2 = Up(128, 256, 128)
        self.up3 = Up(128, 128, 64)
        self.up4 = Up(64, 64, out_channel)
        if use_img:
            self.img_conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            self.img_down1 = Down(64, 128)
            self.img_down2 = Down(128, 256)
            self.img_down3 = Down(256, 256)
            self.img_down4 = Down(256, 256)
            self.img_lidar_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            self.img_lidar_conv2 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            self.img_lidar_conv3 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            self.img_lidar_conv4 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, img=None):
        x = self.conv(x)

        if img is not None:
            img = img[:, :, (img.shape[2]) // 3:, :]
            img = F.interpolate(img, (48, 512))
            img = self.img_conv(img)

            img1 = self.img_down1(img)
            x_d1 = self.down1(x)
            x_d1 = torch.cat((x_d1, img1), dim=1)
            x_d1 = self.img_lidar_conv1(x_d1)

            img2 = self.img_down2(img1)
            x_d2 = self.down2(x_d1)
            x_d2 = torch.cat((x_d2, img2), dim=1)
            x_d2 = self.img_lidar_conv2(x_d2)

            img3 = self.img_down3(img2)
            x_d3 = self.down3(x_d2)
            x_d3 = torch.cat((x_d3, img3), dim=1)
            x_d3 = self.img_lidar_conv3(x_d3)

            img4 = self.img_down4(img3)
            x_d4 = self.down4(x_d3)
            x_d4 = torch.cat((x_d4, img4), dim=1)
            x_d4 = self.img_lidar_conv4(x_d4)

        else:
            x_d1 = self.down1(x)
            x_d2 = self.down2(x_d1)
            x_d3 = self.down3(x_d2)
            x_d4 = self.down4(x_d3)

        x_u1 = self.up1(x_d4, x_d3)
        x_u2 = self.up2(x_u1, x_d2)
        x_u3 = self.up3(x_u2, x_d1)
        x_u4 = self.up4(x_u3, x)

        x_out = self.down_sample(x_u4)
        return x_out


class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        self.dilatedconv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=(3, 3), padding=2, stride=1, dilation=2),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        self.dilatedconv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=(3, 3), padding=2, stride=1, dilation=2),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        self.dilatedconv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=(3, 3), padding=2, stride=1, dilation=2),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        self.down = nn.Sequential(
            nn.Conv2d(in_channels=in_channel * 3, out_channels=out_channel, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.MaxPool2d((2, 2))
        )

    def forward(self, x):
        x = self.conv(x)
        x1 = self.dilatedconv1(x)
        x2 = self.dilatedconv2(x1)
        x3 = self.dilatedconv3(x2)
        x_all = torch.cat((x1, x2, x3), dim=1)
        x_out = self.down(x_all)
        return x_out

class Up(nn.Module):
    def __init__(self, in_channel, res_channel, out_channel):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        mid_channels = (out_channel + res_channel) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=(out_channel + res_channel), out_channels=mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv(x)
        x = nn.Dropout2d(0.25)(x)
        return x
