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
        self.img_encoder = ImgFeatExtractor()
        self.refmap_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.ref_to_cls = nn.Sequential(
            nn.Conv2d(64, 18, kernel_size=1),
            nn.BatchNorm2d(18),
            nn.ReLU()
        )

    def extract_feat(self, points, img_metas):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        #提取出反射率信息
        voxel_features, reflection = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1

        #对反射率地图做scatter
        ref_map = self.middle_encoder(reflection, coors, batch_size)
        ref_map = self.refmap_encoder(ref_map)

        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x, ref_map

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
        x, ref_map = self.extract_feat(points, img_metas)
        #img特征提取
        img_feat = self.img_encoder(img)
        img_feat = self.bbox_head([img_feat])

        outs = self.bbox_head(x)
        #从反射率地图中求cls score
        ref_cls_score = self.ref_to_cls(ref_map)

        #用img求结果与point结果相加
        for i in range(len(outs)):
            outs[i][0] += img_feat[i][0]
        #用反射率加权
        outs[0][0] *= (ref_cls_score * 2 + 1)

        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        x, ref_map = self.extract_feat(points, img_metas)

        img_feat = self.img_encoder(imgs)
        img_feat = self.bbox_head([img_feat])

        outs = self.bbox_head(x)
        #从反射率地图中求cls score
        ref_cls_score = self.ref_to_cls(ref_map)

        for i in range(len(outs)):
            outs[i][0] += img_feat[i][0]
        #用反射率加权
        outs[0][0] *= (ref_cls_score * 2 + 1)

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
