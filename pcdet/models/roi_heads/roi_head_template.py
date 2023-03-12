import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import box_coder_utils, common_utils, loss_utils
from ..model_utils.model_nms_utils import class_agnostic_nms
from .target_assigner.proposal_target_layer import ProposalTargetLayer


class RoIHeadTemplate(nn.Module):
    def __init__(self, num_class, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {})
        )
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None

    def build_losses(self, losses_cfg):
        self.add_module(
            'reg_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    @torch.no_grad()
    def proposal_layer(self, batch_dict, nms_config):
        """
        Shortlist predicted boxes from 16384 boxes i.e. (each point has a predicted box) to 512 boxes
        First select 9000 highest scoring boxes and then do nms on them.
        # NMS: 
        # 1. select next highest-scoring box, 
        # 2. eliminate lower-scoring boxes with IoU > threshold=0.8 (NMS_THRESH), 
        # 3. If any boxes remain go to step 1.
        
        Nms will give around 4000 boxes
        Select 512 boxes top scoring boxes from these 
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1): For Pointrcnn (N=16384, 3)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C): For Pointrcnn  (N=16384, 7)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B=2, num_rois=512, 7+C) predicted 512 boxes/points
                roi_scores: (B, num_rois) each of the 512 shortlisted point's predicted class score i.e. max score in batch_cls_preds
                roi_labels: (B, num_rois) each of the 512 shortlisted points/boxes's predicted class label [1, .., numclasses,..] 1:car, 2:pedestrian, 3:cyclist

        """
        if batch_dict.get('rois', None) is not None:
            return batch_dict
            
        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_box_preds']
        batch_cls_preds = batch_dict['batch_cls_preds'] # (N, 3)
        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1])) #(2,512,7) each pc will have 512 rois i.e. predicted boxes
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE)) #(2, 512)
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long) #(2, 512)

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask] # box preds for one pc (16382, 7)
            cls_preds = batch_cls_preds[batch_mask] # class scores preds for one pc (16382, 3)

            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1) # max score and label for each point 0: car, 1: ped, 2: cyclist

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                # shortlist predicted boxes from 16384 to 512: 
                # First select 9000 highest scoring boxes and then do nms on them. Nms will give around 4000 boxes
                # Select 512 boxes top scoring boxes from these
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )

            rois[index, :len(selected), :] = box_preds[selected] # (512, 7) box predictions for each shortlisted point
            roi_scores[index, :len(selected)] = cur_roi_scores[selected] # max class score for each shortlisted point
            roi_labels[index, :len(selected)] = cur_roi_labels[selected] # class label for each shortlisted point

        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        batch_dict.pop('batch_index', None)
        return batch_dict

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']

        # Proposal target layer fucntion:
        # sample_rois_for_rcnn:
        # 1. Match 512 predicted boxes with the 41 gt boxes by computing iou3D matrix of size (512, 41) and taking max over each row
        # 2. Subsample/shortlist 512 predicted boxes to 128 boxes depending on their matched iou3D score
        # rois: (2, 128, 7), roi_labels: (2, 128), roi_scores: (2, 128), gt_iou_of_rois: (2, 128), gt_of_rois: (2, 128, 8)
        
        # regression valid mask: 1 for predicted boxes that have high iou3D with their matched gt boxes 
        # i.e. reg_valid_mask = if predicted boxes with iou3d > 0.55, then 1 else 0
        # i.e. possible foreground boxes

        # rcnn_cls_labels: (2, 128)
            # rcnn_cls_labels: is 1 if predicted boxes iou3d > 0.6 = CLS_FG_THRESH, i.e. for possible foreground boxes
            #                      0 if predicted boxes iou3d < 0.45 = CLS_BG_THRESH, i.e. for possible background boxes
            #                     -1 if 0.45 < predicted boxes iou3d < 0.6 (ignore these), i.e. for hard objects
        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict)

        # In the following lines: targets_dict['gt_of_rois'][:,:,0:3] is made to contain (gt_center_xyz - pred_center_xyz) in predicted box frame
        #  targets_dict['gt_of_rois'][:,:,6] contains heading error i.e. gt box heading - predicted box heading  and this is made to lie in (-90, 90 deg) range
        rois = targets_dict['rois']  # (B, N, 7 + C) e.g. (2, 128, 7)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1) e.g. (2, 128, 8)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi) # transform predicted box heading from (-pi, pi) to (0, 2pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center # gt_of_rois[:,:,0:3] now stores a vector from predicted box center to gt_center, rep in lidar frame
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry # gt_of_rois[:,:,6] stores gt_box heading - predicted box heading

        # transfer LiDAR coords to local coords i.e. (gt_center - predicted box center)_in_lidar_frame to (gt_center - predicted box center)_in_predicted_box_frame
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # transform to 0 ~ 2pi. This is the heading error i.e. gt-predicted heading
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5) # if  90 deg < heading error < 270 deg
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi) heading_error[opposite_flag] is from 91 to 269 deg so we transform this to lie from 271 deg, ..., 359 deg, 0 deg,... to 89 deg. Just flip ground truth box x and y i.e. rotate by 180 deg (which represents the same gt box)
        # all heading errors (including opposite flagged) are now between 0 and 90 deg and 270 and 359 deg.
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # transforms heading errors to lie in (-pi/2, pi/2) e.g. 270 deg - 360 deg = -90 deg, 359 - 360 = -1 deg
        # all heading errors between gt and predicted box are now between (-pi/2, pi/2) 
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        # regression valid mask: 1 for predicted boxes that have high iou3D with their matched gt boxes 
        # i.e. reg_valid_mask = if predicted boxes with iou3d > 0.55, then 1 else 0
        # i.e. possible foreground boxes

        # 'gt_of_rois_src': (2, 128, 8) matched gt box with the predicted rois 
        # 'gt_of_rois'[:,:,0:3] is made to contain (gt_center_xyz - pred_center_xyz) in predicted box frame
        # 'gt_of_rois'[:,:,6] contains heading error i.e. gt box heading - predicted box heading  and this is made to lie in (-90, 90 deg) range

        # rcnn_reg: (256, 7) : predicted box offsets (from predicted rois to gt boxes?) for 256 rois

        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size # 7
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1) #(2 x 128)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size] #(2 , 128, 7)
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size) #(2 x 128, 7)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C) #(2 x 128, 7)
        roi_boxes3d = forward_ret_dict['rois'] #(2, 128, 7) predicted boxes i.e. predicted rois after first stage 
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0] # 256 = 2x128

        fg_mask = (reg_valid_mask > 0) # (256) true if predicted boxes with iou3d > 0.55
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size) # (256, 7)
            rois_anchor[:, 0:3] = 0 # since gt_boxes3d_ct already contains the offset gt_center_xyz - predicted box center
            rois_anchor[:, 6] = 0 # since gt_boxes3d_ct already contains the offset gt_heading - predicted box heading 
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            ) #reg_targets: (256, 7): gt residual i.e. gt_box - roi
            #gt_boxes3d_ct.view(rcnn_batch_size, code_size) -> (256, 7)

            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # rcnn_loss_reg: (1, 256, 7), both inputs also have shape (1, 256, 7)
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1) # (1): choose rcnn_loss_reg for possible fg boxes, sum them and divide by num fg boxes
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                # TODO: NEED to BE CHECK
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask] # (64x2, 7) predicted box offsets for fg objects after stage 2 i.e. from rcnn
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask] # (64x2, 7) predictded boxes for fg objects after stage 1 i.e. predicted rois

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size) # (1, 128, 7)
                batch_anchors = fg_roi_boxes3d.clone().detach() # (1, 128, 7)
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1) #(128)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3) #(128, 3)
                batch_anchors[:, :, 0:3] = 0 
                # fg_rcnn_reg contains predictions for the offset = actual box - predicted roi stage 1
                # Setting batch_anchors[:, :, 0:3] = 0 means after decoding, rcnn_boxes3d[:,:, 0:3] will still contain the offset i.e. gt_center - predicted roi center in predicted roi frame
                # b/c fg_rcnn_reg[:,:,0:3] (predicted offset) = rcnn_boxes3d[:,:, 0:3] (xg, yg, zg we want to find) - batch_anchors[:,:,0:3] (predicted roi)
                # get predicted rcnn boxes (xg) from the rcnn predicted box offsets (xt) and predicted roi stage 1 as anchors (xa)
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size) # both inputs (1, 128, 7) and output (128, 7)

                # To change rcnn_boxes3d[:, :, 0:3] from (gt_box_center - pred_roi_center) in pred_roi_frame to (...) in lidar frame  
                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1) # now rcnn_boxes3d = (gt_box_center - pred_roi_center) in lidar frame
                rcnn_boxes3d[:, 0:3] += roi_xyz # now rcnn_boxes3d[: 0:3] = gt_box_center in lidar frame

                # rcnn_boxes3d: (64x2, 7) predicted rcnn boxes for fg objects
                # gt_of_rois_src[fg_mask]: (64x2, 7) gt boxes matched with predicted rois for fg objects
                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                ) # (128)
                loss_corner = loss_corner.mean() #(1) averaged over all boxes
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict #rcnn_loss_reg = rcnn_reg_loss + rcnn_corner_loss

    def get_box_cls_layer_loss(self, forward_ret_dict):
        
        # rcnn_cls: (2 pcs x 128 rois = 256 rois, 1) : predicted objectness scores for 256 rois 
        # rcnn_reg: (256, 7) : predicted box offsets (from predicted rois to gt boxes?) for 256 rois

        # rcnn_cls_labels: (2, 128)
        # rcnn_cls_labels: is 1 if predicted boxes iou3d > 0.6 = CLS_FG_THRESH, i.e. for possible foreground boxes i.e. labelled as foreground box
        #                      0 if predicted boxes iou3d < 0.45 = CLS_BG_THRESH, i.e. for possible background boxes i.e. labelled as background box
        #                     -1 if 0.45 < predicted boxes iou3d < 0.6 (ignore these), i.e. for hard objects
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none') #(256 rois)
            cls_valid_mask = (rcnn_cls_labels >= 0).float() #(256) 1 for possible foreground objects, 0 otherwise
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0) # (1): only choose losses for possible foreground boxes (ignore background boxes), sum these losses and divide by num possible fg boxes
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item() # rcnn_loss_cls + reg + corner
        return rcnn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7): (2, 100, 7) From 1st stage box predictions
            cls_preds: (BN, num_class): (200, 1) From rcnn output: objectness score of rois
            box_preds: (BN, code_size): (200, 7) From rcnn output: box predicted offsets (from predicted rois to gt boxes) for 200 rois

        Returns:
            batch_cls_preds: (2, 100, 1) From rcnn output: objectness score of rois. Same as cls preds
            batch_box_preds: (2, 100, 7) final box prediction extracted from rcnn predicted box offset and predicted roi from 1st stage as anchors

        """
        code_size = self.box_coder.code_size #7
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1]) # (2, 100, 1)
        batch_box_preds = box_preds.view(batch_size, -1, code_size) # (2, 100, 7)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        # box_preds contains predictions for the offset = actual box - predicted roi stage 1
        # Setting local_rois[:, :, 0:3] = 0 means after decoding, batch_box_preds[:,:, 0:3] will still contain the offset i.e. gt_center - predicted roi center in predicted roi frame
        # b/c box_preds[:,:,0:3] (predicted offset) = batch_box_preds[:,:, 0:3] (xg, yg, zg we want to find) - local_rois[:,:,0:3] (predicted roi)
        # get predicted rcnn boxes (xg) from the rcnn predicted box offsets (xt) and predicted roi stage 1 as anchors (xa)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        # To change batch_box_preds[:, :, 0:3] from (gt_box_center - pred_roi_center) in pred_roi_frame to (...) in lidar frame
        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1) # now batch_box_preds = (gt_box_center - pred_roi_center) in lidar frame
        batch_box_preds[:, 0:3] += roi_xyz # now batch_box_preds[: 0:3] = gt_box_center in lidar frame
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds
