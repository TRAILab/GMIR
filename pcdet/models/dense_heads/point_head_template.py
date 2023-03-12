import torch
import torch.nn as nn
import torch.nn.functional as F

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, loss_utils


class PointHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class

        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_type = losses_cfg.get('LOSS_REG', None)
        if reg_loss_type == 'smooth-l1':
            self.reg_loss_func = F.smooth_l1_loss
        elif reg_loss_type == 'l1':
            self.reg_loss_func = F.l1_loss
        elif reg_loss_type == 'WeightedSmoothL1Loss':
            self.reg_loss_func = loss_utils.WeightedSmoothL1Loss(
                code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None)
            )
        else:
            self.reg_loss_func = F.smooth_l1_loss

    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def assign_stack_targets(self, points, gt_boxes, extend_gt_boxes=None,
                             ret_box_labels=False, ret_part_labels=False,
                             set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8) B= batch size, M=number of boxes in each pc
            (one PC will have all M boxes while other pc will have less than M and the rest of the entries will be zero)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels:
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored pts that are just outside gt box but within extended gt box, 1:Car, 2:Pedestrian, 3:Cyclist
            point_box_labels: (N1 + N2 + N3 + ..., code_size=8) groundtruth box residuals [xt, yt, zt, log(dx_gt/dx_mean_anchor), log(dy_gt/dy_anchor), log(dz_gt/dz_anchor), cos(r_gt), sin(r_gt) ]

            Each foreground point is assumed to be the center of an anchor whose class belongs to the fg point gt class.
            point_box_labels is 1x8 zeros for all background pts and for foreground pts it contains the residual 
            between the anchor centered at the fg pt and the gt box that fg pt lies within. For localization residual definition, see papers: SECOND or pointpillars
        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0] #(N)
        point_cls_labels = points.new_zeros(points.shape[0]).long() #(N)
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None #(N,8)
        point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None #(N,3)
        for k in range(batch_size): # for each pc in batch
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4] # Extract points of batch id k (N1, xyz)

            # Assign point_cls_labels
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum()) #(N1)
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0) # (N1) assigns box ids 0,...,M for each point and -1 for background 
            box_fg_flag = (box_idxs_of_pts >= 0) #True for foreground points
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0) # flag points that are just outside the gt boxes and within the extended gt boxes
                point_cls_labels_single[ignore_flag] = -1 #ignore class labels for them
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]] #(fg_flag.sum(), 8) gt box for every fg point
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single

            # Assign point_box_labels
            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                ) # (num fg pts, 8) localization residuals for each fg point assuming an anchor centered on each pt: [(x_gt_box_center-x_fg_pt/diagonal of anchor), ... ]
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single

            if ret_part_labels:
                point_part_labels_single = point_part_labels.new_zeros((bs_mask.sum(), 3))
                transformed_points = points_single[fg_flag] - gt_box_of_fg_points[:, 0:3]
                transformed_points = common_utils.rotate_points_along_z(
                    transformed_points.view(-1, 1, 3), -gt_box_of_fg_points[:, 6]
                ).view(-1, 3)
                offset = torch.tensor([0.5, 0.5, 0.5]).view(1, 3).type_as(transformed_points)
                point_part_labels_single[fg_flag] = (transformed_points / gt_box_of_fg_points[:, 3:6]) + offset
                point_part_labels[bs_mask] = point_part_labels_single

        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'point_part_labels': point_part_labels
        }
        return targets_dict

    def get_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1) # (16384x2) 0: background, -1: around gtbbox ignored, car:1, ped:2, cyclist: 3
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class) #(16384x2, 3) predicted class scores for each point

        positives = (point_cls_labels > 0) # (num points: 16382 x 2): true for gt object pts, false for background and ignored pts
        negative_cls_weights = (point_cls_labels == 0) * 1.0 # (16382 x 2): 1 for gt background pts, 0 otherwise
        cls_weights = (negative_cls_weights + 1.0 * positives).float() #  (16382 x 2):1 for object and background pt, 0 for ignored pts i.e. around gt box, we don't include loss on ignored pts
        pos_normalizer = positives.sum(dim=0).float() # total num gt object pts
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)  #  (16382 x 2):1/(num gt obj pts) for object and background pt, 0 for ignored pts i.e. around gt box, we don't include loss on ignored pts

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1) # (16384 x 2, 3+1=4)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0) #Create one hot vector of size (16384x2, 4) ->(point_cls_labels * (point_cls_labels >= 0).long()) gives (16384x2) vector which is 0: background and ignored points, 1: Car, 2: Ped, 3: Cyc
        one_hot_targets = one_hot_targets[..., 1:] # only take one hot vectors for car, ped, cyc (N=16384 x 2, 3), ignore background column
        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights) #(N=16384 x 2, 3)
        point_loss_cls = cls_loss_src.sum() # (1): sum all elements

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'point_loss_cls': point_loss_cls.item(),
            'point_pos_num': pos_normalizer.item()
        })

        # # Get point classification losses for car, ped, cyc separately
        # tb_dict.update({
        #     'point_loss_cls_car': cls_loss_src[:,0].sum().item(),
        #     'point_loss_cls_pedestrian': cls_loss_src[:,1].sum().item(),
        #     'point_loss_cls_cyclist': cls_loss_src[:,2].sum().item()
        # })
        return point_loss_cls, tb_dict

    def get_part_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        pos_normalizer = max(1, (pos_mask > 0).sum().item())
        point_part_labels = self.forward_ret_dict['point_part_labels']
        point_part_preds = self.forward_ret_dict['point_part_preds']
        point_loss_part = F.binary_cross_entropy(torch.sigmoid(point_part_preds), point_part_labels, reduction='none')
        point_loss_part = (point_loss_part.sum(dim=-1) * pos_mask.float()).sum() / (3 * pos_normalizer)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_part = point_loss_part * loss_weights_dict['point_part_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_part': point_loss_part.item()})
        return point_loss_part, tb_dict

    def get_box_layer_loss(self, tb_dict=None):
        # point_cls_preds: (N1 + N2 + N3 + ..., num_class=3) predicted class scores for each point (same as batch_cls_preds)
        # point_box_preds: (N1 + N2 + N3 + ..., box_code_size=8) Predicted box residuals for each pt (xt,yt,zt,dxt,dyt,dzt, cos r, sin r)
        # point_cls_labels: (N1 + N2 + N3 + ...), long type, gt class labels for each point  0:background, -1:ignored pts that are just outside gt box but within extended gt box, 1:Car, 2:Pedestrian, 3:Cyclist
        # point_box_labels: (N1 + N2 + N3 + ..., code_size=8) groundtruth box residuals [xt, yt, zt, log(dx_gt/dx_mean_anchor), log(dy_gt/dy_anchor), log(dz_gt/dz_anchor), cos(r_gt), sin(r_gt) ]

        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0 # (num points: 16382 x 2): true for gt object pts
        point_box_labels = self.forward_ret_dict['point_box_labels']
        point_box_preds = self.forward_ret_dict['point_box_preds']

        reg_weights = pos_mask.float() # (16384x2): 1 for gt object pt, 0 else
        pos_normalizer = pos_mask.sum().float() #num gt object pts
        reg_weights /= torch.clamp(pos_normalizer, min=1.0) # (16384x2): (1/num gt obj pts) for gt object pt, 0 else

        point_loss_box_src = self.reg_loss_func(
            point_box_preds[None, ...], point_box_labels[None, ...], weights=reg_weights[None, ...]
        )# (1, 16384x2, 8)
        point_loss_box = point_loss_box_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_box': point_loss_box.item()})

        # # Get point-wise box regression losses for car, ped, cyc separately
        # gt_car_pts_mask =  self.forward_ret_dict['point_cls_labels'] == 1
        # gt_pedestrian_pts_mask = self.forward_ret_dict['point_cls_labels'] == 2
        # gt_cyclist_pts_mask = self.forward_ret_dict['point_cls_labels'] == 3
        # point_loss_box_car = point_loss_box_src[:, gt_car_pts_mask, :].sum()
        # point_loss_box_pedestrian = point_loss_box_src[:, gt_pedestrian_pts_mask, :].sum()
        # point_loss_box_cyclist = point_loss_box_src[:, gt_cyclist_pts_mask, :].sum()
        # tb_dict.update({
        #     'point_loss_box_car': point_loss_box_car.item(),
        #     'point_loss_box_pedestrian': point_loss_box_pedestrian.item(),
        #     'point_loss_box_cyclist': point_loss_box_cyclist.item()
        # })
        return point_loss_box, tb_dict

    def generate_predicted_boxes(self, points, point_cls_preds, point_box_preds):
        """
        Args:
            points: (N, 3) xyz
            point_cls_preds: (N, num_class) class scores (car, ped, cyclist) for each pt i.e. 1x3 for each pt
            point_box_preds: (N, box_code_size=8) predicted box residuals for each pt
        Returns:
            point_cls_preds: (N, num_class) same as input
            point_box_preds: (N, 7) predicted box for each pt [x,y,z,dx,dy,dz,r] extracted from the predicted box residuals

        """
        _, pred_classes = point_cls_preds.max(dim=-1) # (16384 x 2) 0: car, 1: ped, 2: cyclist
        point_box_preds = self.box_coder.decode_torch(point_box_preds, points, pred_classes + 1)

        return point_cls_preds, point_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
