import torch

from ...utils import box_coder_utils, box_utils
from .point_head_template import PointHeadTemplate


class PointHeadBox(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PointRCNN.
    Reference Paper: https://arxiv.org/abs/1812.04244
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training
        #cin = 128, 2 hidden layers = [[linear = 256 (bias=False), batchnorm1d, relu], 
        #                       [linear=256 (bias=False), batchnorm1, relu],
        #                        [linear=3 (bias=true)]] 
        #cout = 3 = num_class
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        
        #cin = 128, 2 hidden layers = [[linear = 256 (bias=False), batchnorm1d, relu], 
        #                       [linear=256 (bias=False), batchnorm1, relu],
        #                        [linear=8 (bias=true)]] 
        #cout = 8 = box code size = residual size [xt, yt, zt, dxt, dyt, dzt, cos(r), sin(r)]
        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=input_channels,
            output_channels=self.box_coder.code_size
        )

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C) N1=number of points in pc 1, N2= number of points in pc2, ...
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, gt class labels for each point  0:background, -1:ignored pts that are just outside gt box but within extended gt box, 1:Car, 2:Pedestrian, 3:Cyclist
            point_box_labels: (N1 + N2 + N3 + ..., code_size=8) groundtruth box residuals [xt, yt, zt, log(dx_gt/dx_mean_anchor), log(dy_gt/dy_anchor), log(dz_gt/dz_anchor), cos(r_gt), sin(r_gt) ]

            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])

        # point_cls_labels: For each pt: Use gt boxes to assign gt class labels to all points i.e. 0 for background, -1 for pts just around the gt boxes, 1:car, 2: pedestrian etc
        # point_box_labels: For each foreground point: compute groundtruth box residuals between anchors centered at each foregorund point and the gt box
        # we use gt boxes to assign labels to each point and assume an anchor (at the center of fg pts) of mean size == to the mean size of the gt class that fg pt lies in
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=True
        )
        
        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss() # 1 number
        point_loss_box, tb_dict_2 = self.get_box_layer_loss() # 1 number

        point_loss = point_loss_cls + point_loss_box
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]

                point_labels (optional): (N1 + N2 + N3 + ...)
                point_features_before_fusion (optional): (N1 + N2 + N3 + ..., C)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1): predicted objectness confidence/probability of each pt (max class score converted into prob)
                batch_cls_preds: (N1 + N2 + N3 + ..., num_class=3): predicted class scores for each point
                batch_box_preds: (N1 + N2 + N3 + ..., 7): predicted boxes for each point (already extracted from their predicted residuals)
                batch_index: (N1 + N2 + N3 + ...): batch id for each point
                point_part_offset: (N1 + N2 + N3 + ..., 3)
            forward_ret_dict:
                point_cls_preds: (N1 + N2 + N3 + ..., num_class=3) predicted class scores for each point (same as batch_cls_preds)
                point_box_preds: (N1 + N2 + N3 + ..., box_code_size=8) Predicted box residuals for each pt (xt,yt,zt,dxt,dyt,dzt, cos r, sin r)
                point_cls_labels: (N1 + N2 + N3 + ...), long type, gt class labels for each point  0:background, -1:ignored pts that are just outside gt box but within extended gt box, 1:Car, 2:Pedestrian, 3:Cyclist
                point_box_labels: (N1 + N2 + N3 + ..., code_size=8) groundtruth box residuals [xt, yt, zt, log(dx_gt/dx_mean_anchor), log(dy_gt/dy_anchor), log(dz_gt/dz_anchor), cos(r_gt), sin(r_gt) ]

        """
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features'] # (total_points in batch = N =16384x2, 128) 
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class=3) Predict class scores (car, pedestrian, cyclist) for every point
        point_box_preds = self.box_layers(point_features)  # (total_points, box_code_size=8) Predicted box residuals for each pt (xt,yt,zt,dxt,dyt,dzt, cos r, sin r)

        point_cls_preds_max, _ = point_cls_preds.max(dim=-1) #(N, 3) -> max score for each point (N)
        batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max) # (N) turn max scores in probabilities i.e. objectness confidence for each point

        ret_dict = {'point_cls_preds': point_cls_preds,
                    'point_box_preds': point_box_preds}
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_box_labels'] = targets_dict['point_box_labels']

        if not self.training or self.predict_boxes_when_training: # for testing or for rcnn head training we need box predictions from 1st stage as anchors in rcnn stage
            # Extract predicted boxes from predicted box residuals
            # generate_predicted_boxes returns:
            # point_cls_preds: (N, num_class=3) same as input
            # point_box_preds: (N, 7) predicted box for each pt [x,y,z,dx,dy,dz,r] extracted from the predicted box residuals
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, point_box_preds=point_box_preds
            )
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
            batch_dict['cls_preds_normalized'] = False

        self.forward_ret_dict = ret_dict

        return batch_dict
