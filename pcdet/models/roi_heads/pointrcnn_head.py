import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.roipoint_pool3d import roipoint_pool3d_utils
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate


class PointRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        use_bn = self.model_cfg.USE_BN
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels #128

        self.num_prefix_channels = 3 + 2  # xyz + point_scores + point_depth
        xyz_mlps = [self.num_prefix_channels] + self.model_cfg.XYZ_UP_LAYER
        shared_mlps = []
        for k in range(len(xyz_mlps) - 1):
            shared_mlps.append(nn.Conv2d(xyz_mlps[k], xyz_mlps[k + 1], kernel_size=1, bias=not use_bn))
            if use_bn:
                shared_mlps.append(nn.BatchNorm2d(xyz_mlps[k + 1]))
            shared_mlps.append(nn.ReLU())
        self.xyz_up_layer = nn.Sequential(*shared_mlps)

        c_out = self.model_cfg.XYZ_UP_LAYER[-1]
        self.merge_down_layer = nn.Sequential(
            nn.Conv2d(c_out * 2, c_out, kernel_size=1, bias=not use_bn),
            *[nn.BatchNorm2d(c_out), nn.ReLU()] if use_bn else [nn.ReLU()]
        )

        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = [channel_in] + self.model_cfg.SA_CONFIG.MLPS[k]

            npoint = self.model_cfg.SA_CONFIG.NPOINTS[k] if self.model_cfg.SA_CONFIG.NPOINTS[k] != -1 else None
            self.SA_modules.append(
                pointnet2_modules.PointnetSAModule(
                    npoint=npoint,
                    radius=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsample=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=True,
                    bn=use_bn
                )
            )
            channel_in = mlps[-1]
        
        # SA1: sample K = 128 points, group points around each of these 128 points with radius r=0.2 and sample n=16 points with this radius
        # Apply MLP [Cin= 128 + 3 = 131, 128, 128, Cout = 128] (3 layers)
        # SA2: sample K=32 points from the 128 points sampled in SA1, r = 0.4, n=16, MLP: [Cin = Cout of SA1 + 3 xyz = 131,  128, 128, 256]
        # SA3: sample K=-1 points from the 32 points sampled in SA2, r = 100, n=16, MLP: [Cin = Cout of SA1 + 3 xyz = 259,  256, 256, 512]
        self.cls_layers = self.make_fc_layers(
            input_channels=channel_in, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=channel_in,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )

        self.roipoint_pool3d_layer = roipoint_pool3d_utils.RoIPointPool3d(
            num_sampled_points=self.model_cfg.ROI_POINT_POOL.NUM_SAMPLED_POINTS,
            pool_extra_width=self.model_cfg.ROI_POINT_POOL.POOL_EXTRA_WIDTH
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roipool3d_gpu(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C): (2, 128, 7)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C): (16384 x 2, 128)
        Returns:
            pooled_features: (2x128=num rois, 512, 133=[3 = (points (in roi) xyz - roi_center) in roi frame, 1 point cls score, 1 depth, 128 features from pointnet++])
            For empty rois, pooled featrues are filled with zeros
        """
        batch_size = batch_dict['batch_size']
        batch_idx = batch_dict['point_coords'][:, 0]
        point_coords = batch_dict['point_coords'][:, 1:4]
        point_features = batch_dict['point_features']
        rois = batch_dict['rois']  # (B, num_rois=128, 7 + C)
        batch_cnt = point_coords.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert batch_cnt.min() == batch_cnt.max()

        point_scores = batch_dict['point_cls_scores'].detach()
        point_depths = point_coords.norm(dim=1) / self.model_cfg.ROI_POINT_POOL.DEPTH_NORMALIZER - 0.5
        point_features_list = [point_scores[:, None], point_depths[:, None], point_features]
        point_features_all = torch.cat(point_features_list, dim=1) #(16384x2, 130) 
        batch_points = point_coords.view(batch_size, -1, 3) #(2, 16384, 3)
        batch_point_features = point_features_all.view(batch_size, -1, point_features_all.shape[-1]) #(2, 16384, 130)

        with torch.no_grad():
            pooled_features, pooled_empty_flag = self.roipoint_pool3d_layer(
                batch_points, batch_point_features, rois
            )  # pooled_features: (B, num_rois=128, num_sampled_points in each roi=512, 133 = 3 + (C=130)), pooled_empty_flag: (B=2, num_rois=128)

            # canonical transformation
            roi_center = rois[:, :, 0:3]
            pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2) # pooled features[:, :, :, 0:3] contains vector from roi center to points in roi i.e. (points xyz - roi_center) in lidar frame

            pooled_features = pooled_features.view(-1, pooled_features.shape[-2], pooled_features.shape[-1]) # (2x128 num rois, 512 points, 133 feature dim)
            
            # Transform pooled features[:, :, :, 0:3] so that it contains (points xyz - roi_center) in roi frame
            pooled_features[:, :, 0:3] = common_utils.rotate_points_along_z(
                pooled_features[:, :, 0:3], -rois.view(-1, rois.shape[-1])[:, 6]
            )
            pooled_features[pooled_empty_flag.view(-1) > 0] = 0
        return pooled_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:

        Returns:
            For training:
            rcnn_cls: (2 pcs x 128 rois = 256 rois, 1) : predicted objectness scores for 256 rois 
            rcnn_reg: (256, 7) : predicted box offsets (from predicted rois to gt boxes) for 256 rois

            For testing:
            batch_cls_preds: (2, 100, 1) From rcnn output: objectness score of rois
            batch_box_preds: (2, 100, 7) final box prediction extracted from rcnn predicted box offset and predicted roi from 1st stage as anchors

        """
        # batch_dict is updated in this function and returned as target_dict so batch_dict == target_dict
        # Proposal layer: Shortlist predicted boxes from 16384 boxes i.e. (each point has a predicted box) to 512 boxes or 100 boxes for testing
        # 16384 -> select top 9000 scoring boxes -> do NMS which gives 4000 boxes -> select top 512 scoring boxes or top 100 scoring boxes for testing
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            # Proposal target layer fucntion:
            # sample_rois_for_rcnn:
            # 1. Match 512 predicted boxes with the 41 gt boxes by computing iou3D matrix of size (512, 41) and taking max over each row
            # 2. Subsample/shortlist 512 predicted boxes to 128 boxes depending on their matched iou3D score
                    # Threshold 512 boxes based on their matched iou3D score to create three categories to sample from:
                    # 1. fg boxes = boxes (out of 512 boxes) with matched iou3D > 0.55
                    # 2. hard bg boxes = boxes with matched iou3D < 0.55 and > 0.1
                    # 3. easy bg boxes = boxes with matched iou3D <  0.1
                    # Randomly sample 64 boxes from these possible fg boxes
                    # Randomly sample 0.8 * 64 boxes from hard_bg boxes
                    # Randomly sample 0.2 * 64 boxes from easy_bg
            # rois: (2, 128, 7), roi_labels: (2, 128), roi_scores: (2, 128), gt_iou_of_rois: (2, 128), gt_of_rois: (2, 128, 8)
            
            # regression valid mask: (2, 128) 
            # 1 for predicted boxes (rois) that have high iou3D with their matched gt boxes 
            # i.e. reg_valid_mask = if predicted boxes with iou3d > 0.55, then 1 else 0
            # i.e. possible foreground boxes

            # rcnn_cls_labels: (2, 128)
                # rcnn_cls_labels: is 1 if predicted boxes iou3d > 0.6 = CLS_FG_THRESH, i.e. for possible foreground boxes
                #                      0 if predicted boxes iou3d < 0.45 = CLS_BG_THRESH, i.e. for possible background boxes
                #                     -1 if 0.45 < predicted boxes iou3d < 0.6 (ignore these), i.e. for hard objects
            
            # target_dict['gt_of_rois'][:,:,0:3] is made to contain (gt_center_xyz - pred_center_xyz) in predicted box frame
            # targets_dict['gt_of_rois'][:,:,6] contains heading error i.e. gt box heading - predicted box heading  and this is made to lie in (-90, 90 deg) range
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # pooled_features: (2x128=num rois, 512 points per roi, 133 feature dim=[3 = (points (in roi) xyz - roi_center) in roi frame, 1 point cls score, 1 depth, 128 features from pointnet++])
        # For empty rois, pooled featrues are filled with zeros
        pooled_features = self.roipool3d_gpu(batch_dict)  # (total_rois, num_sampled_points, 3 + C) (2x128 rois, 512 points per roi, 133 feature dim), for testing total rois = 2x100

        xyz_input = pooled_features[..., 0:self.num_prefix_channels].transpose(1, 2).unsqueeze(dim=3).contiguous() # (256 rois, 5, 512, 1): 5 features are xyz vector (from roi center to point in roi) in roi frame, point cls score, depth
        xyz_features = self.xyz_up_layer(xyz_input) # input: (B=256, C=5, 512=H, 1=W) -> Conv2d(5,128, kernel size = (1,1)) + Relu -> Cpnv2d(128,128, kernel size = (1,1)) + relu -> output:(B=256, C=128, 512=H, 1=W) 
        point_features = pooled_features[..., self.num_prefix_channels:].transpose(1, 2).unsqueeze(dim=3) #pointnet++ roi pooled features (256=rois, 128=feature dim, 512=points, 1)
        merged_features = torch.cat((xyz_features, point_features), dim=1) # (256 rois, 128 xyz_features + 128 point net features, 512, 1)
        merged_features = self.merge_down_layer(merged_features) # input: (B=256, C=256, H=512, W=1) -> Conv2d(256, 128, kernel size = (1,1)) + relu -> output: (B=256, C=128, H=512, W=1)

        l_xyz, l_features = [pooled_features[..., 0:3].contiguous()], [merged_features.squeeze(dim=3).contiguous()]# lxyz: (256, 512, 3): these are delta xyz b/w points and roi center in roi frame, l_features: (256, 128, 512): these are merged xyz_up + point net++ roi pooled features

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        # Do SetAbstraction within each roi 
        # SA1: sample K = 128 points from each roi containing 512 points, group points around each of these 128 points with radius r=0.2 and sample n=16 points with this radius
        # Apply MLP [Cin= 128 features dim + 3 xyz= 131, 128, 128, Cout = 128] (3 layers)
        # SA2: sample K=32 points from the 128 points sampled in SA1, r = 0.4, n=16, MLP: [Cin = Cout of SA1 + 3 xyz = 131,  128, 128, 256]
        # SA3: sample K=-1 points from the 32 points sampled in SA2, r = 100, n=16, MLP: [Cin = Cout of SA1 + 3 xyz = 259,  256, 256, 512]
        #
        # l_xyz = [input: (256, 512, 3), after SA1: (256, 128, 3), after SA2: (256, 32, 3), after SA3: None]
        # l_features = [input: (256 rois, 128 feature dim, 512 points), after SA1: (256, 128, 128), after SA2: (256 , 256 , 32), after SA3: (256, 512, 1)]

        shared_features = l_features[-1]  # (total_rois=256, num_features=512, 1 point) i.e. we get one 512 dim feature for each of the 256 rois via Set Abstraction
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # input: (B=256, C=512, H and W =1) > conv1D(512, 256)+bn+relu > conv1D(256, 256)+bn+relu > Conv1d(256, 1) > output: (B=256, 1) : objectness scores for 256 rois 
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # input: (B=256, C=512, H and W =1) > conv1D(512, 256)+bn+relu > conv1D(256, 256)+bn+relu > Conv1d(256, 1) > output: (B=256, 7) : box predicted offsets (from predicted rois to gt boxes?) for 256 rois

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict
        return batch_dict
