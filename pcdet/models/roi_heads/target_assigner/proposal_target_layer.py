import numpy as np
import torch
import torch.nn as nn

from ....ops.iou3d_nms import iou3d_nms_utils


class ProposalTargetLayer(nn.Module):
    def __init__(self, roi_sampler_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: 2
                rois: (B, num_rois, 7 + C): (2, 512, 7) predicted boxes from 1st stage
                roi_scores: (B, num_rois): (2, 512) predicted max class score from 1st stage (point head box)
                roi_labels: (B, num_rois): (2, 512) predicted class label from 1st stage
                gt_boxes: (B, N, 7 + C + 1): (2, 41, 8)
        Returns:
            target_dict:
                rois: (B, M, 7 + C): batch_rois (2, 128, 7)
                roi_scores: (B, M): batch_roi_scores (2, 128)
                roi_labels: (B, M): batch_roi_labels (2, 128)
                gt_of_rois: (B, M, 7 + C):  batch_gt_of_rois: (2, 128, 8)
                gt_iou_of_rois: (B, M):  batch_roi_ious: (2, 128)
                
                reg_valid_mask: (B, M): (2, 128)
                rcnn_cls_labels: (B, M): (2, 128) batch_cls_labels

            Read comments below.
        """
        
        # sample_rois_for_rcnn:
        # 1. Match 512 predicted boxes with the 41 gt boxes by computing iou3D matrix of size (512, 41) and taking max over each row
        # 2. Subsample/shortlist 512 predicted boxes to 128 boxes depending on their matched iou3D score
        # batch_rois: (2, 128, 7), batch_roi_labels: (2, 128), batch_roi_scores: (2, 128), batch_roi_ious: (2, 128), batch_gt_of_rois: (2, 128, 8)

        batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels = self.sample_rois_for_rcnn(
            batch_dict=batch_dict
        )
        # regression valid mask: #(2, 128) 1 for predicted boxes that have high iou3D with their matched gt boxes 
        # i.e. reg_valid_mask = if predicted boxes with iou3d > 0.55, then 1 else 0
        # i.e. possible foreground boxes
        reg_valid_mask = (batch_roi_ious > self.roi_sampler_cfg.REG_FG_THRESH).long() #(2, 128) 

        # classification label
        if self.roi_sampler_cfg.CLS_SCORE_TYPE == 'cls':
            # batch_cls_labels: (2, 128)
            # batch_cls_labels: is 1 if predicted boxes iou3d > 0.6 = CLS_FG_THRESH, i.e. for possible foreground boxes
            #                      0 if predicted boxes iou3d < 0.45 = CLS_BG_THRESH, i.e. for possible background boxes
            #                     -1 if 0.45 < predicted boxes iou3d < 0.6 (ignore these), i.e. for hard objects
            batch_cls_labels = (batch_roi_ious > self.roi_sampler_cfg.CLS_FG_THRESH).long()
            ignore_mask = (batch_roi_ious > self.roi_sampler_cfg.CLS_BG_THRESH) & \
                          (batch_roi_ious < self.roi_sampler_cfg.CLS_FG_THRESH)
            batch_cls_labels[ignore_mask > 0] = -1
        elif self.roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_iou':
            iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
            iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
            fg_mask = batch_roi_ious > iou_fg_thresh
            bg_mask = batch_roi_ious < iou_bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)

            batch_cls_labels = (fg_mask > 0).float()
            batch_cls_labels[interval_mask] = \
                (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)
        else:
            raise NotImplementedError

        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious,
                        'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels,
                        'reg_valid_mask': reg_valid_mask,
                        'rcnn_cls_labels': batch_cls_labels}

        return targets_dict

    def sample_rois_for_rcnn(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois = 512, 7 + C): (2, 512, 7) [x,y,z,dx,dy,dz,r] predicted boxes
                roi_scores: (B, num_rois): (2, 512) predicted box scores
                roi_labels: (B, num_rois): (2, 512) [1, ..., numclass] predicted class labels
                gt_boxes: (B, N, 7 + C + 1): (2, num boxes, 8) [x,y,z,dx,dy,dz,r, label] 
        Returns:
            batch_rois: (2, 128, 7) Shortlisted predicted boxes 
            batch_roi_labels: (2, 128) Shortlisted predicted boxes predicted class labels
            batch_roi_scores: (2, 128) Shortlisted predicted boxes predicted class score
            batch_roi_ious: (2, 128) max_overlaps i.e. matched iou3D for shortlisted predicted boxes
            batch_gt_of_rois: (2, 128, 8) Matched gt boxes for shortlisted predicted boxes 
            
            1. Match 512 predicted boxes with the 41 gt boxes by computing iou3D matrix of size (512, 41) and taking max over each row
            2. Subsample/shortlist 512 predicted boxes to 128 boxes depending on their matched iou3D score
                    Sample 64 boxes with matched iou3D > 0.55, these are most probably fg boxes
                    Sample 0.8 * 64 boxes with matched iou3D < 0.55 and > 0.1, ... hard_bg
                    Sample 0.2 * 64 boxes with matched iou3D <  0.1, ... easy_bg


        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois'] # (2, 512, 7)
        roi_scores = batch_dict['roi_scores'] # (2, 512)
        roi_labels = batch_dict['roi_labels'] # (2, 512)
        gt_boxes = batch_dict['gt_boxes'] # (2, num boxes, 8)

        code_size = rois.shape[-1] # 7
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size) # (2, 128, 7)
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size + 1)  # (2, 128, 8)
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE) # (2, 128)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE) # (2, 128)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long) # (2, 128)

        for index in range(batch_size):
            # get this pc predicted boxes=rois, gt boxes, predicted roi labels, predicted roi scores
            # cur_roi: (512, 7), cur_roi_labels: (512), cur_roi_scores: (512)
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = \
                rois[index], gt_boxes[index], roi_labels[index], roi_scores[index]
            k = cur_gt.__len__() - 1
            # move k to the valid gt box which is not all zeros
            while k >= 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1] # all valid (non-zero) gt boxes
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt # handle case when we have no valid gt boxes

            if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                # match 512 predicted boxes with the 41 gt boxes: For all car predicted boxes, get all car gt boxes and find gt box that gives max iou3d
                # Do the same for pedestrian and cyclist
                # Problem: we are only finding matches with gt boxes of the same class as the predicted boxes. 
                # If the predicted class of a box is wrong for example a car box is predicted as cyclist, we will be only finding matches of this box with cyclist gt boxes
                # this will lead to wrong matches
                # Also, here same gt box can be matched with multiple predicted boxes.
                # max_overlaps: (512) iou3D [0,..,1] of the matches 
                # gt_assignment: (512) index of matched gt box
                max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                    rois=cur_roi, roi_labels=cur_roi_labels,
                    gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, -1].long()
                )
            else:
                # match 512 predicted boxes with the 41 gt boxes without worrying about same class. i.e. find matches with max iou3D
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)

            # sample total ROI_PER_IMAGE:128 predicted boxes i.e. sample  (FG_RATIO * 128 = 64) fg boxes and (128-64=64) bg boxes
            # by thresholding max_overlaps with predefined thresholds to determine fg, easy bg and hard bg boxes and then randomly sample
            sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)

            batch_rois[index] = cur_roi[sampled_inds]
            batch_roi_labels[index] = cur_roi_labels[sampled_inds]
            batch_roi_ious[index] = max_overlaps[sampled_inds]
            batch_roi_scores[index] = cur_roi_scores[sampled_inds]
            batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]

        return batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels

    def subsample_rois(self, max_overlaps):
        # sample fg, easy_bg, hard_bg
        # Out of 512 predicted boxes we want to shortlist 128 boxes: 64 fg and 64 bg boxes
        # A predicted box is:
        # fg: if max_overlap (i.e. iou3d for a match) is >= min(REG_FG_THRESH: 0.55, CLS_FG_THRESH:0.6)
        # easy_bg: if max_overlap < CLS_BG_THRESH_LO: 0.1
        # hard_bg: CLS_BG_THRESH_LO: 0.1 <= max_overlap < REG_FG_THRESH: 0.55
        # After thresholding to find total fg, easy_bg, hard_bg boxes
        # randomly sample 64 of fg boxes, and 0.8*64 of hard_bg and 0.2*64 of easy_bg boxes. 0.8 is HARD_BG_RATIO
        # return sampled box indices in max_overlaps

        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH, self.roi_sampler_cfg.CLS_FG_THRESH)

        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)
        easy_bg_inds = ((max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)
        hard_bg_inds = ((max_overlaps < self.roi_sampler_cfg.REG_FG_THRESH) &
                (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = fg_inds[fg_inds < 0] # yield empty tensor

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois (predicted boxes): (N=512, 7): Pointrcnn N=512
            roi_labels (predicted box labels): (N=512)
            gt_boxes: (M, 7) e.g. (41, 7)
            gt_labels: (M)  e.g. (41)

        Returns:
            max_overlaps: (512) iou3d of the gt box matched with the predicted box
            gt_assignment: (512) index of gt box matched with the predicted box

            Match 512 predicted boxes with the 41 gt boxes and return the index of matched gt box and the iou3d between the matches
            Algorithm:
            for class_label in [1:car, 2:ped, 3:cyc]:
                cur_roi <- get all predicted boxes with predicted labels as 'class_label' (i.e. if class_label is 1 then get all car predicted boxes) # (e.g. 168 predicted car boxes)
                cur_gt <- get all car gt boxes (e.g. 14 car gt boxes)

                iou3d <- Compute 3d IOU matrix of size (168, 14)
                Find the gt box that gives max iou3D for each of the 168 car predicted boxes 
                store this gt box index in gt_assignment and the max iou3d in max_overlaps
                        
        """
        max_overlaps = rois.new_zeros(rois.shape[0]) #(512)
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0]) #(512)

        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1): #1, ..., num class=3
            roi_mask = (roi_labels == k) # true for rois which are labelled as k=1 i.e. car
            gt_mask = (gt_labels == k) # true for rois which are labelled as k=1 i.e. car
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_roi = rois[roi_mask] # (168,7) get all car rois 
                cur_gt = gt_boxes[gt_mask] # (14, 7) get all car gt_boxes 
                original_gt_assignment = gt_mask.nonzero().view(-1) # (14) get indices of car gt boxes

                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt)  # (predicted boxes num=168, gt boxes num=14) 3D iou = overlap_bev_area * overlap_height between each predicted and gt box of car class
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1) # for each predicted box get cur_max_overlaps: max iou3D with all gt box cur_max_overlaps, and cur_gt_assignment: gt box index that gives max iou3D with this predicted box
                max_overlaps[roi_mask] = cur_max_overlaps #(168)
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment] #(168)

        return max_overlaps, gt_assignment
