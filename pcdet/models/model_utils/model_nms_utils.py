import torch

from ...ops.iou3d_nms import iou3d_nms_utils


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    # box_scores: (N=16384 or N=100 for testing) max predicted class score for each point or box for example if for a point the scores are 
    # [car: 0.6, ped: 0.7, cycl: 0.3] then the box_score is 0.7 for that point
    # For testing box_scores are final rcnn predicted probabilities for objectness
    # box_preds: (N, 7) predicted box

    # 1. Select boxes with scores > score_thresh= 0.1
    # 2. Select topk scoring boxes (k = NMS_PRE_MAXSIZE)
    # 3. Perform NMS on these k boxes -> gives M boxes (where M < k)
    # 4. Select top "k=NMS_POST_MAXSIZE" scoring boxes out of M boxes
    src_box_scores = box_scores
    if score_thresh is not None:
        # discard all rcnn box predictions whose objectness prob < 0.1
        scores_mask = (box_scores >= score_thresh) 
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        # Select top 9000 points/boxes from 16384 points with highest box_scores
        # For testing select all boxes and sort them [highest box score, ..., lowest box score]
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        # Perform NMS on 9000 points/boxes
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
        ) # keep_idx can be of len less than 9000 after nms
        # Select 512 boxes after nms (from the valid boxes output by nms)
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]] # (512) indices of boxes to keep after nms

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]


def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]
            cur_box_preds = box_preds

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
            )
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes
