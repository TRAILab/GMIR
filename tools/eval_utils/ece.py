from pcdet.datasets.kitti.kitti_object_eval_python.eval import d3_box_overlap

import numpy as np
from math import ceil

def ece(gt, pred, iou_thres, num_bins=10, labels=['Car', 'Truck', 'Pedestrian']):
    """ Calculate Expected Calibration Error

    Args:
        gt (list of dictionaries): label, box
        pred (list of dictionaries): name, boxes_lidar, score
        iou_thres: used to separate TP and FP
        num_bins: number of intervals of scores between 0 and 1
        labels: labels that are of interest

    Returns:
        acc (numpy array of shape (num_bins, )): the accuracy of each pred score bin
            p.s. this should really be *precision* no accuracy, we are using the term loosely based on Di Feng's paper
        ece (scaler): expected calibration error
    """
    assert len(gt) == len(pred), "The length of gt frames and pred frames should be the same."

    bins = np.arange(0, 1+1/num_bins, 1/num_bins)
    conf_mat = [{'TP': 0, 'FP': 0} for i in range(len(bins))]
    
    for frame_idx in range(len(pred)):
        gt_label = gt[frame_idx]['label']
        gt_box = gt[frame_idx]['box']
        pred_label = pred[frame_idx]['name']
        pred_box = pred[frame_idx]['boxes_lidar']
        pred_score = pred[frame_idx]['score']

        for label in labels:
            gt_mask = gt_label == label
            pred_mask = pred_label == label
            gt_label_masked = gt_label[gt_mask]
            pred_label_masked = pred_label[pred_mask]
            gt_box_masked = gt_box[gt_mask]
            pred_box_masked = pred_box[pred_mask]
            pred_score_masked = pred_score[pred_mask]

            # all gt_box are FN
            if len(pred_box_masked) == 0:
                continue

            # all pred_box are FP
            if len(gt_box_masked) == 0:
                for score in pred_score_masked:
                    bin_num = ceil(score * num_bins)
                    conf_mat[bin_num]['FP'] += 1
                continue
            
            # when we have both gt_boxes and pred_box
            overlap = d3_box_overlap(gt_box_masked, pred_box_masked, z_axis=2, z_center=0.5)
            
            iou = np.max(overlap, axis=0)
            gt_index = np.argmax(overlap, axis=0)

            for i in range(len(iou)):
                bin_num = ceil(pred_score_masked[i] * num_bins)
                # Criteria for TP -> iou thres + label matching
                if iou[i] >= iou_thres and pred_label_masked[i] == gt_label_masked[gt_index[i]]:
                    conf_mat[bin_num]['TP'] += 1
                else:
                    conf_mat[bin_num]['FP'] += 1
                    
    # Calculate Precision from TP and FP
    acc = []
    count = []
    for bin_mat in conf_mat:
        if bin_mat['TP'] + bin_mat['FP'] == 0:
            acc.append(0)
            count.append(0)
        else:
            precision = bin_mat['TP'] / (bin_mat['TP'] + bin_mat['FP'])
            acc.append(precision)
            count.append(bin_mat['TP'] + bin_mat['FP'])

    acc = acc[1:] # the first element is always going to be zero
    count = count[1:]
    acc = np.array(acc)
    ece = np.sum(np.array(count) * np.abs(acc - np.arange(0.05, 1.05, 0.1)) / np.sum(count))
    return acc, ece