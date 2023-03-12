import io as sysio

import numba
import numpy as np

from .rotate_iou import rotate_iou_gpu_eval


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    # num_gt: num valid gt boxes
    # example: there are 400 gt boxes and 200 dt boxes matched i.e. we have 200 true positive/assigned dt scores
    # Recall on PR curve increases from 0, 1/400, 2/400, ..., 1 as gt boxes find a match one by one
    # We want to make PR curve with 40 points i.e. 40 score thresholds.
    #  So 400 num gt / 40 score thresh = 10 -> record dt_score as a score threshold on every 10th gt box matched
    # If we have 200 dt scores -> this will give us 200/10 = 20 score thresholds
    scores.sort()
    scores = scores[::-1] #sort dt_scores for true positive detections in descending order [highest score, ..., lowest score]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall # when i == 16 i.e. the last score
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds

# 9
def clean_data(gt_anno, dt_anno, current_class, difficulty):
    """
        This function is called on one pc's gt_anno and dt_anno. It extracts ground truth bounding boxes for current evaluation class
    Args:
        current_class: 0 for car, 1 for ped, 3 for cyc
        difficulty: 0 for easy, 1 for moderate, 3 for hard
    Returns:
        dc_bboxes: list of DontCare gt bboxes i.e. [dc_box_1: [x1,y1,x2,y2], ..., dc_box_m: [x1,y1,x2,y2]]
        
        num_valid_gt: num of gt boxes whose gt name matches the current class and whose difficulty is lower than the curent difficulty argument 
        i.e. (occlusion <= MAX_OCCLUSION[difficulty], truncation <= MAX_TRUNCATION[difficulty], 2d img bbox height in pixels > MIN_HEIGHT[diff])

        ignored_gt = -1, 0, 1 for each gt bbox in this pc
        [0 means valid gt_box bcz gt_name == current_class and gt_box_difficulty  is lower or equal to argument difficulty,
        1 means ignore gt box if gt_name == current_class and gt_box_difficulty  is higher than difficulty 
            OR gt class is a neighboring class i.e. gt_name == person_sitting while current class == Pedestrian OR  gt_name == VAN while current class == Car,
        -1 means do not use this gt box for evaluation bcz name does not match the current class argument]


        ignored_dt = -1, 0, 1 for each dt bbox in this pc
        [0 means valid dt_box bcz dt_name == current_class and dt_bbox_height is higher than min height for this difficulty,
        1 means ignore dt box if dt_bbox_height is lower than min height for this difficulty,
        -1 means do not use this dt box for evaluation bcz name does not match the current class argument]

    """
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'truck']
    MIN_HEIGHT = [40, 25, 25] # for easy, moderate, hard difficulty
    MAX_OCCLUSION = [0, 1, 2] # for easy, moderate, hard difficulty
    MAX_TRUNCATION = [0.15, 0.3, 0.5] # for 15 % for easy, 30 % for moderate, 50 % for hard difficulty
    # In annos:
    #    truncated    Float from 0 (non-truncated) to 1 (truncated), where
    #                      truncated refers to the % of object leaving image boundaries
    #    occluded     Integer (0,1,2,3) indicating occlusion state:
    #                      0 = fully visible, 1 = partly occluded
    #                      2 = largely occluded, 3 = unknown
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0

    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()

        # only bounding boxes with a minimum height are used for evaluation
        height = bbox[3] - bbox[1] #height of gt bbox in pixels

        # neighboring classes are ignored ("van" for "car" and "person_sitting" for "pedestrian")
        # (lower/upper cases are ignored)
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1 # gt class same as evaluation class
        elif (current_cls_name == "Pedestrian".lower()
              and "Person_sitting".lower() == gt_name):
            valid_class = 0 # gt class is a neighboring class of evaluation class
        elif (current_cls_name == "Car".lower() and "Van".lower() == gt_name):
            valid_class = 0 # gt class is a neighboring class of evaluation class
        else:
            valid_class = -1 # classes not used for evaluation for current class
        

        # if gt box occlusion, truncation and height are more difficult than the current difficulty level we are evaluating for then ignore these gt boxes
        # see https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
        #  ground truth is ignored, if occlusion, truncation exceeds the difficulty or ground truth is too small
        #  (doesn't count as FN nor TP, although detections may be assigned)
        ignore = False
        if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
                or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
                or (height <= MIN_HEIGHT[difficulty])):
            # if gt_anno["difficulty"][i] > difficulty or gt_anno["difficulty"][i] == -1:
            ignore = True 

        # Set ignored vector for ground truth boxes:
        #  Use gt boxes for evaluation if gt_name == current_class and gt_box_difficulty  is lower or equal to argument difficulty
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1 # (total no. of valid ground truth boxes for recall denominator)
        
        # Ignore gt boxes if neighboring class, or current class but ignored
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        
        # all other gt boxes which are not used in the evaluation
        else:
            ignored_gt.append(-1)

        # extract dontcare gt bboxes
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    
    # extract detections bounding boxes of the current class
    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1

        # set ignored vector for detections
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou

# 7
@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))

                #overlapping_height (iw) = min(y coord of bottom of two boxes) - max(y coord of top of two boxes)
                #since y axis of camera frame is pointing downwards, bottom of a box has higher y coord than top of a box in camera frame
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4])) 

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j] # intersection 3d = overlapping_height (iw) * overlapping bev area (rinc)
                    if criterion == -1:
                        ua = (area1 + area2 - inc) # union
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua #intersection over union
                else:
                    rinc[i, j] = 0.0

# 6
def d3_box_overlap(boxes, qboxes, criterion=-1):
    # boxes: (N, 7) : gt boxes [xyz box center in cam 0 rectified frame, l, h, w, rot_y]
    # qboxes: (M, 7) : pred boxes [xyz box center in cam 0 rectified frame, l, h, w, rot_y]

    # Find rinc = (N, M) bev overlap area
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2) # input: (N,5) gt boxes [x,z,l,w, rot_y] and (M,5) pred boxes -> output: (N, M) intersection areas of boxes in bev   
    # Compute iou3d i.e. rinc = iou3d (N, M)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc

# 10
@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):
    # This function operates on one frame and for one class e.g Car and one min_overlap threshold (e.g. 0.7) to be considered a match b/w gt and dt box
    # This function loops over gt boxes and then detections
    # and assigns same class detections to gt boxes (with same class as curr class or neighboring class) 
    # if the iou3d is > min_overlap, dt_score is higher than a thresh and iou3d is max compared to other detections 

    # After a possible detection candidate has been found for a gt box, we add to fn and tp counts:
    # fn: any good/valid gt box not assigned with any detection (neighboring class unassigned gt boxes or same class gt boxes with higher difficulty than curr diff are not included in fn)
    # e.g. a good car gt box is fn if it is not matched with/assigned to any car pred box (big or small). Even if it overlaps with cyclist pred box, it will still be a FN gt box.
    
    # tp: good/valid gt box assigned to good/valid dt box (matches not included in tp if gt is of neighboring class or more difficult or if dt box has small height )
    # valid dt box: same class as curr class and big enough height
    # valid gt box: same class as curr class and difficulty lower than or equal to curr difficulty

    # For fp count:
    # assign the remaining valid detections to DontCare boxes
    # Loop over detections: A car predicted dt box is false positive if it is not assigned to any Car or Van gt box (of any difficulty) 
    # AND it is a valid detection i.e. big enough and matching class to curr class AND its score is higher than thresh.
    # It is still a FP detection if the car pred box overlaps with cyclist gt box. 

    det_size = dt_datas.shape[0] # (M1 = number of pred boxes in frame 1)
    gt_size = gt_datas.shape[0] # (N1 = number of gt boxes in frame 1)
    dt_scores = dt_datas[:, -1] # (M1) pred box objectness scores
    dt_alphas = dt_datas[:, 4] # (M1) pred box alphas
    gt_alphas = gt_datas[:, 4] # (N1) gt bos alphas
    dt_bboxes = dt_datas[:, :4] # (M1, 4) pred 2d bbox img [x_left, y_top, x_right, y_bottom]
    gt_bboxes = gt_datas[:, :4]

    assigned_detection = [False] * det_size # holds whether a detection was assigned to a valid or ignored ground truth
    ignored_threshold = [False] * det_size # holds detections with a score lower than thresh=0, used if FP are computed i.e. for computing precision

    # detections with a low score are ignored for computing precision (which needs FP)
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True 
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size, )) # holds dt_scores for TP: true positive detection
    thresh_idx = 0
    delta = np.zeros((gt_size, )) #  holds angular difference for TPs (needed for AOS evaluation)
    delta_idx = 0

    # for gt box in all gt boxes in this frame
    # (evaluate all ground truth boxes)
    for i in range(gt_size): 
        if ignored_gt[i] == -1: # ignore this gt box if its class is not current eval class or a neighboring class
            continue
            
        # if gt box is of current class or neighboring class
        # /*=======================================================================
        # find candidate detections (overlap with ground truth > min overlap e.g 0.7) 
        # =======================================================================*/
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False # whether this gt bbox is assigned an ignored_det (i.e. a detection too small)

        # Search for a possible detection
        for j in range(det_size): # For pred box in all pred boxes this frame
            
            # detections not of the current class, already assigned or with a low threshold are ignored
            if (ignored_det[j] == -1): # if pred box's pred name does not match the current class
                continue
            if (assigned_detection[j]): # if pred box's is already assigned a gt box
                continue
            if (ignored_threshold[j]): # if pred box objectness score is below a true positive score threshold
                continue

            # so if current class is car, this makes sure we are at a ith gt box that is a car or Van and jth pred box that is predicted to be car 
            # So for a car gt box, we only search for a candidate in car detections. These car detections can be too small (i.e. ignored_det = 1) 
            # or big enough i.e. (ignored_det = 0)

            # find the maximum score for the candidates and get idx of respective detection
            overlap = overlaps[j, i] #iou3d between jth pred box and ith gt box
            dt_score = dt_scores[j]

            # for computing recall, the candidate (belonging to same class as gt) with the highest score is considered
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score

            # for computing pr curve values (i.e. compute fp), the candidate with the greatest overlap is considered and 
            # candidate which is not ignored_det (i.e. same class as gt and big enough) is preferred to be assigned. 
            # Meaning:
            # if we previously assigned a candidate which is an ignored det (i.e. a small detection) and now we have another
            # candidate which is not ignored_det (i.e. same class as gt and big enough) so consider this new detection 
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False

            # if no detection has been assigned yet and this candidate is an overlapping detection but a small detection, assign this.   
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j # store possible candidate id
                valid_detection = 1 # possible candidate found
                assigned_ignored_det = True # possible candidate is a small detection i.e. ignored_det == 1

        # End of for loop: Now that we have searched for possible candidate detection for this gt box
        # =======================================================================
        # compute TP, FP and FN
        # =======================================================================

        # A false negative gt box: if nothing was assigned to this valid ground truth (valid: matching class and difficulty to current class and difficulty)
        # Notice neighboring class gt boxes are not added in fn upon unassigned detection
        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1

        # if a possible candidate was found (i.e. valid_detection != NO_DETECTION) and 
        # either this gt box is a neighboring class box or matching class but higher difficulty than current diff
        # or the candidate detection is too small, record this detection as assigned and do not consider assigning it to other car gt boxes.
        # However, this assignment is not considered in tp
        
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        
        # only consider valid ground truth to valid detection assignments as tp i.e. when ignored_gt[i] == 0 and ignored_det[det_idx] == 0
        # (i.e. if both have matching class names to current class and matching difficulty level to curr diff)
        # Record this detection as assigned and do not consider assigning it to other car gt boxes.
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True
    
    # If we have to compute precision, consider dontcare areas
    if compute_fp:
        for i in range(det_size):

            # A detection is false positive if
            # it is not assigned to any gt box (whether same class or neighboring class) AND it is a valid detection i.e. big enough and matching class to curr class AND its score is higher than thresh
            # If the dt box was small or low scored or not of matching class or already assigned to a same class gt ot neighboring class gt then we would not have added/considered it in fp
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        
        
        # Do not consider detections overlapping with stuff area as FP, they are ok to be overlapping
        nstuff = 0
        if metric == 0: # if metric is 2d bbox in img
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)

            # Find overlapping detections to DontCare gt boxes and record them as assigned
            for i in range(dc_bboxes.shape[0]):
                # Search for possible candidate detection
                for j in range(det_size):
                    # The possible candidate cannot be already assigned, has to be of the same class as current class and big enough height and high score i.e.
                    # Detections not of the current class, already assigned, with a low threshold or a low minimum height are ignored
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (ignored_threshold[j]):
                        continue

                    # Assign this candidate detection to this DontCare box if overlap exceeds class specific value 
                    # And record this assignment 
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        # Do not consider detections overlapping with stuff area as FP, they are ok to be overlapping
        fp -= nstuff

        if compute_aos:
            # FP have a angle similarity of 0, for all TP compute AOS
            tmp = np.zeros((fp + delta_idx, )) # zeros of size fp + tp because delt_idx == tp
            # tmp = [0] * fp

            # for all TP compute AOS
            for i in range(delta_idx):
                # tmp is an angle similarity score for each detection.
                # tmp is 0 if a deetction is FP.
                # For TP detection: tmp is closer to 1 if angle difference b/w gt and assigned dt is close to 0, tmp is close to 0 if angle diff is close to 180 deg 
                # e.g tmp is 1 if delta[i] is 0 deg, tmp is 1/2 if delta[i] is 90 deg,  tmp is 0 if delta[i] is 180 deg
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0 # delta[i] is the ith tp's delta alpha between gt and assigned dt
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp) # higher this number, better heading estimation
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]

# 4
def get_split_parts(num, num_part):
    # if num examples or frames = 501
    # num parts to split these in is 100
    # same_part = 501 // 100 = 5
    # remain_num = 501 % 100 = 1
    # return [5, 5, 5, ...., 5, 1] i.e first 100 values are 5 i.e. 500 frames divided into 100 equal parts and then remaining part 1 is in the end.
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [num]

    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    """
    Inputs: 
        overlaps: iou3d of shape (M, N). M = num pred boxes in all frames in a part, N = num gt boxes in all frames in a part
        pr: (num true positive detections, 4) zeros, 4 cols for [tp, fp, fn, similarity]
        gt_nums: list containing num gt boxes in each frame e.g. [4 gt boxes, 5, 18, 2,...]
        dt_nums: list containing num dt boxes in each frame e.g. [6 dt boxes, 4, 16, 2,...]
        dc_nums: list containing num dc boxes in each frame 
        gt_datas: (N, 5) [2d bbox, alpha]
        dt_datas: (M, 6) [2d bbox, alpha, score]
        dontcares: (K, 4) [[2d bbox for dc box 1], ...., [2d bbox for dc box K]]
        ignored_gts: (N)
        ignored_dets: (M)
        thresholds: list of objectness score thresholds in descending order 

    """
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]): # for frame_i in frames
        # compute tp, fp, fn and similarity for this frame and for all objectness score thresholds (a.k.a. recall thresholds) arranged in descending order
        # As we decrease score thresh -> more dt boxes qualify for matching, tp increases which decreases fn by the same amount hence recall increases but fp also increases since we are relaxing the threshold and considering more dt boxes for matching so precision reduces. 
        for t, thresh in enumerate(thresholds):
            #For this frame: extract iou3d, gt data, dt data, ignored gt for the current class and diff, ignored dt for the current class and diff and dc boxes
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:
                               gt_num + gt_nums[i]] # iou3d for this frame (M1 = num dt boxes in frame 1, N1= num gt boxes in frame 1)
            
            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]

            # compute tp, fp, fn and AOS for this frame and this detection score
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            
            # add no. of TP, FP, FN, AOS for current frame to total evaluation for current score threshold
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]

# 5
def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    returns:
        overlap is a list of same len as number of frames in test dataset.
        overlap = [iou3d of shape (N1 gt boxes, M1 pred boxes) for frame 1, 
                   iou3d of shape (N2 gt boxes, M2 pred boxes) for frame 2, ..., 
                   iou3d for last frame in dataset]
        
        parted_overlaps is a list of same len as split_parts.
        parted_overlaps = [iou3d of shape (N_part1, M_part1), 
                           iou3d of shape (N_part2, M_part2),..., 
                           iou3d of shape (N_part_last, M_part_last)]
        total_gt_num = (num_frames) [N1, N2, ..., N_lastframe] stores num gt boxes in each frame
        total_dt_num = (num_frames) [M1, M2, ..., M_lastframe] stores num pred boxes in each frame
    """
    assert len(gt_annos) == len(dt_annos) # e.g.  (20 frames)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0) # list of len = len(dt_annos) = 20: contains num predicted boxes in each frame e.g. [11 (boxes in frame 0), 4, 3 etc]
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0) # list of len = len(gt_annos) = 20: contains number of gt boxes in each frame
    num_examples = len(gt_annos)
    
    # divide dataset into 100 parts/chunks of equal sized frames + 1 chunk of remaining frames
    split_parts = get_split_parts(num_examples, num_parts) # gives list of chunk sizes,  elem_i == num frames in part_i
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0) # x,y,z center of bbox in camera frame (total bboxes = num frames x num bboxes per frame, 3) 
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0) # (total bboxes = num frames x num bboxes per frame, 3) 
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0) # (total bboxes = num frames x num bboxes per frame,) 
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1) #  (total bboxes = num frames x num bboxes per frame, 7) 
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64) # iou3d = (N gt boxes, M pred boxes)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part

    # parted_overlaps is a list of same len as split_parts.
    # parted_overlaps = [iou3d of shape (N_part1, M_part1), 
    #                    iou3d of shape (N_part2, M_part2),..., 
    #                    iou3d of shape (N_part_last, M_part_last)]
    # N_part1 is num gt boxes in all frames in part/chunk 1
    # M_part1 is num pred boxes in all frames in part 1
    overlaps = []
    example_idx = 0
    # Fill 'overlaps' = list of len ==number of frames in dataset
    # ith element of overlap is a matrix of iou3d of shape (M_i, N_i) for i_th frame
    # overlap = [iou3d of shape (N1 gt boxes, M1 pred boxes) for frame 1, 
    #           iou3d of shape (N2 gt boxes, M2 pred boxes) for frame 2, ..., iou3d for last frame in dataset]
    # overlap and parted_overlaps contain the same info except overlap is frame-wise iou3d, where as the parted_overlaps has part-wise iou3d
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num

# 8
def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    """
    Args:
        gt_annos: list of len == num frames in dataset, containing gt_annos for each frame
        dt_annos: list of len == num frames in dataset, containing pred_annos for each frame
        current_class: 1 value from 0 for car, 1 for ped, 3 for cyc
        difficulty: 1 value from 0 for easy, 1 for moderate, 3 for hard

    Returns:
        
        gt_datas_list: list of len == num frames in dataset, [(N1, 5), ..., (N_last_frame, 5)]
        element 1 in list is matrix of size (N1= num of gt boxes in frame 1, 5) containing gt bbox and alpha [x_left, y_top, x_right, y_bottom, alpha]

        dt_datas_list: list of len == num frames in dataset, [(M1, 6), ..., (M_last_frame, 6)]
        element 1 in list is matrix of size (M1= num of dt boxes in frame 1, 6) containing dt bbox, alpha and box objectness score [x_left, y_top, x_right, y_bottom, alpha]

        dontcares: list of len == num frames in dataset, [[DontCare gt 2d img bboxes in frame 1], ..., [DontCare gt 2d img bboxes in last frame]]
        total_dc_num:  list of len == num frames in dataset, [num DontCare gt boxes in frame 1, num in frame 2, ..., num in last frame]

        For a particular current class and difficulty:

        ignored_gts: list of len == num frames in dataset, [(N1), (N2), ..., (N_last_frame)]
        element 1 in list is a vector of size (N1= num of gt boxes in frame 1) containing a value of either 0, -1 or 1 for each gt box
    
        0 means valid gt_box bcz gt_name == current_class and gt_box_difficulty  is lower than difficulty,
        1 means ignored gt box if gt_name == current_class and gt_box_difficulty  is higher than difficulty 
            OR gt_name == person_sitting while current class == Pedestrian OR  gt_name == VAN while current class == Car,
        -1 means ignore this gt box bcz name does not match the current class argument


        ignored_dets: list of len == num frames in dataset, [(M1), (M2), ..., (M_last_frame)]
        element 1 in list is a vector of size (M1= num of dt boxes in frame 1) containing a value of either 0, -1 or 1 for each dt box

        0 means valid dt_box bcz dt_name == current_class and dt_bbox_height is higher than min height for this difficulty,
        1 means ignored dt box if dt_bbox_height is lower than min height for this difficulty,
        -1 means ignore this dt box bcz name does not match the current class argument

        total_num_valid_gt = sum of all 0 entries in ignored_gts
    """
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    # Loop over all frames in the dataset
    for i in range(len(gt_annos)): # for each frame
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1) # (N1, 5)
        dt_datas = np.concatenate([
            dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis],
            dt_annos[i]["score"][..., np.newaxis]
        ], 1)  # (M1, 6)
        gt_datas_list.append(gt_datas) # [(N1, 5), ..., (N_last_frame, 5)]
        dt_datas_list.append(dt_datas) # [(M1, 6), ..., (M_last_frame, 6)]
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)

# 3
def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               metric,
               min_overlaps,
               compute_aos=False,
               num_parts=100):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: list of len == num of frames in dataset. Each element is gt_annos dict containing gt boxes for a frame, must from get_label_annos() in kitti_common.py
        dt_annos: list of len == num of frames in dataset. Each element is dt_annos dict containing pred boxes for a frame, must from get_label_annos() in kitti_common.py
        current_classes: list of int, 0: car, 1: pedestrian, 2: cyclist
        difficultys: list of int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps: float, min overlap. format: [num_overlap=2, metric=3, class=3].
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos) #number of frames are equal
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts) # if num examples is 501, and num_parts = 100 then split_parts = [5,5, ..., 5, 1] i.e. [5] * 100 + [1]

    # ATTENTION: Because we pass dt_annos to gt_annos arg in calculate_iou_partly function,
    # the returned overlaps is of shape:
    #         overlap = [iou3d of shape (M1 pred boxes, N1 gt boxes) for frame 1, 
    #                    iou3d of shape (M2 pred boxes, N2 gt boxes) for frame 2, ..., 
    #                    iou3d for last frame in dataset]
    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps) # 2 = 0th dim of min_overlaps
    num_class = len(current_classes) # 3
    num_difficulty = len(difficultys) # 3

    # Precision-Recall curve is a 41 point curve for each of the 3x3x2 combos i.e. 1) 41 point curve for (car, easy, 0.7 iou thresh), 2) 41 point curve for (car, easy, 0.5 iou3d thresh) etc
    # Each point in PR curve is the precision and recall is computed by setting a dt objectness score threshold. If a dt box's score > this score thresh -> this predicted box can be used in tp, fp calc i.e. this is an object box otherwise dont consider this pred box as a pred object box and assume it doesnt exist i.e. don't include it in tp and fp calc.
    # Sampled score thresholds can be less than 41 so the PR curve can stop at recall < 1. If num score thresholds < 41, the rest of P-R values are zero.
    precision = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]) # (3 classes, 3 difficulties easy moderate hard, 2 iou3d thresh i.e. 0.7 and 0.5, 41 sample points i.e. score thresholds)
    recall = np.zeros( [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]) # (3,3,2,41)
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]) # (3,3,2,41)
    for m, current_class in enumerate(current_classes):
        for l, difficulty in enumerate(difficultys):
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]): # for k, min_overlap in [0.7, 0.5] for car class m = 0
                thresholdss = [] # detection scores for true positive/assigned dt boxes in all frames for the current class and difficulty
                for i in range(len(gt_annos)): # For each frame, get dt scores from true positive/assigned dt boxes. In compute_statistics_jit, assign a valid (same class and big height) dt box to a gt box which has max iou3d
                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        metric,
                        min_overlap=min_overlap,
                        thresh=0.0,
                        compute_fp=False)
                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist()
                thresholdss = np.array(thresholdss) # dt_scores for true positive detections (tp: valid detections overlapping with valid gt boxes with iou3d > min_overlap)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt) # dt score thresholds in descending order sampled from true positive dt scores. Each of this score threshold will give one point on PR curve
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4]) # [tp, fp, fn, AOS similarity]: For each score thresh, we get one point in PR curve i.e. one precision and recall value calculated from [tp, fp, fn, AOS similarity]
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx:idx + num_part], 0) # (N= num gt boxes in all frames in this part, 5)
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx:idx + num_part], 0)  # (M= num dt boxes in all frames in this part, 6)
                    dc_datas_part = np.concatenate(
                        dontcares[idx:idx + num_part], 0)  # (k= num dc boxes in all frames in this part, 4)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx + num_part], 0) # (N,)
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0) # (M,)
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos)
                    idx += num_part
                for i in range(len(thresholds)):
                    # Calculate precision and recall value for m_th class, l_th difficulty, k_th iou3d thresh and i_ith score threshold
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2]) # tp_for dt_score threshold i / (tp_for dt_score threshold i + fn_for dt_score threshold i)
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1]) # tp_i / (tp_i + fp_i)
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1]) # similarity sum over all detections / (tp_i + fp_i)
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(
                        precision[m, l, k, i:], axis=-1) # fill in precision at threshold_i with max precision at and after this thresh i 
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)  # fill in recall at threshold_i with max recall at and after this thresh i, basically fills in recall at all thresh with the max value of recall
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
    ret_dict = {
        "recall": recall,
        "precision": precision,
        "orientation": aos,
    }
    return ret_dict


def get_mAP(prec):
    # returns AP_R11 i.e. average of precision values averaged over all 11 true positive or recall score thresholds 
    sums = 0
    # sum precision values for (3 classes, 3 diff, 2 min overlaps) over 11 score thresholds (i.e. i= 0, 4, .. 40)
    for i in range(0, prec.shape[-1], 4): # i = 0, 4, 8, ..., 40
        sums = sums + prec[..., i]
    return sums / 11 * 100 # sum is of shape (3,3,2)


def get_mAP_R40(prec):
    # prec for a metric e.g. 3d bbox is a (3 classes, 3 diffi, 2 iou thresh, 41 score thresholds)
    # returns AP_R40 i.e. average of precision values averaged over all 40 true positive or recall score thresholds
    sums = 0
    # Average 40 precision points in one 41 point PR curve to get average precision. 
    # sum precision values for (3 classes, 3 diff, 2 min overlaps) PR curves over 40 score thresholds (i.e. i= 1, 2, .. 40)
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100 # sum is of shape (3,3,2)


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()

# 2
def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            compute_aos=False,
            PR_detail_dict=None):
    # min_overlaps: [num_minoverlap=2, metric=3, num_class=3]
    # num_minoverlap = 2 = first dim of min_overlaps i.e. overlap_0_7, overlap_0_5
    # metric = 3 = second dim of min_overlaps i.e. 0th row = 2d bbox, 1st row = bev, 2nd row = 3d bbox
    # num_class = 3 = 3rd dim i.e. columns i.e. 0th col = car, 1st col= ped, 2nd col = cyc

    # min_overlaps[0] = overlap_0_7's =                car, ped, cyc 
                                        #   2d bbox [[0.7, 0.5, 0.5], 
                                        #    BEV     [0.7, 0.5, 0.5],
                                        #    3d bbox [0.7, 0.5, 0.5]])
    # min_overlaps[1] = overlap_0_5
    # columns: car, ped, cyc,
    # rows: 2d bbox, bev, 3d bbox
    #  = np.array([[0.7, 0.5, 0.5], 
                 # [0.5, 0.25, 0.25],
                 # [0.5, 0.25, 0.25]])

    difficultys = [0, 1, 2]
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 0,
                     min_overlaps, compute_aos)
    # ret: [num_class, num_diff, num_minoverlap, num_sample_points]
    mAP_bbox = get_mAP(ret["precision"]) # for 2d bbox in camera image 
    mAP_bbox_R40 = get_mAP_R40(ret["precision"])

    if PR_detail_dict is not None:
        PR_detail_dict['bbox'] = ret['precision']

    mAP_aos = mAP_aos_R40 = None
    if compute_aos:
        mAP_aos = get_mAP(ret["orientation"])
        mAP_aos_R40 = get_mAP_R40(ret["orientation"])

        if PR_detail_dict is not None:
            PR_detail_dict['aos'] = ret['orientation']

    # metric = 1 i.e. bev
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1,
                     min_overlaps)
    mAP_bev = get_mAP(ret["precision"])
    mAP_bev_R40 = get_mAP_R40(ret["precision"])

    if PR_detail_dict is not None:
        PR_detail_dict['bev'] = ret['precision']

    # metric = 2 i.e. 3d bbox
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2,
                     min_overlaps)
    mAP_3d = get_mAP(ret["precision"])
    mAP_3d_R40 = get_mAP_R40(ret["precision"])
    if PR_detail_dict is not None:
        PR_detail_dict['3d'] = ret['precision']
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos, mAP_bbox_R40, mAP_bev_R40, mAP_3d_R40, mAP_aos_R40


def do_coco_style_eval(gt_annos, dt_annos, current_classes, overlap_ranges,
                       compute_aos):
    # overlap_ranges: [range, metric, num_class]
    min_overlaps = np.zeros([10, *overlap_ranges.shape[1:]])
    for i in range(overlap_ranges.shape[1]):
        for j in range(overlap_ranges.shape[2]):
            min_overlaps[:, i, j] = np.linspace(*overlap_ranges[:, i, j])
    mAP_bbox, mAP_bev, mAP_3d, mAP_aos = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos)
    # ret: [num_class, num_diff, num_minoverlap]
    mAP_bbox = mAP_bbox.mean(-1)
    mAP_bev = mAP_bev.mean(-1)
    mAP_3d = mAP_3d.mean(-1)
    if mAP_aos is not None:
        mAP_aos = mAP_aos.mean(-1)
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos

#1
def get_official_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    # dt_annos are prediction outputs from model. len(dt_annos) == len(test dataset)
    # gt annos have corresponding gt boxes and labels for every frame

    # Columns are for car, ped, cyc, van, person sitting, truck
    # Rows are for 2d bbox, bev, 3d bbox
    # We are only interested in 3d bbox row and car, ped, cyc columns
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7], 
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])  
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], 
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
        5: 'Truck'
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int #[car, ped, cyc] convert to ints -> [0, 1, 2]
    min_overlaps = min_overlaps[:, :, current_classes] 
    # min_overlaps[0] = overlap_0_7's =                car, ped, cyc 
                                        #   2d bbox [[0.7, 0.5, 0.5], 
                                        #    BEV     [0.7, 0.5, 0.5],
                                        #    3d bbox [0.7, 0.5, 0.5]])
    # min_overlaps[1] = overlap_0_5
    # columns: car, ped, cyc,
    # rows: 2d bbox, bev, 3d bbox
    #  = np.array([[0.7, 0.5, 0.5], 
                 # [0.5, 0.25, 0.25],
                 # [0.5, 0.25, 0.25]])
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    
    # Start Evaluation
    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict)
    # End Evaluation
    
    # Populate ret_dict and results_str
    ret_dict = {}
    for j, curcls in enumerate(current_classes): # [0:car, 1:ped, 2:cyc]
        # min_overlaps: [num_minoverlap, metric, class]
        # mAP per metric result: [num_class, num_diff, num_minoverlap]
        cls = class_to_name[curcls]
        for i in range(min_overlaps.shape[0]): #[0.7 min overlap, 0.5 min overlap]
            
            bbox_min_overlap = min_overlaps[i, 0, j]
            bev_min_overlap = min_overlaps[i, 1, j]
            d3_min_overlap = min_overlaps[i, 2, j]

            #--------------------- AP at 11 recall points --------------------
            # result += print_str((f"{cls} AP_R11@{bbox_min_overlap}, {bev_min_overlap}, {d3_min_overlap}:")) # Car AP_R11@   0.7(min overlap for 2dbbox) 0.7(for bev) 0.7(for 3d bbox)
            # result += print_str((f"bbox AP:{mAPbbox[j, 0, i]:.4f}, "
            #                      f"{mAPbbox[j, 1, i]:.4f}, "
            #                      f"{mAPbbox[j, 2, i]:.4f}")) # Car bbox AP_R11@  easy moderate hard
            # result += print_str((f"bev  AP:{mAPbev[j, 0, i]:.4f}, "
            #                      f"{mAPbev[j, 1, i]:.4f}, "
            #                      f"{mAPbev[j, 2, i]:.4f}"))  # Car bev AP_R11@  easy moderate hard
            # result += print_str((f"3d   AP:{mAP3d[j, 0, i]:.4f}, "
            #                      f"{mAP3d[j, 1, i]:.4f}, "
            #                      f"{mAP3d[j, 2, i]:.4f}"))  # Car 3d AP_R11@  easy moderate hard

            # if compute_aos:
            #     result += print_str((f"aos  AP:{mAPaos[j, 0, i]:.2f}, "
            #                          f"{mAPaos[j, 1, i]:.2f}, "
            #                          f"{mAPaos[j, 2, i]:.2f}"))  # Car aos AP_R11@  easy moderate hard

            #--------------------- AP at 40 recall points --------------------
            # result += print_str( (f"{cls} AP_R40@{bbox_min_overlap}, {bev_min_overlap}, {d3_min_overlap}:")) # Car AP_R40@   0.7(min overlap for 2dbbox) 0.7(for bev) 0.7(for 3d bbox)
            # result += print_str((f"bbox AP:{mAPbbox_R40[j, 0, i]:.4f}, "
            #                      f"{mAPbbox_R40[j, 1, i]:.4f}, "
            #                      f"{mAPbbox_R40[j, 2, i]:.4f}")) # Car bbox AP_R40@  easy moderate hard
            # result += print_str((f"bev  AP:{mAPbev_R40[j, 0, i]:.4f}, "
            #                      f"{mAPbev_R40[j, 1, i]:.4f}, "
            #                      f"{mAPbev_R40[j, 2, i]:.4f}")) # Car bev AP_R40@  easy moderate hard
            # result += print_str((f"3d   AP:{mAP3d_R40[j, 0, i]:.4f}, "
            #                      f"{mAP3d_R40[j, 1, i]:.4f}, "
            #                      f"{mAP3d_R40[j, 2, i]:.4f}")) # Car 3d AP_R40@  easy moderate hard

            result += print_str( (f"{cls}_3d_AP_R40_iou_{d3_min_overlap}: "
            f"Easy: {mAP3d_R40[j, 0, i]:.4f}, " 
            f"Moderate: {mAP3d_R40[j, 1, i]:.4f}, "
            f"Hard: {mAP3d_R40[j, 2, i]:.4f}"))


            if compute_aos:
                result += print_str( (f"{cls}_aos_AP_R40_iou_{bbox_min_overlap}: "
                    f"Easy: {mAPaos_R40[j, 0, i]:.4f}, " 
                    f"Moderate: {mAPaos_R40[j, 1, i]:.4f}, "
                    f"Hard: {mAPaos_R40[j, 2, i]:.4f}"))
                # result += print_str((f"aos  AP:{mAPaos_R40[j, 0, i]:.2f}, "
                #                      f"{mAPaos_R40[j, 1, i]:.2f}, "
                #                      f"{mAPaos_R40[j, 2, i]:.2f}")) # Car aos AP_R40@  easy moderate hard
                ret_dict[f'{cls}_aos_AP_R40_iou_{bbox_min_overlap}/easy'] = mAPaos_R40[j, 0, i] #[class, diff, min overlap]
                ret_dict[f'{cls}_aos_AP_R40_iou_{bbox_min_overlap}/moderate'] = mAPaos_R40[j, 1, i]
                ret_dict[f'{cls}_aos_AP_R40_iou_{bbox_min_overlap}/hard'] = mAPaos_R40[j, 2, i]

            ret_dict[f'{cls}_3d_AP_R40_iou_{d3_min_overlap}/easy'] = mAP3d_R40[j, 0, i]
            ret_dict[f'{cls}_3d_AP_R40_iou_{d3_min_overlap}/moderate'] = mAP3d_R40[j, 1, i]
            ret_dict[f'{cls}_3d_AP_R40_iou_{d3_min_overlap}/hard'] = mAP3d_R40[j, 2, i]
            
            # ret_dict[f'{cls}_bev_R40_iou_{bev_min_overlap}/easy'] = mAPbev_R40[j, 0, i]
            # ret_dict[f'{cls}_bev_R40_iou_{bev_min_overlap}/moderate'] = mAPbev_R40[j, 1, i]
            # ret_dict[f'{cls}_bev_R40_iou_{bev_min_overlap}/hard'] = mAPbev_R40[j, 2, i]
            
            # ret_dict[f'{cls}_image_R40_iou_{bbox_min_overlap}/easy'] = mAPbbox_R40[j, 0, i]
            # ret_dict[f'{cls}_image_R40_iou_{bbox_min_overlap}/moderate'] = mAPbbox_R40[j, 1, i]
            # ret_dict[f'{cls}_image_R40_iou_{bbox_min_overlap}/hard'] = mAPbbox_R40[j, 2, i]

    return result, ret_dict


def get_coco_eval_result(gt_annos, dt_annos, current_classes):
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
    }
    class_to_range = {
        0: [0.5, 0.95, 10],
        1: [0.25, 0.7, 10],
        2: [0.25, 0.7, 10],
        3: [0.5, 0.95, 10],
        4: [0.25, 0.7, 10],
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    overlap_ranges = np.zeros([3, 3, len(current_classes)])
    for i, curcls in enumerate(current_classes):
        overlap_ranges[:, :, i] = np.array(
            class_to_range[curcls])[:, np.newaxis]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    mAPbbox, mAPbev, mAP3d, mAPaos = do_coco_style_eval(
        gt_annos, dt_annos, current_classes, overlap_ranges, compute_aos)
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        o_range = np.array(class_to_range[curcls])[[0, 2, 1]]
        o_range[1] = (o_range[2] - o_range[0]) / (o_range[1] - 1)
        result += print_str((f"{class_to_name[curcls]} "
                             "coco AP@{:.2f}:{:.2f}:{:.2f}:".format(*o_range)))
        result += print_str((f"bbox AP:{mAPbbox[j, 0]:.2f}, "
                             f"{mAPbbox[j, 1]:.2f}, "
                             f"{mAPbbox[j, 2]:.2f}"))
        result += print_str((f"bev  AP:{mAPbev[j, 0]:.2f}, "
                             f"{mAPbev[j, 1]:.2f}, "
                             f"{mAPbev[j, 2]:.2f}"))
        result += print_str((f"3d   AP:{mAP3d[j, 0]:.2f}, "
                             f"{mAP3d[j, 1]:.2f}, "
                             f"{mAP3d[j, 2]:.2f}"))
        if compute_aos:
            result += print_str((f"aos  AP:{mAPaos[j, 0]:.2f}, "
                                 f"{mAPaos[j, 1]:.2f}, "
                                 f"{mAPaos[j, 2]:.2f}"))
    return result
