import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    # metric contains:
    # gt_num: total num all relevant gt boxes in the whole dataset 
    # recall_roi_0.3: num of all gt boxes which overlapped with the roi (1st stage) predicted boxes with iou3d > 0.3
    # recall_rcnn_0.3: num of all gt boxes which overlapped with the rcnn (final stage) predicted boxes with iou3d > 0.3
    # recall_roi_0.5:
    # recall_rnn_0.5:
    # recall_roi_0.7: (hardest so fewest boxes)
    # recall_rnn_0.7: (hardest so fewest boxes)
    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = [] # len: number of frames in dataset. Contains pred dicts for the whole dataset (see generate_single_sample_dict: how one pc's pred  dect is stored)

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        
        # Forward pass
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict) #batch_dict['points']:(16384x2, 5) [b_id, xyzi] #batch_dict['gt_boxes']: (2, num boxes, 8) [x, y, z, dx dy dz, r, class_id] class_id can be 1,2,3
        disp_dict = {}

        # Add ret_dict i.e. recall_dict for this batch to metric dict 
        statistics_info(cfg, ret_dict, metric, disp_dict)
        # Convert 3d boxes in lidar frame to 3d boxes in camera frame and store as kitti annos format
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir,
        eval_levels_cfg=cfg.MODEL.POST_PROCESSING.get('EVAL_LEVELS', None)
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


def eval_pickle(cfg, pickle_file, disard_results, dataloader, result_dir, epoch_id, logger):
    result_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataloader.dataset
    class_names = dataset.class_names

    logger.info('Evaluation on pickle file: {}'.format(pickle_file))

    det_annos = []
    with open(pickle_file, 'rb') as f:
        det_annos = pickle.load(f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        eval_levels_list_cfg=cfg.MODEL.POST_PROCESSING.get('EVAL_LEVELS_LIST', None)
    )

    ret_dict = {
        'EVAL_LEVELS_LIST': cfg.MODEL.POST_PROCESSING.get('EVAL_LEVELS_LIST', None)
    }

    logger.info(result_str)
    if isinstance(result_dict, dict):
        ret_dict.update(result_dict)
    else:
        ret_dict['distance'] = result_dict
    logger.info(ret_dict)

    if not disard_results:
        logger.info('Eval is saved to %s' % result_dir)
        with open(result_dir / '{}_eval.pkl'.format(cfg.TAG), 'wb') as f:
            pickle.dump(ret_dict, f)
    else:
        logger.info('Not saving results')

    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
