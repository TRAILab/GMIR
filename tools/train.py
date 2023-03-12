import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import pickle
from tqdm import tqdm
import copy
import numpy as np

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader, build_dataset
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
from pcdet.utils import wandb_utils

from checkpoint import init_model_from_weights
torch.backends.cudnn.enabled = False

def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=none_or_str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=none_or_str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=True, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--save_ckpt_after_epoch', type=int, default=0, help='number of training epochs to save ckpt after')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=200, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = args.cfg_file.split('/')[-2] #'/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def compute_fisher(cfg, args, dist_train, logger):
    cfg.DATA_CONFIG.DATA_SPLIT['train'] = cfg.EWC.PREVIOUS_DATA_SPLIT
    cfg.DATA_CONFIG.INFO_PATH['train'] = cfg.EWC.PREVIOUS_INFO_PATH
    cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].DB_INFO_PATH = cfg.EWC.PREVIOUS_DBINFO_PATH
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if dist_train and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    # load checkpoint if it is possible
    if args.pretrained_model is not None:
        # if cfg.get('LOAD_WHOLE_MODEL', False):
        logger.info('**********************Loading whole model**********************')
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger) 

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)
    
    # -----------------------start training---------------------------
    logger.info('**********************Start Computing Fisher %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    
    model_func=model_fn_decorator()
    total_it_each_epoch = len(train_loader)
    dataloader_iter = iter(train_loader)

    model.train()
    optimizer.zero_grad()

    # Accumulate gradients for one epoch
    for cur_it in tqdm(range(total_it_each_epoch)):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        loss, _, _ = model_func(model, batch)

        loss.backward()        
    
    fisher_dict= {}
    optpar_dict = {}

    # gradients accumulated can be used to calculate fisher
    for name, param in model.named_parameters():
        optpar_dict[name] = param.data.clone()
        fisher_dict[name] = param.grad.data.clone().pow(2)

    return {'fisher' :fisher_dict, 'optpar': optpar_dict}

def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG # exp_group = 'dense_models', Tag = 'pvrcnn_train_clear_FOV3000_60' (.yaml)
    if args.extra_tag != 'default':
        output_dir = output_dir / args.extra_tag # 'pretrained chkpt or backbone ssl model'
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None
    # -----------------------compute or get ewc params---------------------------
    ewc_params = None
    if 'EWC' in cfg:
        ewc_param_path = output_dir / 'ewc_params.pkl'
        ewc_params = {}
        if ewc_param_path.exists():
            logger.info('**********************Loading EWC Params **********************')
            logger.info(f'EWC params path exists in: {ewc_param_path}')
            with open(ewc_param_path, 'rb') as f:
                ewc_params = pickle.load(f)
        else:
            ewc_params = compute_fisher(copy.deepcopy(cfg), copy.deepcopy(args), dist_train, logger)
            # save fisher and opt param means 
            if cfg.LOCAL_RANK == 0:
                logger.info('**********************Saving EWC Params **********************')
                logger.info(f'EWC params save path: {ewc_param_path}')
                with open(ewc_param_path, 'wb') as f:
                    pickle.dump(ewc_params, f)

    # -----------------------create dataloader & network & optimizer---------------------------
    if 'REPLAY' in cfg:
        original_dataset = build_dataset(dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        logger=logger,
        training=True) # build dataset of all splits = clear+adverse i.e. old + new

        train_set = copy.deepcopy(original_dataset)
         
        adverse_indices = train_set.get_adverse_indices()
        if cfg.REPLAY.method == 'AGEM':
            train_set.update_infos(adverse_indices) #only add adverse weather data for AGEM ## HERE ## adverse_indices[:20]
        else:
            # include 5% randomly selected clear weather examples
            clear_indices = train_set.get_clear_indices()
            #all samples = 6996 = 3365 adverse + 3631 clear
            # Select a subset of orig_clear_indices for e.g 50% of orig_clear_indices
            fn = output_dir / 'clear_indices.npy'
            if fn.exists():
                with open(fn, 'rb') as f:
                    clear_indices = np.load(f)
            else:
                if cfg.REPLAY.dataset_buffer_size == 1816:
                    clear_indices = np.array([i for i in clear_indices if i % 2 == 0])
                elif cfg.REPLAY.dataset_buffer_size == 1815:
                    clear_indices = np.array([i for i in clear_indices if i % 2 != 0])
                else:
                    clear_indices = np.random.permutation(clear_indices)[:cfg.REPLAY.dataset_buffer_size] # 720 clear indices
                if cfg.LOCAL_RANK == 0:
                    with open(fn, 'wb') as f:
                        np.save(fn, clear_indices)

            clear_indices = clear_indices.tolist() # this is not used for fixed replay

            clear_indices_selected_list = glob.glob(str(output_dir / '*clear_indices_selected*.npy'))
            if len(clear_indices_selected_list) > 0:
                clear_indices_selected_list.sort(key=os.path.getmtime)
                filename = clear_indices_selected_list[-1]
                if not os.path.isfile(filename):
                    raise FileNotFoundError

                logger.info('==> Loading clear_indices_selected from %s' % (filename))
                with open(filename, 'rb') as f:
                    clear_indices_selected = np.load(f)
            else:
                clear_indices_selected = np.random.permutation(clear_indices)[:cfg.REPLAY.memory_buffer_size] # random sample 180 clear samples
                if cfg.LOCAL_RANK == 0:
                    fn = output_dir / f'clear_indices_selected.npy'
                    with open(fn, 'wb') as f:
                        np.save(f, clear_indices_selected)
            
            clear_indices_selected = clear_indices_selected.tolist()
            all_indices = adverse_indices + clear_indices_selected ## HERE ## adverse_indices[:20]
            train_set.update_infos(all_indices)
        #assert len(all_indices) == len(train_set.get_adverse_indices()) + len(train_set.get_clear_indices())


    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None, 
        dataset = train_set if 'REPLAY' in cfg else None
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if dist_train and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        if cfg.get('LOAD_WHOLE_MODEL', False):
            ## For project
            logger.info('**********************Loading whole model**********************')
            model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger) 
        else:
            ### Change for finetuning
            logger.info('**********************Loading SSL backbone**********************')
            state = torch.load(args.pretrained_model)
            init_model_from_weights(model, state, freeze_bb=cfg.OPTIMIZATION.get('FREEZE_BB', False), logger=logger)

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # To continue training from last saved chkpt
    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1
            
            if 'REPLAY' in cfg and cfg.REPLAY.method == 'EMIR' and cfg.REPLAY.method_variant == 'None':
                grad_list = glob.glob(str(output_dir / '*model_param_grad_*.npy'))
                if len(grad_list) > 0:
                    grad_list.sort(key=os.path.getmtime)
                    with open(grad_list[-1], 'rb') as f:
                        model_grad = np.load(f)
                    
                    index = 0
                    for p in model.parameters():
                        if p.requires_grad:
                            n_param = p.numel()
                            p.grad = torch.zeros_like(p)
                            p.grad.copy_(torch.from_numpy(model_grad[index:index+n_param]).view_as(p))
                            index += n_param

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )
    
    wandb_utils.init(cfg, args, job_type='train')
    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    train_model(cfg, 
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        save_ckpt_after_epoch=args.save_ckpt_after_epoch,
        ewc_params=ewc_params,
        original_dataset= original_dataset if 'REPLAY' in cfg else None,
        dist_train=dist_train,
        args= args if 'REPLAY' in cfg else None, 
        output_dir=output_dir
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # logger.info('**********************Start evaluation %s/%s(%s)**********************' %
    #             (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    # wandb_utils.init(cfg, args, job_type='train_eval')
    # test_set, test_loader, sampler = build_dataloader(
    #     dataset_cfg=cfg.DATA_CONFIG,
    #     class_names=cfg.CLASS_NAMES,
    #     batch_size=args.batch_size,
    #     dist=dist_train, workers=args.workers, logger=logger, training=False
    # )
    # eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    # eval_output_dir.mkdir(parents=True, exist_ok=True)
    # args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)  # Only evaluate the last args.num_epochs_to_eval epochs


    # repeat_eval_ckpt(
    #     model.module if dist_train else model,
    #     test_loader, args, eval_output_dir, logger, ckpt_dir,
    #     dist_test=dist_train
    # )
    # logger.info('**********************End evaluation %s/%s(%s)**********************' %
    #             (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
