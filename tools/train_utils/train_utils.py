import glob
import os

import torch
import tqdm
import time
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils
from pcdet.utils import wandb_utils
from pcdet.datasets import build_dataloader
import copy
import math
import numpy as np
#from matplotlib import pyplot as plt

def train_one_epoch(cfg, cur_epoch, model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, ewc_params=None, old_dataloader = None):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)
    if old_dataloader is not None:
        old_dataloader_iter = iter(old_dataloader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()
        if old_dataloader is not None:
            overhead_time_agem = common_utils.AverageMeter() #overhead time taken for one iter avged over iters in one epoch
            if cfg.get('overhead_time_agem_sum', 0) == 0:
                cfg['overhead_time_agem_sum'] = 0 # summed over all iters


    for cur_it in range(total_it_each_epoch):
        end = time.time()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')
        if old_dataloader is not None:
            #overhead_start = time.time()
            cur_overhead_time = 0 #cfg.get('iter_cur_overhead_time_agem', 0) #this iter's overhead time for agem
            try:
                old_batch = next(old_dataloader_iter)
            except StopIteration:
                old_dataloader_iter = iter(old_dataloader)
                old_batch = next(old_dataloader_iter)
                print('new iters')
            
            #cur_overhead_time += time.time() - overhead_start  #old dataloading time for agem

        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter) #lr_head
            # if len(optimizer.param_groups) > 1:
            #     tb_log.add_scalar('meta_data/learning_rate_1', optimizer.param_groups[1]['lr'], accumulated_iter) #lr_bb

        model.train()
        if old_dataloader is not None:
            start_time  = time.time()
            optimizer.zero_grad()
            old_loss, _, _ = model_func(model, old_batch)
            old_loss.backward()
            clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            grad_old = []
            for p in model.parameters():
                if p.requires_grad:
                    grad_old.append(p.grad.view(-1))  
            grad_old = torch.cat(grad_old) 
            cur_overhead_time += time.time() - start_time # time for computing gradients wrt a minibatch of old data

        optimizer.zero_grad()
        loss, tb_dict, disp_dict = model_func(model, batch)

        if ewc_params is not None:
            for name, param in model.named_parameters():
                try:
                    fisher = ewc_params['fisher'][name]
                    optpar = ewc_params['optpar'][name]
                except:
                    fisher = ewc_params['fisher'][name[7:]]
                    optpar = ewc_params['optpar'][name[7:]]
                fisher = fisher.to(model.device)
                optpar = optpar.to(model.device)
                assert(fisher.requires_grad == False)
                assert(optpar.requires_grad == False)
                
                loss += (fisher * (optpar - param).pow(2)).sum() * cfg.EWC.LAMBDA


        cur_forward_time = time.time() - data_timer

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        if old_dataloader is not None:
            start_time = time.time()
            grad_new = []
            for p in model.parameters():
                if p.requires_grad:
                    grad_new.append(p.grad.view(-1))  
            grad_new = torch.cat(grad_new) 

            angle = (grad_old * grad_new).sum()
            if angle < 0:
                grad_old_sq_mag = (grad_old * grad_old).sum()
                grad_proj = grad_new - (angle/grad_old_sq_mag) * grad_old

                index = 0
                for p in model.parameters():
                    if p.requires_grad:
                        n_param = p.numel()
                        p.grad.copy_(grad_proj[index:index+n_param].view_as(p))
                        index += n_param
            
            cur_overhead_time += time.time() - start_time


        optimizer.step()
        accumulated_iter += 1

        cur_batch_time = time.time() - end

        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)
        if old_dataloader is not None:
            avg_overhead_agem_time = commu_utils.average_reduce_value(cur_overhead_time)
        

        # log to console and tensorboard
        if rank == 0:
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            
            if len(optimizer.param_groups) > 1:
                disp_dict.update({
                    'loss': loss.item(), 'lr': cur_lr, 'lr_bb': optimizer.param_groups[1]['lr'], 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                    'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
                })
            else:
                 disp_dict.update({
                    'loss': loss.item(), 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                    'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
                })

            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()


            wb_dict = {}
            if len(optimizer.param_groups) > 1:
                for i, pg in enumerate(optimizer.param_groups):
                    wb_dict[f'lr_{i}'] = pg['lr']
            else:
                wb_dict['lr'] = cur_lr
            wb_dict['loss'] = loss
            wb_dict['epoch'] = cur_epoch
            if old_dataloader is not None:
                overhead_time_agem.update(avg_overhead_agem_time)
                cfg['overhead_time_agem_sum'] += overhead_time_agem.val
                #print(f'overhead: {overhead_time}')
                wb_dict['iter_overhead_time_agem'] = overhead_time_agem.val
                wb_dict['iter_overhead_time_agem_avg'] = overhead_time_agem.avg
                wb_dict['iters_overhead_time_agem_sum'] = cfg['overhead_time_agem_sum']


            wandb_utils.log(cfg, wb_dict, accumulated_iter)

            # if tb_log is not None:
            #     tb_log.add_scalar('train/loss', loss, accumulated_iter)
            #     tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
            #     # if len(optimizer.param_groups) > 1:
            #     #     tb_log.add_scalar('meta_data/learning_rate_1', optimizer.param_groups[1]['lr'], accumulated_iter)
            #     for key, val in tb_dict.items():
            #         tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(cfg, model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, save_ckpt_after_epoch=0, ewc_params=None, original_dataset=None, args=None, dist_train=False, output_dir=None):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)

        if rank == 0:
            epoch_time = common_utils.AverageMeter()
            overhead_time = common_utils.AverageMeter()

        if 'REPLAY' in cfg:
            orig_clear_indices = original_dataset.get_clear_indices() # HERE [:50]
            adverse_indices = original_dataset.get_adverse_indices() # HERE [:20]

            # Select a subset of orig_clear_indices for e.g 50% of orig_clear_indices
            fn = output_dir / 'clear_indices.npy'
            if fn.exists():
                with open(fn, 'rb') as f:
                    clear_indices = np.load(f)
            else:
                clear_indices = np.random.permutation(orig_clear_indices)[:cfg.REPLAY.dataset_buffer_size] # 720 clear indices
                if rank == 0:
                    with open(fn, 'wb') as f:
                        np.save(fn, clear_indices)

            clear_indices = clear_indices.tolist() # this is not used for fixed replay

            if (cfg.REPLAY.method == 'MIR' and cfg.REPLAY.epoch_interval == 1):
                # Reduce and fix samples in dataset buffer to save time
                
                model.train()
                loss_prev_epoch_for_all_clear_samples = []

                # Calculate loss using cur epoch params for each sample in original dataset
                # here single batch is single sample
                with torch.no_grad():
                    for idx in tqdm.tqdm(clear_indices): #calc loss of model on 720 clear samples
                        sample = original_dataset[idx]
                        batch = original_dataset.collate_batch([sample])

                        loss, _, _ = model_func(model, batch)
                        loss_prev_epoch_for_all_clear_samples.append(loss.item())

            elif cfg.REPLAY.method == 'AGEM':
                # Reduce and fix samples in dataset buffer to save time
                clear_indices_selected_list = glob.glob(str(output_dir / '*clear_indices_selected*.npy'))
                if len(clear_indices_selected_list) > 0:
                    clear_indices_selected_list.sort(key=os.path.getmtime)
                    filename = clear_indices_selected_list[-1]
                    if not os.path.isfile(filename):
                        raise FileNotFoundError

                    with open(filename, 'rb') as f:
                        clear_indices_selected = np.load(f)
                else:
                    clear_indices_selected = np.random.permutation(clear_indices)[:cfg.REPLAY.memory_buffer_size] # random sample 180 clear samples
                    if rank == 0:
                        fn = output_dir / f'clear_indices_selected.npy'
                        with open(fn, 'wb') as f:
                            np.save(f, clear_indices_selected)

                clear_indices_selected = clear_indices_selected.tolist()
                old_train_set = copy.deepcopy(original_dataset)
                old_train_set.update_infos(clear_indices_selected)
                old_train_set, old_train_loader, old_train_sampler = build_dataloader(
                    dataset_cfg=cfg.DATA_CONFIG,
                    class_names=cfg.CLASS_NAMES,
                    batch_size=args.batch_size,
                    dist=dist_train, workers=2,
                    training=True,
                    seed=666 if args.fix_random_seed else None, 
                    dataset = old_train_set
                )
            elif cfg.REPLAY.method == 'GSS':
                num_params_model =  sum(p.numel() for p in model.parameters() if p.requires_grad)

        if dist_train:
            torch.distributed.barrier()
        
        # start_condition = True
        for cur_epoch in tbar:
            
            end = time.time()

            if 'REPLAY' in cfg and cfg.REPLAY.method != 'fixed':
                
                if cur_epoch > 0 and (cur_epoch) % cfg.REPLAY.epoch_interval == 0:
                    # Generate new trainset, train loader and sampler
                    
                    model.train()

                    if cfg.REPLAY.method == 'MIR':
                        assert len(loss_prev_epoch_for_all_clear_samples) > 0

                        decrease_in_loss_all_clear_samples = []
                            
                        # Calculate loss using cur epoch params for each sample in original dataset
                        # here single batch is single sample
                        with torch.no_grad():
                            for i, idx in tqdm.tqdm(enumerate(clear_indices)):  #calc loss of model on 720 clear samples
                                sample = original_dataset[idx]
                                batch = original_dataset.collate_batch([sample])

                                loss, _, _ = model_func(model, batch)
                                prev_loss = loss_prev_epoch_for_all_clear_samples[i]
                                decrease_in_loss_all_clear_samples.append(prev_loss-loss.item())
                                loss_prev_epoch_for_all_clear_samples[i] = loss.item() #update loss on all clear samples
                            
                        #Sort decrease_in_loss_all_clear_samples in ascending order and choose k samples with lowest decrease in loss
                        decrease_in_loss_idx_selected = np.argsort(np.array(decrease_in_loss_all_clear_samples))[:cfg.REPLAY.memory_buffer_size]
                        clear_indices_selected = [clear_indices[i] for i in decrease_in_loss_idx_selected]
                        if rank == 0:
                            fn = output_dir / f'clear_indices_selected_{cur_epoch}.npy'
                            with open(fn, 'wb') as f:
                                np.save(f, clear_indices_selected)


                    elif cfg.REPLAY.method == 'random':
                        clear_indices_selected = np.random.permutation(clear_indices)[:cfg.REPLAY.memory_buffer_size].tolist()
                        if rank == 0:
                            fn = output_dir / f'clear_indices_selected_{cur_epoch}.npy'
                            with open(fn, 'wb') as f:
                                np.save(f, clear_indices_selected)

                    elif cfg.REPLAY.method == 'AGEM' and cfg.REPLAY.method_variant == 'plus':
                        # resample randomly 180 clear samples from full clear dataset after every 10 epochs and update the replay buffer with new samples
                        clear_indices_selected = np.random.permutation(clear_indices)[:cfg.REPLAY.memory_buffer_size].tolist()
                        old_train_loader.dataset.update_infos_given_infos(clear_indices_selected, original_dataset.dense_infos)
                        fn = output_dir / f'clear_indices_selected_{cur_epoch}.npy'
                        if rank == 0:
                            with open(fn, 'wb') as f:
                                np.save(fn, clear_indices_selected)
                        
                    elif cfg.REPLAY.method == 'GSS':
                        bool_arr = [True, False]
                        param_grad_mask = np.random.choice(bool_arr, num_params_model, p=[0.01, 0.99])
                        num_params_selected = param_grad_mask.sum()
                        #grad_old_dataset = np.zeros((len(clear_indices), num_params_model), dtype=np.float32) #, device=model.device
                        grad_old_dataset = np.zeros((len(clear_indices), num_params_selected), dtype=np.float32) #, device=model.device

                        #start = time.time()
                        for i, idx in tqdm.tqdm(enumerate(clear_indices)):
                            sample = original_dataset[idx]
                            batch = original_dataset.collate_batch([sample])

                            loss, _, _ = model_func(model, batch)

                            optimizer.zero_grad()
                            loss.backward()
                            grad_idx_sample = []
                            for p in model.parameters():
                                if p.requires_grad:
                                    grad_idx_sample.append(p.grad.view(-1))  
                            grad_idx_sample = torch.cat(grad_idx_sample) 
                            grad_old_dataset[i, :] = grad_idx_sample.cpu().numpy()[param_grad_mask]
                        
                        # print(f'Time: {time.time() - start}')
                        # torch.matmul(grad_old_dataset, grad_old_dataset)
                        grad_dot = grad_old_dataset @ grad_old_dataset.T
                        grad_mag = np.linalg.norm(grad_old_dataset, axis=1).reshape((len(clear_indices),1)) # get column vector of len clear_indices
                        grad_mag_cross = grad_mag @ grad_mag.T # col vector x row vector
                        cos_theta_old_dataset = np.divide(grad_dot, grad_mag_cross) # 50 x 50 i.e. len clear_indices x len clear_indices
                        np.fill_diagonal(cos_theta_old_dataset, -np.inf)
                        max_cos_theta = np.max(cos_theta_old_dataset, axis=0) # column-wise max i.e. get a row vector 1 x  len clear_indices
                        cos_theta_idx_selected = np.argsort(np.array(max_cos_theta))[:cfg.REPLAY.memory_buffer_size]
                        clear_indices_selected = [clear_indices[idx] for idx in cos_theta_idx_selected]
                        # plt.imshow(cos_theta_old_dataset, cmap='gray')
                        # plt.show()
                        if rank == 0:
                            fn = output_dir / f'clear_indices_selected_{cur_epoch}.npy'
                            with open(fn, 'wb') as f:
                                np.save(f, clear_indices_selected)

                    elif cfg.REPLAY.method == 'EMIR':
                        models_current_grad = {}

                        if cfg.REPLAY.method_variant == 'plus':
                            for name, param in model.named_parameters():
                                if param.requires_grad:
                                    models_current_grad[name] = torch.zeros_like(param)
                            
                            # Calculate average gradient over all adverse data using current model
                            for idx in tqdm.tqdm(adverse_indices):
                                sample = original_dataset[idx]
                                batch = original_dataset.collate_batch([sample])

                                loss, _, _ = model_func(model, batch)

                                optimizer.zero_grad()
                                loss.backward()
                                for name, param in model.named_parameters():
                                    if param.requires_grad:
                                        models_current_grad[name] += param.grad.data

                            for name, param in model.named_parameters():
                                if param.requires_grad:
                                    models_current_grad[name] = models_current_grad[name] / len(adverse_indices)
                        
                        elif cfg.REPLAY.method_variant == 'None':
                            for name, param in model.named_parameters():
                                if param.requires_grad:
                                    models_current_grad[name] = param.grad.data.clone()


                        cos_theta_for_all_clear_samples = []

                        # Calculate cos_theta = - interference score for each sample in original dataset
                        # here single batch is single sample
                        # start = time.time()
                        for idx in tqdm.tqdm(clear_indices):
                            model.train()
                            sample = original_dataset[idx]
                            batch = original_dataset.collate_batch([sample])

                            loss, _, _ = model_func(model, batch)

                            optimizer.zero_grad()
                            loss.backward() 
                            a_dot_b = 0
                            norm_a = 0
                            norm_b = 0
                            # Take dot product between current gradient vector and gradient vector
                            # due to this sample to get the cos(angle) = a . b / (norm_a * norm_b)

                            with torch.no_grad():
                                if cfg.REPLAY.method_variant == 'EMA':
                                    param_idx = 0
                                    state_dict = optimizer.state_dict()['state']
                                    for group in optimizer.param_groups:
                                        for param in group['params']:
                                            if param.requires_grad:
                                                a_dot_b += torch.dot(state_dict[param_idx]['exp_avg'].view(-1), param.grad.view(-1)).item()
                                                norm_a += state_dict[param_idx]['exp_avg'].pow(2).sum().item()
                                                norm_b += param.grad.pow(2).sum().item()
                                            param_idx += 1
                                else:
                                    for name, param in model.named_parameters():
                                        if param.requires_grad:
                                            a_dot_b += torch.dot(models_current_grad[name].view(-1), param.grad.view(-1)).item()
                                            norm_a += models_current_grad[name].pow(2).sum().item()
                                            norm_b += param.grad.pow(2).sum().item()

                                norm_a = math.sqrt(norm_a)
                                norm_b = math.sqrt(norm_b)

                                cos_theta_for_all_clear_samples.append(a_dot_b/(norm_a * norm_b))

                        #print(f'Time: {time.time() - start}')
                        #Sort cos_theta_clear_samples in ascending order and choose k samples with lowest cos(theta)
                        cos_theta_idx_selected = np.argsort(np.array(cos_theta_for_all_clear_samples))[:cfg.REPLAY.memory_buffer_size]
                        clear_indices_selected = [clear_indices[idx] for idx in cos_theta_idx_selected]
                        if rank == 0:
                            fn = output_dir / f'clear_indices_selected_{cur_epoch}.npy'
                            with open(fn, 'wb') as f:
                                np.save(f, clear_indices_selected)

                    
                    if cfg.REPLAY.method != 'AGEM':
                        # include max interfered clear weather examples 
                        all_indices = adverse_indices + clear_indices_selected
                        train_loader.dataset.update_infos_given_infos(all_indices, original_dataset.dense_infos)

            cur_overhead_time = time.time() - end

            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)
            
            if 'REPLAY' in cfg and cfg.REPLAY.method == 'AGEM' and old_train_sampler is not None:
                old_train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            if dist_train:
                torch.distributed.barrier()
            end = time.time()

            accumulated_iter = train_one_epoch(cfg, cur_epoch,
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter, ewc_params=ewc_params, old_dataloader = old_train_loader if 'REPLAY' in cfg and cfg.REPLAY.method == 'AGEM' else None
            )

            cur_epoch_time = time.time() - end

            if rank == 0:
                overhead_time.update(cur_overhead_time)
                epoch_time.update(cur_epoch_time)
                wb_dict = {'epoch_overhead_time': overhead_time.val, 
                           'epoch_overhead_time_avg': overhead_time.avg,
                           'epoch_overhead_time_sum': overhead_time.sum,
                           'epoch_time': epoch_time.val, 
                           'epoch_time_sum': epoch_time.sum,
                           'epoch_time_avg': epoch_time.avg}
                wandb_utils.log(cfg, wb_dict, accumulated_iter)
                #print(f'EPOCH {cur_epoch} time: {epoch_time.val} sec\n')

                # save model param grads for EMIR variant None since it depends on grad from last iteration
                if 'REPLAY' in cfg and cfg.REPLAY.method == 'EMIR' and cfg.REPLAY.method_variant  == 'None':
                    if cur_epoch > 0 and (cur_epoch+1) % cfg.REPLAY.epoch_interval == 0:
                        model_param_grads = []
                        for p in model.parameters():
                            if p.requires_grad:
                                model_param_grads.append(p.grad.view(-1))  
                        model_param_grads = torch.cat(model_param_grads).cpu().detach().numpy()

                        fn = output_dir / f'model_param_grad_{cur_epoch+1}.npy'
                        with open(fn, 'wb') as f:
                            np.save(f, model_param_grads)
                
            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch >= save_ckpt_after_epoch and trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        if torch.__version__ >= '1.4':
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename, _use_new_zipfile_serialization=False)
        else:
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    if torch.__version__ >= '1.4':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)
