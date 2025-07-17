# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import sys
import os
import math
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size, all_gather, is_main_process, broadcast_data, get_rank
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.ema import ModelEma
from maskrcnn_benchmark.utils.amp import autocast, GradScaler
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from .inference import inference
from torch import nn
import pdb

import argparse

import cv2
import numpy as np
import torch
from torch import nn

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        val_data_loader=None,
        meters=None,
        zero_shot=False
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")

    def cal_torch_model_params(model):
        '''
        :param model:
        :return:
        '''
        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'total_params': total_params/10000, 'total_trainable_params': total_trainable_params/10000}

    logger.info(cal_torch_model_params(model))

    # meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    logger.info(cal_torch_model_params(model))
    model_ema = None
    if cfg.SOLVER.MODEL_EMA > 0:
        model_ema = ModelEma(model, decay=cfg.SOLVER.MODEL_EMA)
    start_training_time = time.time()
    end = time.time()

    if cfg.SOLVER.USE_AMP:
        scaler = GradScaler()

    global_rank = get_rank()

    if cfg.SOLVER.CHECKPOINT_PER_EPOCH != -1 and cfg.SOLVER.MAX_EPOCH >= 1:
        checkpoint_period = len(data_loader) * cfg.SOLVER.CHECKPOINT_PER_EPOCH // cfg.SOLVER.MAX_EPOCH
    
    if global_rank <= 0 and cfg.SOLVER.MAX_EPOCH >= 1:
        print("Iter per epoch ", len(data_loader) // cfg.SOLVER.MAX_EPOCH )

    if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:
        patience_counter = 0
        previous_best = 0.0

    # Adapt the weight decay
    # if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
    #     milestone_target = 0
    #     for i, milstone in enumerate(list(scheduler.milestones)):
    #         if scheduler.last_epoch >= milstone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
    #             milestone_target = i+1
    for iteration, (images, targets, idxs, positive_map, positive_map_eval, greenlight_map) in enumerate(data_loader, start_iter):
        nnegative = sum(len(target) < 1 for target in targets)
        nsample = len(targets)
        if nsample == nnegative or nnegative > nsample * cfg.SOLVER.MAX_NEG_PER_BATCH:
            logger.info('[WARNING] Sampled {} negative in {} in a batch, greater the allowed ratio {}, skip'.
                        format(nnegative, nsample, cfg.SOLVER.MAX_NEG_PER_BATCH))
            continue

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        captions = None
        try:
            targets = [target.to(device) for target in targets]
            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
        except:
            pass
        # Freeze language backbone
        if not cfg.PROMPT.ENABLE and cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            if hasattr(model, "module"):
                model.module.language_backbone.eval()
            else:
                model.language_backbone.eval()

        if cfg.SOLVER.USE_AMP:
            with autocast():
                if len(captions) > 0:
                    loss_dict = model(images, targets, captions, positive_map, greenlight_map = greenlight_map)
                else:
                    loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # save checkpoints for further debug if nan happens
            # loss_value = losses.item()
            # if not math.isfinite(loss_value):
            #     logging.error(f'=> loss is {loss_value}, stopping training')
            #     logging.error("Losses are : {}".format(loss_dict))
            #     time_str = time.strftime('%Y-%m-%d-%H-%M')
            #     fname = os.path.join(checkpointer.save_dir, f'{time_str}_states.pth')
            #     logging.info(f'=> save error state to {fname}')
            #     dict_to_save = {
            #         'x': images,
            #         'y': targets,
            #         'loss': losses,
            #         'states': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            #     }
            #     if len(captions) > 0:
            #         dict_to_save['captions'] = captions
            #         dict_to_save['positive_map'] = positive_map
            #     torch.save(
            #             dict_to_save,
            #             fname
            #         )


            if torch.isnan(losses) or torch.isinf(losses):
                logging.error("NaN encountered, ignoring")
                losses[losses != losses] = 0
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            # FIXME
            # if iteration % len(data_loader) == 0:
            #     scheduler.step()

        else:
            if len(captions) > 0:
                loss_dict = model(images, targets, captions, positive_map)
            else:
                loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # loss_value = losses.item()
            # if not math.isfinite(loss_value):
            #     logging.error(f'=> loss is {loss_value}, stopping training')
            #     time_str = time.strftime('%Y-%m-%d-%H-%M')
            #     fname = os.path.join(checkpointer.save_dir, f'{time_str}_states.pth')
            #     logging.info(f'=> save error state to {fname}')
            #     dict_to_save = {
            #         'x': images,
            #         'y': targets,
            #         'loss': losses,
            #         'states': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            #     }
            #     if len(captions) > 0:
            #         dict_to_save['captions'] = captions
            #         dict_to_save['positive_map'] = positive_map
            #     torch.save(
            #         dict_to_save,
            #         fname
            #     )
                

            if torch.isnan(losses) or torch.isinf(losses):
                losses[losses != losses] = 0
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            # scheduler.step()

        # Adapt the weight decay: only support multiStepLR
        # if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
        #     if milestone_target < len(scheduler.milestones):
        #         next_milestone = list(scheduler.milestones)[milestone_target]
        #     else:
        #         next_milestone = float('inf')
        #     if scheduler.last_epoch >= next_milestone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
        #         gamma = scheduler.gamma
        #         logger.info("Drop the weight decay by {}!".format(gamma))
        #         for param in optimizer.param_groups:
        #             if 'weight_decay' in param:
        #                 param['weight_decay'] *= gamma
        #         # move the target forward
        #         milestone_target += 1

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)
        if model_ema is not None:
            model_ema.update(model)
            arguments["model_ema"] = model_ema.state_dict()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
        # if iteration % 1 == 0 or iteration == max_iter:
            #logger.info(
            if global_rank <= 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "wd: {wd:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        wd=optimizer.param_groups[0]["weight_decay"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
        if val_data_loader and (iteration % checkpoint_period == 0 or iteration == max_iter):
        # if True:
            if is_main_process():
                print("Evaluating")
            eval_result = 0.0
            model.eval()
            if cfg.SOLVER.TEST_WITH_INFERENCE:
                with torch.no_grad():
                    try:
                        _model = model.module
                    except:
                        _model = model
                    _result = inference(
                        model = _model,
                        data_loader = val_data_loader,
                        dataset_name="val",
                        device=device,
                        expected_results=cfg.TEST.EXPECTED_RESULTS,
                        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                        output_folder=None,
                        cfg=cfg,
                        verbose=False
                    )
                    if is_main_process():
                        try:
                            eval_result = _result[0].results['bbox']['AP']
                        except:
                            print(' trainer.py 271 err')
            else:
                results_dict = {}
                cpu_device = torch.device("cpu")
                for i, batch in enumerate(val_data_loader):
                    images, targets, image_ids, positive_map, *_ = batch
                    with torch.no_grad():
                        images = images.to(device)
                        if positive_map is None:
                            output = model(images)
                        else:
                            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
                            output = model(images, captions, positive_map)
                        output = [o.to(cpu_device) for o in output]
                    results_dict.update(
                        {img_id: result for img_id, result in zip(image_ids, output)}
                    )
                all_predictions = all_gather(results_dict)
                if is_main_process():
                    predictions = {}
                    for p in all_predictions:
                        predictions.update(p)
                    predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                    eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
                                            box_only=cfg.DATASETS.CLASS_AGNOSTIC)
                    if cfg.DATASETS.CLASS_AGNOSTIC:
                        eval_result = eval_result.results['box_proposal']['AR@100']
                    else:
                        eval_result = eval_result.results['bbox']['AP']
            model.train()

            # if model_ema is not None and cfg.SOLVER.USE_EMA_FOR_MONITOR:
            #     model_ema.ema.eval()
            #     results_dict = {}
            #     cpu_device = torch.device("cpu")
            #     for i, batch in enumerate(val_data_loader):
            #         images, targets, image_ids, positive_map, positive_map_eval = batch
            #         with torch.no_grad():
            #             images = images.to(device)
            #             if positive_map is None:
            #                 output = model_ema.ema(images)
            #             else:
            #                 captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
            #                 output = model_ema.ema(images, captions, positive_map)
            #             output = [o.to(cpu_device) for o in output]
            #         results_dict.update(
            #             {img_id: result for img_id, result in zip(image_ids, output)}
            #         )
            #     all_predictions = all_gather(results_dict)
            #     if is_main_process():
            #         predictions = {}
            #         for p in all_predictions:
            #             predictions.update(p)
            #         predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
            #         eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
            #                                   box_only=cfg.DATASETS.CLASS_AGNOSTIC)
            #         if cfg.DATASETS.CLASS_AGNOSTIC:
            #             eval_result = eval_result.results['box_proposal']['AR@100']
            #         else:
            #             eval_result = eval_result.results['bbox']['AP']
                
            arguments.update(eval_result=eval_result)

            # if cfg.SOLVER.USE_AUTOSTEP:
            #     eval_result = all_gather(eval_result)[0] #broadcast_data([eval_result])[0]
            #     # print("Rank {} eval result gathered".format(cfg.local_rank), eval_result)
            #     scheduler.step(eval_result)
            
            # if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:
            #     if eval_result < previous_best:
            #         patience_counter += 1
            #     else:
            #         patience_counter = 0
            #         previous_best = eval_result
            #         checkpointer.save("model_best", **arguments)
            #     print("Previous Best", previous_best, "Patience Counter", patience_counter, "Eval Result", eval_result)
            #     if patience_counter >= cfg.SOLVER.AUTO_TERMINATE_PATIENCE:
            #         if is_main_process():
            #             print("\n\n\n\nAuto Termination at {}, current best {}\n\n\n".format(iteration, previous_best))
            #         break

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

class CamModel(nn.Module):
    def __init__(self, model, targets, captions, positive_map, greenlight_map):
        super(CamModel, self).__init__()
        self.model = model
        self.targets = targets
        self.captions = captions
        self.positive_map = positive_map
        self.greenlight_map = greenlight_map

    def forward(self, x):
        loss_dict = self.model(x, self.targets, self.captions, self.positive_map, self.greenlight_map)
        print(loss_dict)
        losses = sum(loss for loss in loss_dict.values())
        return losses

def reshape_transform(tensor, height=16, width=16):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def create_positive_map_label_to_token_from_positive_map(positive_map, plus = 0):
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = torch.nonzero(positive_map[i], as_tuple=True)[0].tolist()
    return positive_map_label_to_token

def do_train_cam(
        cfg,
        model,
        data_loader,
        device,
):
    # cam_model = CamModel(model)
    # # cam_model.train()
    # cam_model.to(device).eval()
    # target_layers = [cam_model.model.backbone[0].layers[-1].blocks[-1].norm1]
    # cam = GradCAM(model=cam_model, target_layers=target_layers)

    for iteration, (images, targets, idxs, positive_map, positive_map_eval, greenlight_map) in enumerate(data_loader):
        images = images.to(device)
        
        targets = [target.to(device) for target in targets]
        captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
        
        # captions = [t.get_field("caption") for t in targets]
        # print(f'local_rank:{os.environ["LOCAL_RANK"]}, captions:{captions}')
        # exit(0)
        # try:
        #     positive_map_eval = [t.get_field("positive_map_eval") for t in targets]
        # except:
        #     positive_map_eval = [t.get_field("positive_map") for t in targets]
        # if cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
        #     plus = 1
        # else:
        #     plus = 0
        # assert(len(positive_map_eval) == 1) # Let's just use one image per batch
        # positive_map_eval = positive_map_eval[0]
        # positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map_eval, plus=plus)
        
        cam_model = CamModel(model, targets, captions, positive_map=positive_map, greenlight_map=greenlight_map)
        cam_model.to(device).train()
        target_layers = [cam_model.model.backbone[0].layers[-1].blocks[-1].norm1]
        cam = GradCAM(model=cam_model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=images, targets=None)
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(images[0], reshape_transform(grayscale_cam.cpu().numpy()))
        cv2.imwrite("cam.jpg", cam_image)
