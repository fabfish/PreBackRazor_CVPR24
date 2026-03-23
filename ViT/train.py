# coding=utf-8
"""
Training entrypoint for PreBackRazor ViT experiments.

This file is adapted from the original BackRazor ViT training pipeline and
keeps the JointSparse-related logic used by this project.
"""
from __future__ import absolute_import, division, print_function

import sys
sys.path.append(".")
import argparse
import os

from datetime import timedelta

import torch

from torch.utils.tensorboard import SummaryWriter

from ViT.utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from ViT.utils.data_utils import get_loader
from ViT.utils.utils import *
from ViT.utils.memory_cost_profiler import profile_memory_cost


import time
import mesa as ms

from torch import nn

import numpy as np

from tqdm import tqdm

class Coding:
    """Base class for coding."""
    def __init__(self, *args, **kwargs):
        self.codes = []

    def encode(self, grad, *args, **kwargs):
        raise NotImplementedError()

    def decode(self, code, *args, **kwargs):
        raise NotImplementedError()

class MyJointSpar(Coding):
    """
    JointSparse-style parameter selection with adaptive probabilities.

    Algorithm reference:
      Rui Liu and Barzan Mozafari,
      "Communication-efficient Distributed Learning for Large Batch Optimization" (ICML 2022).

    Implementation contribution in this codebase:
      Chen Shixiang and Cheng Yifei.
    """
    def __init__(self, sparsity=0.5, network = None, *args, **kwargs):
        self.s = sparsity
        self.prob = None
        self.grad = None
        self.alpha_p = 0.001
        # self. active_params: Bool array of parameters that originally requires grad
        
        self.params_prob_list = []
        count = 0
        # for p_index, p in enumerate(network.parameters()):
        #     if p.requires_grad:
        #         self.params_prob_list.append(count)
        #         count = count + 1
        #     else:
        #         self.params_prob_list.append(-1)
        
        for p_index, p in enumerate(network.named_parameters()):
            if p[1].requires_grad and not ("head" in p[0]) and not  ("embeddings" in p[0]) and not ("norm" in p[0]):
                self.params_prob_list.append(count)
                count = count + 1
            else:
                self.params_prob_list.append(-1)

        # print("active_params: ", self.active_params)

    def set_sparsity(self, sparsity):
        self.s = sparsity

    def get_prob(self, abs_grad_array, idx_array):
        L = max(abs_grad_array)
        p_min = 0.001        
        w = self.prob
        budget = self.s * len(w)
        prob_array = self.prob[idx_array]
        l_hat_raw = - (abs_grad_array / prob_array) ** 2 + (L / p_min) ** 2
        l_hat = l_hat_raw - np.min(l_hat_raw)
        w[idx_array] = prob_array * np.exp(-self.alpha_p * l_hat / prob_array)
        sort_indx = np.argsort(-w)
        w_sort = -np.sort(-w)
        j = 0 
        while (j < len(w)):
            i = sort_indx[j]
            temp_sum = max(sum(w_sort[j:]), 1e-5)
            if w[i] * budget <= temp_sum:
                for k in range(j, len(w)):
                    i = sort_indx[k]
                    self.prob[i] = w[i] * budget / float(temp_sum)
                j = len(w) + 1
            else:
                self.prob[i] = 1
                budget -= 1
            j += 1
        self.prob = np.clip(self.prob, a_min=1e-3, a_max=1.0)

    def init_distribution(self, network):
        # len_param = len([_ for _ in network.parameters() if _.requires_grad])
        len_param = len([p for p in network.named_parameters() if p[1].requires_grad and not ("head" in p[0]) and not  ("embeddings" in p[0]) and not ("norm" in p[0])])
        self.prob = self.s * np.ones(len_param)
        # self.prob = self.prob * self.active_params

    def update_distribution(self, network):
        grads = []
        idx_set = []
        for p_index, p in enumerate(network.parameters()):
            if self.params_prob_list[p_index] == -1:
                continue
            if p.requires_grad is True:
                if p.grad is not None:
                    grad = p.grad.data.cpu().numpy().astype(np.float32)
                else:
                    grad = np.zeros(p.shape)
                grad_norm = np.linalg.norm(grad)
                if grad_norm == 0 or grad_norm == np.inf:
                    continue
                # idx_set.append(p_index)
                idx_set.append(self.params_prob_list[p_index])
                grads.append(grad_norm)

        if len(idx_set) > 0:
            idx_array = np.array(idx_set)
            self.get_prob(np.array(grads), idx_array)

    def set_requires_grad(self, network):
        while True:
            num_req_grad = 0
            for p_index, p in enumerate(network.parameters()):
                prob_index = self.params_prob_list[p_index]
                if prob_index == -1:
                    continue

                if self.prob is None:
                    threshold = 0.5
                else:
                    threshold = self.prob[prob_index]
                if torch.rand(1).cuda() > threshold:
                    p.requires_grad_(False)
                else:
                    num_req_grad += 1
                    p.requires_grad_(True)
            if num_req_grad > 0:
                break     
    
    def set_requires_grad_with_lr(self, network, optimizer):
        while True:
            num_req_grad = 0
            for p_index, p in enumerate(network.parameters()):
                prob_index = self.params_prob_list[p_index]
                if prob_index == -1:
                    continue

                if self.prob is None:
                    threshold = 0.5
                else:
                    threshold = self.prob[prob_index]
                if torch.rand(1).cuda() > threshold:
                    p.requires_grad_(False)
                else:
                    num_req_grad += 1
                    p.requires_grad_(True)
                    optimizer.param_groups[p_index]["lr"] /= threshold
            if num_req_grad > 0:
                break   

    def encode(self, v, **kwargs):
        if isinstance(v, (torch.Tensor, torch.cuda.FloatTensor)):
            w = v.cpu().numpy()
        elif isinstance(v, np.ndarray):
            w = v
        elif v is None:
            w = None
        else:
            raise ValueError("Object passed to encode not ndarray or torch.Tensor")
        p_index = kwargs['p_index']
        if w is not None:
            code = w / self.prob[p_index]
        else:
            code = w
        return code

    def decode(self, code, **kwargs):
        if code is not None:
            v = torch.Tensor(code)
        else:
            v = None
        return v


def valid(args, model, writer, test_loader, global_step, log, phase="Validation"):
    # Validation!
    eval_losses = AverageMeter()

    log.info("***** Running {} *****".format(phase))
    log.info("  Num steps = {}".format(len(test_loader)))
    log.info("  Batch size = {}".format(args.eval_batch_size))

    end = time.time()

    model.eval()
    all_preds, all_label = [], []
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(test_loader):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            if "resnet" in args.model_type:
                logits = model(x)
            else:
                logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )

        if step % 50 == 0:
            log.info("Validating {}/{} (loss={:2.5f})".format(step, len(test_loader), eval_losses.val))

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    log.info("\n")
    log.info("Validation Results")
    log.info("Global Steps: {}".format(global_step))
    log.info("Valid Loss: {}".format(eval_losses.avg))
    log.info("Valid Accuracy: {}".format(accuracy))
    log.info("Time spent: {:.2f}".format(time.time() - end))


    if args.local_rank in [-1, 0]:
        writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train(args, model, train_loader, val_loader, test_loader, log, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    memory_meter = AverageMeter()

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare optimizer and scheduler
    params_list = []
    if args.HeadLr10times:
        head_params = []
        feature_params = []
        for name, parameter in model.named_parameters():
            if "resnet" in args.model_type:
                head_name = "fc"
            else:
                head_name = "head"

            if head_name in name:
                head_params.append(parameter)
            else:
                feature_params.append(parameter)

        params_list = [{"params": filter(lambda p: p.requires_grad, feature_params)},
                       {"params": filter(lambda p: p.requires_grad, head_params), "lr": args.learning_rate * 10}]
    else:
        # params_list = model.parameters()
        if args.layerwiselr:
            for name, parameter in model.named_parameters():
                params_list.append({'params': parameter, 'lr': args.learning_rate})
        else:
            params_list = model.parameters()

    optimizer = torch.optim.SGD(params_list,
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    # Train!
    log.info("***** Running training *****")
    log.info("  Total optimization steps = {}".format(args.num_steps))
    log.info("  Instantaneous batch size per GPU = {}".format(args.train_batch_size))
    log.info("  Total train batch size (w. parallel, distributed & accumulation) = {}".format(
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1)))
    log.info("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step = 0

    # Initialize JointSparse probability distribution.
    if args.jointsparse:
        coder = MyJointSpar(sparsity=args.joint_sparsity if args.joint_sparsity is not None else 0.5, network=model)
        coder.init_distribution(model)
        log.info("Initialized JointSparse distribution.")

    epoch = 0
    accuracy = -1
    while True:
        log.info("set epochs: {}".format(epoch))
        train_loader.sampler.set_epoch(epoch)
        epoch += 1

        model.train()

        # Progress bar for training loop.
        with tqdm(
            total=len(train_loader), 
            desc="Training", 
            disable=args.local_rank not in [-1, 0]
        ) as pbar:
            end = time.time()
            for step, batch in enumerate(train_loader):
                # Memory monitor for each step.
                torch.cuda.reset_peak_memory_stats()

                data_time.update(time.time() - end)
                batch = tuple(t.to(args.device) for t in batch)

                x, y = batch

                # Update active trainable parameters based on JointSparse policy.
                jointsparse_update_distribution = False
                if args.jointsparse and step % args.jointsparse_update_freq == 0:
                    if args.layerwiselr and not args.HeadLr10times:
                        coder.set_requires_grad_with_lr(model, optimizer)
                    else:
                        coder.set_requires_grad(model)
                    jointsparse_update_distribution = True

                if "resnet" in args.model_type:
                    pred = model(x)
                    loss = nn.CrossEntropyLoss()(pred, y)
                else:
                    loss = model(x, y)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                # Update probability distribution after gradients are available.
                if jointsparse_update_distribution:
                    coder.update_distribution(model)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                MB = 1024 * 1024
                memory_meter.update(torch.cuda.max_memory_allocated() / MB)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    losses.update(loss.item()*args.gradient_accumulation_steps)
                    # if args.fp16:
                    #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    # else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()

                    optimizer.zero_grad()
                    global_step += 1

                    pbar.set_postfix(
                        {
                            "loss": losses.avg,
                            "lr": scheduler.get_last_lr()[0],
                            "data_time": data_time.val,
                            "batch_time": batch_time.val,
                            "mem_alloc": memory_meter.val,
                            "avg_mem_alloc": memory_meter.avg,
                            "max_reserved": torch.cuda.max_memory_reserved() / MB,
                        }
                    )
                    pbar.update(1)
                    end = time.time()

                    if args.local_rank in [-1, 0]:
                        writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                        writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)
                        writer.add_scalar("train/mem",scalar_value=memory_meter.avg, global_step=global_step)
                    if global_step % args.eval_every == 0:
                        accuracy = valid(args, model, writer, val_loader, global_step, log)
                    if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                        save_model(args, model, log)
                        model.train()
                    torch.distributed.barrier()

                    if global_step % t_total == 0:
                        break

        losses.reset()
        if global_step % t_total == 0:
            break

    log.info("Final Accuracy: \t{}".format(accuracy))
    log.info("End Training!")
    if args.jointsparse:
        log.info("Final JointSparse probability vector: {}".format(coder.prob))
    if args.local_rank in [-1, 0]:
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True, help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "aircraft",
                                              "Pet37", "flowers", "stanford_car",
                                              "cub200", "food101"],
                        default="cifar10", help="Which downstream task.")
    parser.add_argument("--data", default="placeholder", help="Which downstream task.")
    parser.add_argument("--customSplit", default="", help="the downstream custom split.")

    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default=".", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for eval.")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local-rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    # for 2080 Ti, original setting
    # parser.add_argument("--local_rank", type=int, default=-1,
    #                     help="local_rank for distributed training on gpus")
    
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    # bitfit
    parser.add_argument('--bitfit', action="store_true", help="if employing bitfit")

    parser.add_argument('--new_backrazor', action="store_true", help="if employing the backrazor")
    parser.add_argument('--back_prune_ratio', type=float, default=0.8, help="the back prune ratio")
    parser.add_argument('--quantize', action="store_true", help="while pruning, also do the quantization")

    parser.add_argument('--fix_backbone', action="store_true", help="if fix the backbone")

    # profile
    parser.add_argument('--memory_cost_profile', action="store_true", help="profile memory cost")

    # fine-tune
    parser.add_argument('--HeadLr10times', action="store_true", help="increase head lr by 10 times")
    parser.add_argument('--train_resize_first', action="store_true", help="resize the image before "
                                                                          "doing other training augmentations")
    parser.add_argument('--cotuning_trans', action="store_true", help="Employ the trans of cotuning")
    parser.add_argument('--color-distort', action="store_true", help="Employ the color distort")

    # jointsparse
    parser.add_argument('--jointsparse', action="store_true", help="Employ the jointsparse")
    parser.add_argument('--joint_sparsity', type=float, default=0.5, help="the sparsity of jointsparse")
    parser.add_argument('--jointsparse_update_freq', type=int, default=20, help="the update freq of jointsparse")
    parser.add_argument('--prebackrazor', action="store_true", help="Employ the prebackrazor")
    parser.add_argument('--flashattn', action="store_true", help="Employ the flashattn")
    parser.add_argument('--layerwiselr', action="store_true", help="If Employ the layerwise lr")
    

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    # else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        args.local_rank = 0
    torch.cuda.set_device(args.local_rank)
    # args.local_rank = os.environ['LOCAL_RANK']
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         timeout=timedelta(minutes=60))
    args.n_gpu = torch.distributed.get_world_size()
    if args.train_batch_size % args.n_gpu != 0:
        raise ValueError("batch size of {} is not divisible by gpu number of {}".format(args.train_batch_size, args.n_gpu))
    args.train_batch_size = args.train_batch_size // args.n_gpu
    args.device = device

    # Setup logging
    """ Train the model """
    save_dir = os.path.join(args.output_dir, "checkpoints", args.name)
    writer = None
    if args.local_rank in [-1, 0]:
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs", args.name))

    log = logger(save_dir, log_name="log.txt", local_rank=args.local_rank)
    log.info("Process rank: {}, device: {}, n_gpu: {}, distributed training: {}, 16-bits training: {}".format
        (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))
    
    # Set seed
    set_seed(args)

    train_loader, val_loader, test_loader = get_loader(args)
    num_classes = len(np.unique(train_loader.dataset.targets))

    # Model & Tokenizer Setup
    args, model = setup(args, log, num_classes)

    if args.bitfit:
        assert not ("resnet" in args.model_type)
        for name, parameter in model.named_parameters():
            if "bias" in name or "head" in name:
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False

    if args.fix_backbone:
        assert not ("resnet" in args.model_type)
        for name, parameter in model.named_parameters():
            if "head" in name:
                print("fix_backbone, not freeze {}".format(name))
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False

    if args.new_backrazor and args.quantize:
        for name, module in model.named_modules():
            module.name = name
        ms.policy.deploy_on_init(model, 'ViT/policy_tiny-8bit.txt', verbose=print, override_verbose=False)
        model.to(args.device)

    log.info(str(model))

    activation_bits = 32
    memory_cost, memory_cost_dict = profile_memory_cost(model, input_size=(1, 3, 224, 224), require_backward=True,
                                                        activation_bits=activation_bits, trainable_param_bits=32,
                                                        head_only=args.fix_backbone,
                                                        frozen_param_bits=8, batch_size=args.train_batch_size)
    MB = 1024 * 1024
    log.info("memory_cost is {:.1f} MB, param size is {:.1f} MB, act_size each sample is {:.1f} MB".
             format(memory_cost / MB, memory_cost_dict["param_size"] / MB, memory_cost_dict["act_size"] / MB))

    log.info("Dataset {}, Train dataset len is {}, test dataset len is {}".
             format(args.dataset, len(train_loader.dataset), len(val_loader.dataset)))
    # Training
    # Prepare dataset
    train(args, model, train_loader, val_loader, test_loader, log, writer)


if __name__ == "__main__":
    main()
