# !Luca: added 
import sys
sys.path.append('/media/hdd/luca_s/code/DDPMotion/motion-diffusion-model')

import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from eval import eval_humanml, eval_humanact12_uestc
from data_loaders.get_data import get_dataset_loader


from utils.reconstruction_loss import mpjpe_error


class TrainLoop:
    def __init__(self, args, train_platform, model, data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model

        # !Luca: to remove
        # self.diffusion = diffusion
        # self.cond_mode = model.cond_mode

        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint

        # !Luca: to remove
        # self.use_fp16 = False  # deprecating this option
        # self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()

        # !Luca: to remove
        # self.mp_trainer = MixedPrecisionTrainer(
        #     model=self.model,
        #     use_fp16=self.use_fp16,
        #     fp16_scale_growth=self.fp16_scale_growth,
        # )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.model.parameters() ,lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        # !Luca: to remove
        # self.schedule_sampler_type = 'uniform'
        # self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None

        self.use_ddp = False
        self.ae_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):

        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for motion, cond in tqdm(self.data): # motion [bs, joints, channels, frames]
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
                batch = cond['y']['motion_condition']

                for i in range(0, batch.shape[0], self.microbatch):
                    # Eliminates the microbatch feature
                    assert i == 0
                    assert self.microbatch == self.batch_size
                    micro = batch
                    output = self.model(micro)
                    loss = mpjpe_error(output, micro)
                    self.train_platform.report_scalar(name='reconstruction', value=loss, iteration=self.step, group_name='Loss')
                    loss.backward()
                    self.opt.step()

                self.step += 1
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint and evaluate every 10 epochs
            if epoch % 10 == 0:
                # save model
                torch.save(self.save_dir, self.model.state_dict())
            # *Luca: add evaluation here   

        

    # def run_step(self, batch, cond):
    #     self.forward_backward(batch, cond)
    #     self.opt.step()
    #     self._anneal_lr()
    #     self.log_step()

    # def forward_backward(self, batch, cond):
    #     self.opt.zero_grad()
    #     for i in range(0, batch.shape[0], self.microbatch):
    #         # Eliminates the microbatch feature
    #         assert i == 0
    #         assert self.microbatch == self.batch_size
    #         micro = batch
    #         micro_cond = cond
    #         last_batch = (i + self.microbatch) >= batch.shape[0]
    #         t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

    #         compute_losses = functools.partial(
    #             self.diffusion.training_losses,
    #             self.ae_model,
    #             micro,  # [bs, ch, image_size, image_size]
    #             t,  # [bs](int) sampled timesteps
    #             model_kwargs = micro_cond,
    #             dataset = self.data.dataset
    #         )

    #         if last_batch or not self.use_ddp:
    #             losses = compute_losses()
    #         else:
    #             with self.ae_model.no_sync():
    #                 losses = compute_losses()

    #         if isinstance(self.schedule_sampler, LossAwareSampler):
    #             self.schedule_sampler.update_with_local_losses(
    #                 t, losses["loss"].detach()
    #             )

    #         loss = (losses["loss"] * weights).mean()
    #         self.mp_trainer.backward(loss)

    # def _anneal_lr(self):
    #     if not self.lr_anneal_steps:
    #         return
    #     frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
    #     lr = self.lr * (1 - frac_done)
    #     for param_group in self.opt.param_groups:
    #         param_group["lr"] = lr

    # def log_step(self):
    #     logger.logkv("step", self.step + self.resume_step)
    #     logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"


    # def save(self):
    #     def save_checkpoint(params):
    #         state_dict = self.mp_trainer.master_params_to_state_dict(params)

    #         # Do not save CLIP weights
    #         clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
    #         for e in clip_weights:
    #             del state_dict[e]

    #         logger.log(f"saving model...")
    #         filename = self.ckpt_file_name()
    #         with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
    #             torch.save(state_dict, f)

    #     save_checkpoint(self.mp_trainer.master_params)

    #     with bf.BlobFile(
    #         bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
    #         "wb",
    #     ) as f:
    #         torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None

