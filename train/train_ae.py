# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""
import sys
sys.path.append('/home/rameez/work/motion-diffusion-model')

from external_models.modules.stsae import STSAE
from external_models.modules.stsgcn_wvlt import Model
import os
import json
from argparse import Namespace
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from training_loop_ae import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform, WandbPlatform  # required for the eval operation

from utils.model_util import get_model_args

def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    if args.resume_checkpoint != '':
        assert os.path.exists(args.save_dir), f"{args.save_dir} path does not exists"
        ckpt_name = args.resume_checkpoint.split("/")[-1]
        assert os.path.exists(args.resume_checkpoint), f"no checkpoint named {ckpt_name} in {args.save_dir}"
        cfg_path = os.path.join(args.save_dir, "args.json")
        with open(cfg_path, 'r') as f:
            old_args = json.load(f)
        args = Namespace(**old_args)
        if args.train_platform_type == "WandbPlatform":
            train_platform = train_platform_type(args.save_dir, resume="must", id=args.wandb_id)
        else:
            train_platform = train_platform_type(args.save_dir)
    else:
        train_platform = train_platform_type(args.save_dir)
        if args.train_platform_type == "WandbPlatform":
            args.wandb_id = train_platform.get_run_id() # For resuming
        train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames)

    print("creating model and diffusion...")
    # model = STSAE(c_in=3,
    #               h_dim=32, 
    #               latent_dim=512, 
    #               n_frames=30, 
    #               n_joints=22
    #             ).cuda()
    model = Model(3,30, 0.1,4,[3, 3],0.0).cuda()
    # Print number of parameters
    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Training...")
    TrainLoop(args, train_platform, model, data).run_loop()
    # train_platform.close()


if __name__ == "__main__":
    main()
