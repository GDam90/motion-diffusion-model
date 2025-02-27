"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import sys
sys.path.append("/media/hdd/guide/motion-diffusion-model")

import json
import os
import torch
import re

from argparse import Namespace
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader, get_h36m_test_sets
from eval.a2m.tools import save_metrics
from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_model_wo_clip


def evaluate(args, model, diffusion, data):
    scale = None
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
        scale = {
            'action': torch.ones(args.batch_size) * args.guidance_param,
        }
    model.to(dist_util.dev())
    model.eval()  # disable random masking


    folder, ckpt_name = os.path.split(args.model_path)
    if args.dataset == "humanact12":
        from eval.a2m.gru_eval import evaluate
        eval_results = evaluate(args, model, diffusion, data)
    elif args.dataset == "uestc":
        from eval.a2m.stgcn_eval import evaluate
        eval_results = evaluate(args, model, diffusion, data)
    elif args.dataset == "h36m":
        from eval.p2m.pose2motion_evaluate import evaluate_copy_from_stgcneval  # [TODO]
        eval_results = evaluate_copy_from_stgcneval(args, model, diffusion, data)
    else:
        raise NotImplementedError("This dataset is not supported.")

    # save results
    iter = int(re.findall('\d+', ckpt_name)[0])
    scale = 1 if scale is None else scale['action'][0].item()
    scale = str(scale).replace('.', 'p')
    metricname = "evaluation_results_iter{}_samp{}_scale{}_a2m.yaml".format(iter, args.num_samples, scale)
    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, eval_results)

    return eval_results


def update_dict(dict_toupdate, old_dict):
    keys = ["enc_type", "cond_frames","hidden_dim", "condition", "n_frames"]
    for key in keys:
        dict_toupdate[key] = old_dict[key]
    return dict_toupdate


def main():
    args = evaluation_parser()
    
    ###
    args_dict = args.__dict__
    ckpt_dir, name = os.path.split(args.model_path)
    cfg_path = os.path.join(ckpt_dir, "args.json")
    with open(cfg_path, 'r') as f:
        old_args = json.load(f)
    args_dict = update_dict(args_dict, old_args)
    
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    
    args = Namespace(**args_dict)

    print(f'Eval mode [{args.eval_mode}]')
    assert args.eval_mode in ['debug', 'full'], f'eval_mode {args.eval_mode} is not supported for dataset {args.dataset}'
    if args.eval_mode == 'debug':
        args.num_samples = 10
        args.num_seeds = 1
    else:
        args.num_samples = 1000
        args.num_seeds = 1

    if args.dataset == 'h36m':
        data_loader = get_h36m_test_sets(num_frames=60, act='phoning')['phoning'] # It serves only as a prototype
    else:
        data_loader = get_dataset_loader(name=args.dataset, num_frames=60, batch_size=args.batch_size,)
    
    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data_loader)
    args.cond_mode = model.cond_mode # [ADDED TO FIX A BUG]
    
    
    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    eval_results = evaluate(args, model, diffusion, data_loader.dataset)
    if not args.dataset == 'h36m':
        fid_to_print = {k : sum([float(vv) for vv in v])/len(v) for k, v in eval_results['feats'].items() if 'fid' in k and 'gen' in k}
        print(fid_to_print)

if __name__ == '__main__':
    main()
