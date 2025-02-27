# from model.mdm import MDM
# !Luca: commented out for motion_condition_batch
# from model.mdm_w_motion import MDM



from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps


def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model_and_diffusion(args, data):
    if args.condition == "motion":
        from model.mdm_w_motion import MDM
    else:
        from model.mdm import MDM
    model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    if args.unconstrained:
        cond_mode = 'no_cond'
    elif args.dataset in ['kit', 'humanml']:
        cond_mode = 'text'
    else:
        cond_mode = 'action'
        enc_type = args.enc_type
        
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1
    
    enc_type = None
    cond_frames = None
    hidden_dim = None
    if args.condition == 'motion':
        cond_mode = args.condition
        enc_type = args.enc_type
        cond_frames = args.cond_frames
        hidden_dim = args.hidden_dim
        reco = True if args.lambda_reco > 0. else False
        
    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1
    elif args.dataset == "h36m":
        # H36M_xyz
        data_rep = 'xyz'
        njoints = 22
        nfeats = 3
    
    modeltype = ''
    translation = False if args.dataset == "h36m" else True # True # Cambio a False altrimenti ha impatto sul tensore di input in
    glob = True
    glob_rot = True
    dropout = 0.1
    activation = "gelu"
    ff_size = 1024
    num_heads = 4
    
    
    model_args = {'modeltype': modeltype, 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
                  'translation': translation, 'pose_rep': data_rep, 'glob': glob, 'glob_rot': glob_rot,
                  'latent_dim': args.latent_dim, 'ff_size': ff_size, 'num_layers': args.layers, 'num_heads': num_heads,
                  'dropout': dropout, 'activation': activation, 'data_rep': data_rep, 'cond_mode': cond_mode,
                  'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
                  'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset,
                  'enc_type': enc_type, 'cond_frames': cond_frames, 'hidden_dim': hidden_dim, 'reco': reco}
    if args.dataset == "h36m":
        return model_args
    else:
        return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
                'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
                'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
                'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset}


def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    data_rep = 'xyz' if args.dataset == "h36m" else 'rot6d'
    
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        data_rep=data_rep,
        lambda_vel_rcxyz=args.lambda_vel_rcxyz,
        DCT_coeffs = args.DCT_coeffs,
        lambda_smooth=args.lambda_smooth,
        lambda_reco = args.lambda_reco
    )