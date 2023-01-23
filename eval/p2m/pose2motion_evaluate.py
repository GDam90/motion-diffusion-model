from utils import dist_util
from utils.fixseed import fixseed
from data_loaders.get_data import get_h36m_test_sets
from data_loaders.human36m.dataset_h36m import NewDataloaderH36M
from torch.utils.data import DataLoader
import functools
from eval.a2m.stgcn_eval import NewDataloader
import copy
from data_loaders.tensors import collate
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class MPJPEEvaluator:
    def __init__ (self, dataname, parameters, device, seed=None):
        
        assert dataname in ['h36m'], 'only defined for h36m dataset'
        self.path, _ = os.path.split(parameters["model_path"])
        self.num_classes = parameters["num_classes"]
        self.channels = parameters["nfeats"]
        self.batch_size = parameters["batch_size"]
        self.num_frames = parameters["n_frames"]
        self.njoints = parameters["njoint"]
        self.layout = parameters["pose_rep"]
        self.h36m_standard_joints = 32
        self.dataname = dataname
        self.device = device
        self.testing_frames = [1, 3, 7, 9, 13, 17, 21, 24]
        self.actions = [
            "walking", "eating", "smoking", "discussion", "directions",
            "greeting", "phoning", "posing", "purchases", "sitting",
            "sittingdown", "takingphoto", "waiting", "walkingdog",
            "walkingtogether"
        ]
        self.dim_used = [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 
                         17, 18, 19, 21, 22, 25, 26, 27, 29, 30]
        self.joint_to_ignore = [16, 20, 23, 24, 28, 31]
        self.joint_equal = [13, 19, 22, 13, 27, 30]
        
    def _h36m_format(self, motion_batch, for_viz=False):
        '''
        convert 22joints motion tensor in standard 32joints for visualization
        '''
        standard_shape = [self.batch_size, self.num_frames, self.channels, self.h36m_standard_joints]
        new_motion = torch.zeros(standard_shape).to(self.device)
        x = motion_batch.permute(0, 3, 2, 1)
        new_motion[:, :, :, self.dim_used] = x
        new_motion[:, :, :, self.joint_to_ignore] = new_motion[:, :, :, self.joint_equal]
        if for_viz:
            new_motion = new_motion.permute(0, 1, 3, 2)
        return new_motion
        
        
    def evaluate(self, loaders):
        
        error_mean_all_actions = torch.zeros(len(self.testing_frames)).to(self.device)
        
        ########## DEBUGGING #########
        loaders_ = {'gt': loaders, 'gen': loaders}
        loaders = loaders_
        ##############################
        for action in self.actions:
            action_gt_loader = loaders["gt"][action]
            action_gen_loader = loaders["gen"][action]
            
            # imposterei tutto per la single batch (no for loop)
            assert len(action_gt_loader) == len(action_gen_loader) == 1, "a single batch with 250 samples should be provided during test"
            gt_motion = next(iter(action_gt_loader))['output_xyz'] # [bs, njoints, channel, nframes]
            gen_motion = next(iter(action_gen_loader))['output_xyz'] # [bs, njoints, channel, nframes]
            gt_motion = gt_motion.permute(0, 3, 1, 2) # [bs, nframes, njoints, channels]
            gen_motion = gen_motion.permute(0, 3, 1, 2) # [bs, nframes, njoints, channels]
            
            # for batch in test_loader:
            with torch.no_grad():

                gt_motion = gt_motion.to(self.device)
                gen_motion = gen_motion.to(self.device)
                batch_dim = gt_motion.shape[0]
                assert batch_dim == self.batch_size

                # sequences_gt_3d = sequences_gt.reshape(-1, args.model_config.output_n, 22, 3)
                # sequences_predict_3d = sequences_predict.reshape(-1, args.model_config.output_n, 22, 3)
                                        
                errors_per_action_at_frame = torch.sum(torch.mean(torch.norm(gt_motion[:, self.testing_frames] - gen_motion[:, self.testing_frames], dim=3), dim=2), dim=0)
                error_mean_all_actions += errors_per_action_at_frame
                errors_per_action_at_frame /= self.batch_size
                for c in range(len(self.testing_frames)):
                    print(f"MPJPE/{action}/{self.testing_frames[c]} : {errors_per_action_at_frame[c]}")
                
            error_mean_all_actions /= self.num_classes * self.batch_size
            for c in range(len(self.testing_frames)):
                print(f"MPJPE/MEAN/{self.testing_frames[c]} : {error_mean_all_actions[c]}")

            return errors_per_action_at_frame, error_mean_all_actions
        
    def _init_plot(self):
        fig = plt.figure()
        ax = Axes3D(fig) # , auto_add_to_figure=False)
        fig.add_axes(ax)
        ax.view_init(elev=20, azim=-40)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(loc='lower left')

        ax.set_xlim3d([-1, 1.5])
        ax.set_xlabel('X')

        ax.set_ylim3d([-1, 1.5])
        ax.set_ylabel('Y')

        ax.set_zlim3d([0.0, 1.5])
        ax.set_zlabel('Z')
        
        return fig, ax
    
    def create_pose(self, ax, plots, vals, pred=True, update=False):
        connect = [
				(15,14), (14,13), (13,25), (25,26), (26,27), (27,29), (27,30),
				(13,17), (17,18), (18,19), (19,22), (19,21), (13,12), (12,7), (7,8),
				(8,10), (8,9), (12,2), (2,3), (3,4), (3,5)]
        # connect = [
        #     (1, 2), (2, 3), (3, 4), (4, 5),
        #     (6, 7), (7, 8), (8, 9), (9, 10),
        #     (0, 1), (0, 6),
        #     (6, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
        #     (1, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        #     (24, 25), (24, 17),
        #     (24, 14), (14, 15)
    # ]
        
        LR = [
            False, True, True, True, True,
            True, False, False, False, False,
            False, True, True, True, True,
            True, True, False, False, False,
            False, False, False, False, True,
            False, True, True, True, True,
            True, True
    ]
        
        I   = np.array([touple[0] for touple in connect])
        J   = np.array([touple[1] for touple in connect])
    # Left / right indicator
        LR  = np.array([LR[a] or LR[b] for a,b in connect])
        if pred:
            lcolor = "#9b59b6"
            rcolor = "#2ecc71"
        else:
            lcolor = "#8e8e8e"
            rcolor = "#383838"

        for i in np.arange( len(I) ):
            x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
            z = np.array( [vals[I[i], 1], vals[J[i], 1]] )
            y = np.array( [vals[I[i], 2], vals[J[i], 2]] )
            if not update:

                if i ==0:
                    plots.append(ax.plot(x, y, z, lw=2,linestyle='--' ,c=lcolor if LR[i] else rcolor,label=['GT' if not pred else 'Pred']))
                else:
                    plots.append(ax.plot(x, y, z, lw=2,linestyle='--', c=lcolor if LR[i] else rcolor))

            elif update:
                plots[i][0].set_xdata(x)
                plots[i][0].set_ydata(y)
                plots[i][0].set_3d_properties(z)
                plots[i][0].set_color(lcolor if LR[i] else rcolor)
        
        return plots
    
    def update(self, num, data_gt, data_pred, plots_gt, plots_pred, fig, ax):
    
        gt_vals=data_gt[num]
        pred_vals=data_pred[num]
        plots_gt=self.create_pose(ax,plots_gt,gt_vals,pred=False,update=True)
        plots_pred=self.create_pose(ax,plots_pred,pred_vals,pred=True,update=True)
        
        

        
        
        r = 0.75
        xroot, zroot, yroot = gt_vals[0,0], gt_vals[0,1], gt_vals[0,2]
        ax.set_xlim3d([-r+xroot, r+xroot])
        ax.set_ylim3d([-r+yroot, r+yroot])
        ax.set_zlim3d([-r+zroot, r+zroot])
        #ax.set_title('pose at time frame: '+str(num))
        #ax.set_aspect('equal')
    
        return plots_gt,plots_pred
    
    def visualize(self, batch_gt, batch_gen):
        # h36 shaped motions
        scale = 0.001
        gt = self._h36m_format(batch_gt, for_viz=True) * scale
        gen = self._h36m_format(batch_gen, for_viz=True) * scale
        
        data_gt = gt[0].cpu().data.numpy()
        data_gen = gen[0].cpu().data.numpy()
        
        fig, ax = self._init_plot()
        gt_plots=[]
        gen_plots=[]
        vals = torch.zeros((32, 3)) ## Boh
        
        gt_plots = self.create_pose(ax, gt_plots, vals, pred=False, update=False)
        gen_plots = self.create_pose(ax, gen_plots, vals, pred=True, update=False)
        fargs = (
            data_gt,
            data_gen,
            gt_plots,
            gen_plots,
            fig, 
            ax
        )
        
        self.save_animation(fig, self.update, self.num_frames, fargs)
        pass
    
    def save_animation(self, figure, funcupdate, nframes, fargs):
        viz_path = os.path.join(self.path, "viz")
        os.makedirs(viz_path, exist_ok=True)
        line_anim = animation.FuncAnimation(figure, funcupdate, nframes, fargs=fargs, interval=70, blit=False)
        line_anim.save(os.path.join(viz_path, 'animation1.gif'),  fps=25, writer='pillow')
        plt.close()
        return
        
def evaluate(args, model, diffusion, data):
    pass

def evaluate_copy_from_stgcneval(args, model, diffusion, data):
    bs = args.batch_size
    args.num_classes = data.num_actions
    args.nfeats = 3
    args.njoint = 22
    device = dist_util.dev()


    recogparameters = args.__dict__.copy()
    recogparameters["pose_rep"] = "xyz"
    recogparameters["nfeats"] = 3
    
    mpjpevaluation = MPJPEEvaluator(args.dataset, recogparameters, device) # TODO
    mpjpe_metrics = {} # TODO
    
    # Instead of data_types, h36m'll test on different actions
    # data_types = ['train', 'test']
    data_types = data.all_acts
    
    # datasetGT now is a dictionary with keys in data_types (as before)
    # datasetGT = {'train': [data], 'test': [copy.deepcopy(data)]} 
    datasetGT = get_h36m_test_sets(num_frames=60, datasets=True)
    model.eval()

    allseeds = list(range(args.num_seeds))

    for index, seed in enumerate(allseeds):
        print(f"Evaluation number: {index + 1}/{args.num_seeds}")
        fixseed(seed)
        dataiterator = {key: [DataLoader(datasetGT[key], batch_size=bs, shuffle=False, num_workers=4, collate_fn=collate)
                                ] for key in data_types}
        
        new_data_loader = functools.partial(NewDataloaderH36M, model=model, diffusion=diffusion, device=device, # [PARTIAL FIX SOME ARGUMENTS]
                                        cond_mode=args.cond_mode, dataset=args.dataset, num_samples=args.num_samples)
        gtLoaders = {key: new_data_loader(mode="gt", dataiterator=dataiterator[key][0], action=key)
                    for key in data_types}
        batch_gt = next(iter(gtLoaders['walking']))['output_xyz']
        batch_gen = batch_gt.clone()
        
        mpjpevaluation.visualize(batch_gt=batch_gt, batch_gen=batch_gen)
        
        mpjpe_metrics[seed] = mpjpevaluation.evaluate(gtLoaders) ##########
        genLoaders = {key: new_data_loader(mode="gen", dataiterator=dataiterator[key][0], action=key)
                      for key in data_types}  
        
        loaders = {"gen": genLoaders,
                   "gt": gtLoaders}

        mpjpe_metrics[seed] = mpjpevaluation.evaluate(loaders)
