
from torch.utils.data import Dataset
import numpy as np
import data_loaders.human36m.data_utils as data_utils
import torch

'''
adapted from https://github.com/wei-mao-2019/HisRepItself/blob/master/utils/h36motion3d.py
'''


class H36M_Dataset(Dataset):

    def __init__(self, split, num_frames, skip_rate=1, actions=None, data_dir='/media/hdd/guide/motion-diffusion-model/dataset/Human36Millions'):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        assert split in ['train', 'val', 'test'], "split should be one of ['train', 'val', 'test']"
        split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        assert num_frames == 60, "for debugging let's leave 60 frames with 30 and 30 for inp and out"
        input_n = 30
        output_n = 60
        self.path_to_data = data_dir
        self.split = split
        self.split_n = split_dict[split]
        self.in_n = input_n
        self.out_n = output_n
        self.sample_rate = 2
        self.p3d = {}
        self.data_idx = []
        self.num_frames = num_frames + 30
        subs = [[1, 6, 7, 8, 9], [11], [5]]
        # acts = data_utils.define_actions(actions)
        if actions is None:
            acts = ["walking", "eating", "smoking", "discussion", "directions",
                    "greeting", "phoning", "posing", "purchases", "sitting",
                    "sittingdown", "takingphoto", "waiting", "walkingdog",
                    "walkingtogether"]
        else:
            acts = actions
        self.num_actions = len(acts)
        # subs = np.array([[1], [11], [5]])
        # acts = ['walking']
        # 32 human3.6 joint name:
        joint_names = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Site", "LeftUpLeg", "LeftLeg",
                      "LeftFoot",
                      "LeftToeBase", "Site", "Spine", "Spine1", "Neck", "Head", "Site", "LeftShoulder", "LeftArm",
                      "LeftForeArm",
                      "LeftHand", "LeftHandThumb", "Site", "L_Wrist_End", "Site", "RightShoulder", "RightArm",
                      "RightForeArm",
                      "RightHand", "RightHandThumb", "Site", "R_Wrist_End", "Site"]

        subs = subs[self.split_n]
        key = 0
        self.act_dict = {} # ADDED TO SAVE THE ACTION
        for subj in subs:
            for action_idx in np.arange(len(acts)):
                action = acts[action_idx]
                self.act_dict[action_idx] = action
                if self.split_n <= 1:
                    for subact in [1, 2]:  # subactions
                        #print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                        filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, subact)
                        the_sequence = data_utils.readCSVasFloat(filename)
                        n, d = the_sequence.shape
                        even_list = range(0, n, self.sample_rate)
                        n_frames = len(even_list)
                        the_sequence = np.array(the_sequence[even_list, :])
                        the_sequence = torch.from_numpy(the_sequence).float().cuda()
                        # remove global rotation and translation
                        the_sequence[:, 0:6] = 0
                        p3d = data_utils.expmap2xyz_torch(the_sequence) # p3d IS FOR pOSE3d
                        # self.p3d[(subj, action, subact)] = p3d.view(n_frames, -1).cpu().data.numpy()
                        self.p3d[key] = p3d.view(n_frames, -1).cpu().data.numpy()

                        valid_frames = np.arange(0, n_frames - self.num_frames + 1, skip_rate)

                        # tmp_data_idx_1 = [(subj, action, subact)] * len(valid_frames)
                        tmp_data_idx_1 = [key] * len(valid_frames)
                        tmp_data_idx_2 = list(valid_frames)
                        act_idx_3 = [action_idx] * len(valid_frames)
                        # self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2, act_idx_3)) # ASSOCIATE MOTION TO THE ACTION (prev line is original)
                        key += 1
                else:
                    #print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 1)
                    the_sequence1 = data_utils.readCSVasFloat(filename)
                    n, d = the_sequence1.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames1 = len(even_list)
                    the_sequence1 = np.array(the_sequence1[even_list, :])
                    the_seq1 = torch.from_numpy(the_sequence1).float().cuda()
                    the_seq1[:, 0:6] = 0
                    p3d1 = data_utils.expmap2xyz_torch(the_seq1)
                    # self.p3d[(subj, action, 1)] = p3d1.view(num_frames1, -1).cpu().data.numpy()
                    self.p3d[key] = p3d1.view(num_frames1, -1).cpu().data.numpy()

                    #print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 2)
                    the_sequence2 = data_utils.readCSVasFloat(filename)
                    n, d = the_sequence2.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames2 = len(even_list)
                    the_sequence2 = np.array(the_sequence2[even_list, :])
                    the_seq2 = torch.from_numpy(the_sequence2).float().cuda()
                    the_seq2[:, 0:6] = 0
                    p3d2 = data_utils.expmap2xyz_torch(the_seq2)

                    # self.p3d[(subj, action, 2)] = p3d2.view(num_frames2, -1).cpu().data.numpy()
                    self.p3d[key + 1] = p3d2.view(num_frames2, -1).cpu().data.numpy()

                    # print("action:{}".format(action))
                    # print("subact1:{}".format(num_frames1))
                    # print("subact2:{}".format(num_frames2))
                    fs_sel1, fs_sel2 = data_utils.find_indices_256(num_frames1, num_frames2, self.num_frames,
                                                                   input_n=self.in_n)

                    valid_frames = fs_sel1[:, 0]
                    tmp_data_idx_1 = [key] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2)) # TODO ADD ACTION IF NEEDED

                    valid_frames = fs_sel2[:, 0]
                    tmp_data_idx_1 = [key + 1] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2)) # TODO ADD ACTION IF NEEDED
                    key += 2

        assert self.num_actions == len(self.act_dict)
        # ignore constant joints and joints at same position with other joints
        joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
        dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        self.dimensions_to_use = np.setdiff1d(np.arange(96), dimensions_to_ignore)
        
    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        # key, start_frame = self.data_idx[item] # ORIGINAL
        key, start_frame, action_idx = self.data_idx[item] # TO RETRIEVE THE ACTION
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        motion = self.p3d[key][fs]
        motion = motion[:, self.dimensions_to_use]
        motion = motion.reshape(motion.shape[0], 3, motion.shape[1]//3)
        motion = motion.transpose(2,1,0)
        motion = torch.tensor(motion)
        output = {} # Dict for {mask, length, action, action_text}
        act_idx = torch.tensor(action_idx) # action
        action = self.act_dict[action_idx] # action_text
        output['inp'] = motion
        output['action'] = act_idx
        output['action_text'] = action
        #print (self.p3d[key][fs].shape)
        return output

