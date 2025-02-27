from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    elif name == "h36m":
        from data_loaders.human36m.dataset_h36m import H36M_Dataset as H36M
        return H36M
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train'):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train'):
    dataset = get_dataset(name, num_frames, split, hml_mode)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader

def get_h36m_test_sets(num_frames, act=None, datasets=False, valid=False):
    '''
    return a dict of datasets, indexed with action names.
    '''
    from data_loaders.human36m.dataset_h36m import H36M_Dataset_test
    dataset = H36M_Dataset_test
    collate = all_collate
    testloaders = {}
    actions = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog",
               "walkingtogether"]
    split = 'test' if not valid else 'val'
    if act in actions:
        single_dataset = dataset(split=split, num_frames=num_frames, actions=[act])
        if datasets:
            return {act: single_dataset}
        loader = DataLoader(
        single_dataset, batch_size=256, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate)
        return {act: loader}
        
    
    for action in actions:
        tmp_dataset = dataset(split=split, num_frames=num_frames, actions=[action])
        if datasets:
            testloaders.update({action: tmp_dataset})
        else:
            loader = DataLoader(
            tmp_dataset, batch_size=256, shuffle=True,
            num_workers=4, drop_last=True, collate_fn=collate
        )
            testloaders.update({action: loader})
        
    return testloaders