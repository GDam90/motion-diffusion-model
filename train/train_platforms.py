import os
import wandb as wb

class TrainPlatform:
    def __init__(self, save_dir):
        pass

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass


class ClearmlPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from clearml import Task
        path, name = os.path.split(save_dir)
        self.task = Task.init(project_name='motion_diffusion',
                              task_name=name,
                              output_uri=path)
        self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name):
        self.logger.report_scalar(title=group_name, series=name, iteration=iteration, value=value)

    def report_args(self, args, name):
        self.task.connect(args, name=name)

    def close(self):
        self.task.close()


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def close(self):
        self.writer.close()

class WandbPlatform(TrainPlatform):
    def __init__(self, save_dir):
        name = os.path.split(save_dir)[-1]
        self.project_name = "motion_diffusion"
        self.group = "guido_exp"
        self.entity = "pinlab-sapienza"
        self.name = name
        wb.init(
            project=self.project_name,
            entity=self.entity,
            group=self.group,
            name=self.name,
            # config=self
        )
        
    def report_scalar(self, name, value, iteration, group_name=None):
        wb.log({f'{group_name}/{name}' : value}, step=iteration)
    
    def close(self):
        wb.finish()

    def report_args(self, args, name):
        wb.config.update(args)

class NoPlatform(TrainPlatform):
    def __init__(self, save_dir):
        pass
