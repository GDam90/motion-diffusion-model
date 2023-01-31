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
    def __init__(self, save_dir, resume=None, id=None):
        
        self.resume = resume
        
        name = os.path.split(save_dir)[-1]
        self.project_name = "motion_diffusion"
        self.group = "luca_exp"
        self.entity = "pinlab-sapienza"
        if "guide" in os.path.realpath(save_dir):
            group = "guido_exp"
        elif "edo" in os.path.realpath(save_dir):
            group = "edo_exp"
        elif "luca_s" in os.path.realpath(save_dir):
            group = "luca_exp"
        elif "ram" in os.path.realpath(save_dir):
            group = "ram_exp"
        else:
            print("no folder with this name")
            raise SystemExit("no folder with this name, exiting...")
        self.group = group
        self.name = name
        self.run = wb.init(
            project=self.project_name,
            entity=self.entity,
            group=self.group,
            name=self.name,
            resume=resume,
            id=id
            )
        
    def report_scalar(self, name, value, iteration, group_name=None):
        wb.log({f'{group_name}/{name}' : value}, step=iteration)

    def close(self):
        wb.finish()

    def report_args(self, args, name):
        wb.config.update(args)
    
    def get_run_id(self):
        return self.run.id
    
    def get_run_config(self):
        '''
        get_run_config method returns the config attribute of the 
        wandb.run.Run object which is a dictionary of the command
        line arguments and environment variables passed to the run.
        '''
        return self.run.config

    def get_run_summary(self):
        '''
        get_run_summary method returns the summary attribute of 
        the wandb.run.Run object which is a dictionary of the 
        summary metrics that have been logged for the current run.
        '''
        return self.run.summary

    def get_run_step(self):
        '''
        get_run_step method returns the step attribute of the 
        wandb.run.Run object which is the global step of the 
        training process (if the training is using steps).
        '''
        return self.run.step

    def get_run_project(self):
        '''
        get_run_project method returns the project attribute 
        of the wandb.run.Run object which is the name of the 
        project that the run belongs to.
        '''
        return self.run.project

class NoPlatform(TrainPlatform):
    def __init__(self, save_dir):
        pass
