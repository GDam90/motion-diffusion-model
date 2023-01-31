import torch

def mpjpe_error(output,micro): 
    
    batch_pred=output.contiguous().view(-1,3)
    batch_gt=micro.contiguous().view(-1,3)

    return torch.mean(torch.norm(batch_gt-batch_pred,2,1))  