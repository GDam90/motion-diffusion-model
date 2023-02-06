# !Luca: added 
import sys
sys.path.append('/media/hdd/luca_s/code/DDPMotion/motion-diffusion-model')

import torch.nn as nn

from external_models.motion_encoders.stsgcn import Encoder
from external_models.motion_decoders.stsgcn import Decoder

import pdb
class STSAE(nn.Module):
    def __init__(self, c_in, h_dim=32, latent_dim=512, n_frames=30, n_joints=22) -> None:
        super(STSAE, self).__init__()
        
        self.c_in = c_in
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.n_frames = n_frames
        self.n_joints = n_joints

        dropout = 0.3

        self.encoder = Encoder(c_in, h_dim, n_frames, n_joints, dropout)
        self.decoder = Decoder(c_out=c_in, h_dim=h_dim, n_frames=n_frames, n_joints=n_joints, dropout=dropout)
        
        self.btlnk = nn.Linear(in_features=h_dim * n_frames * n_joints, out_features=latent_dim)
        self.rev_btlnk = nn.Linear(in_features=latent_dim, out_features=h_dim * n_frames * n_joints)
        

    def encode(self, x):
        assert len(x.shape) == 4
        N, V, C, T = x.size() # 64, 22, 3, 30

        # x = x.permute(0, 3, 1, 2).contiguous()
        # x = x.view(N, V, C, T).permute(0,2,3,1).contiguous()
        x = x.permute(0,2, 3, 1).contiguous() # [N, C, T, V]
            
        x = self.encoder(x)
        N, C, T, V = x.shape
        x_shape = x.size()
        x = x.view([N, -1]).contiguous()
        # x = x.view(N, C, T, V)
        # x_shape = x.size()
        # x = x.view(N, -1)
        x = self.btlnk(x)
        
        return x
    
    def decode(self, z):
        C = 32
        z = self.rev_btlnk(z)
        z = z.view(z.shape[0], C, self.n_frames, self.n_joints).contiguous() # N, C, T, V
        z = self.decoder(z)

        # !Luca: Return the original shape
        z = z.permute(0, 3, 1, 2).contiguous()
        
        return z
        
    def forward(self, x):
        pdb.set_trace()
        x = self.encode(x)
        x = self.decode(x)
        
        return x


# main
if __name__ == "__main__":

    import torch
    model = STSAE(c_in=3, h_dim=32, latent_dim=512, n_frames=30, n_joints=22)
    x = torch.randn(64, 22, 3, 30)
    x = model(x)
    print(x.shape)