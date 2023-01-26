import torch.nn as nn

from external_models.motion_encoders.stsgcn import Encoder
from external_models.motion_decoders.stsgcn import Decoder
        

class STSAE(nn.Module):
    def __init__(self, c_in, h_dim=32, latent_dim=512, n_frames=12, n_joints=18, **kwargs) -> None:
        super(STSAE, self).__init__()
        
        dropout = kwargs.get('dropout', 0.3)

        self.encoder = Encoder(c_in, h_dim, n_frames, n_joints, dropout)
        self.decoder = Decoder(c_out=c_in, h_dim=h_dim, n_frames=n_frames, n_joints=n_joints, dropout=dropout)
        
        self.btlnk = nn.Linear(in_features=h_dim * n_frames * n_joints, out_features=latent_dim)
        self.rev_btlnk = nn.Linear(in_features=latent_dim, out_features=h_dim * n_frames * n_joints)
        

    def encode(self, x, return_shape=False):
        assert len(x.shape) == 4
        x = x.unsqueeze(4)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()
            
        x = self.encoder(x)
        N, C, T, V = x.shape
        x = x.view([N, -1]).contiguous()
        x = x.view(N, M, 64, T, V).permute(0, 2, 3, 4, 1)
        x_shape = x.size()
        x = x.view(N, -1) 
        x = self.btlnk(x)
        
        if return_shape:
            return x, x_shape
        return x
    
    def decode(self, z, input_shape):
        # assert len(input_shape) == 4
        
        z = self.rev_btlnk(z)
        N, C, T, V, M = input_shape
        z = z.view(input_shape).contiguous()
        z = z.permute(0, 4, 1, 2, 3).contiguous()
        z = z.view(N * M, C, T, V)

        z = self.decoder(z)
        
        return z
        
    def forward(self, x):
        x, x_shape = self.encode(x, return_shape=True)
        x = self.decode(x, x_shape)
        
        return x
