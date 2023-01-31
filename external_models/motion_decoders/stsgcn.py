import torch.nn as nn

from external_models.modules import stsgcn

class Decoder(nn.Module):
    def __init__(self, c_out, h_dim, n_frames, n_joints, dropout) -> None:
        super().__init__()
        
        self.model = nn.ModuleList()

        self.model.append(stsgcn.ST_GCNN_layer(h_dim,128,[1,1],1,n_frames,
                                           n_joints,dropout))
        self.model.append(stsgcn.ST_GCNN_layer(128,64,[1,1],1,n_frames,
                                               n_joints,dropout))
            
        self.model.append(stsgcn.ST_GCNN_layer(64,128,[1,1],1,n_frames,
                                               n_joints,dropout))
                                               
        self.model.append(stsgcn.ST_GCNN_layer(128,c_out,[1,1],1,n_frames,
                                               n_joints,dropout))  
        
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        '''
        input shape: [BatchSize, h_dim, n_frames, n_joints]
        output shape: [BatchSize, in_Channels, n_frames, n_joints]
        '''
        return self.model(x)


class STS_Decoder(nn.Module):
    def __init__(self, c_in, h_dim=32, latent_dim=512, n_frames=12, n_joints=18, **kwargs) -> None:
        super(STS_Decoder, self).__init__()

        dropout = kwargs.get('dropout', 0.3)

        self.decoder = Decoder(c_out=c_in, h_dim=h_dim, n_frames=n_frames, n_joints=n_joints, dropout=dropout)
        
        self.rev_btlnk = nn.Linear(in_features=latent_dim, out_features=h_dim * n_frames * n_joints)
    
    def decode(self, z, input_shape):
        # assert len(input_shape) == 4
        N, V, C, T = input_shape
        C = 32
        z = self.rev_btlnk(z)
        z = z.view(N, C, T, V).contiguous()
        z = self.decoder(z)

        # !Luca: Return the original shape
        z = z.permute(0, 3, 1, 2).contiguous()
        
        return z
        
    def forward(self, x, x_shape=None):
        x = self.decode(x, x_shape)
        
        return x